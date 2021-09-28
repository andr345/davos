import torch
from torch import nn
import numpy as np
from torch.nn.modules.utils import _pair
from davos.lib import TensorList
from davos.eval.lib.preprocessing import sample_patch_multiscale, sample_patch_transformed, sample_patch
from collections import OrderedDict as odict
# noinspection PyUnresolvedReferences
from davos.lib.debug import DebugVis
from .common import Frame, uncrop, Memory, SpatialTracker


def get_tracker_class():
    return DLWL

warn_distractors_disabled = True

class DLWL(nn.Module):

    def __init__(self, object_id, params):
        super().__init__()
        self.params = params
        self.update_distractors = eval("self." + getattr(params, 'update_distractor_fn', 'update_distractors_wta'))
        self.ignore_distractors = getattr(params, 'ignore_distractors', False)

        self.object_id = object_id
        self.net = self.params.net
        self.device = next(self.net.parameters()).device
        self.spatial_tracker = SpatialTracker(params)
        self.memory = Memory(params)


    def make_distractors(self, labels, object_id, prev_distractors=None, disable=False):

        fg = (labels == object_id)

        if not disable:
            pdt = (prev_distractors.squeeze() != 0) if prev_distractors is not None else False
            dt = ~fg & ((labels != 0) | pdt)
        else:
            dt = torch.zeros_like(fg, dtype=torch.bool)

        m = torch.stack([fg, dt])
        m = m.view(-1, *m.shape[-3:]).float()

        return m

    def initialize(self, frame: Frame, prev_frame, bbox=None):

        self.cached_object_ids = None
        self.frame_num = 1

        # Set up mask and distractors
        prev_dt = prev_frame.segmentation.get(self.object_id, None) if prev_frame is not None else None
        mask = self.make_distractors(frame.labels, self.object_id, prev_dt, disable=self.ignore_distractors)
        frame.segmentation[self.object_id] = mask.squeeze(0)

        # Initialize target position and size
        self.spatial_tracker.initialize(bbox)

        # Initialize target model
        im = frame.image.unsqueeze(0).float()
        im_patches, feat_bbone, masks = self.generate_init_samples(im, mask)
        masks = masks.unsqueeze(1)  # Add seq dim
        self.init_target_model(feat_bbone, masks)

        return frame

    def track(self, frame, prev_frame):
        self.debug_info = {}

        tg_channel = 0

        assert self.object_id is not None
        obj_id = self.object_id

        self.frame_num += 1
        self.debug_info['frame_num'] = self.frame_num

        # Obtain the merged segmentation prediction for the previous frames.
        # This is used to update the target model and determine the search
        # region for the current frame
        im = frame.image.unsqueeze(0).float()
        prev_seg_prob = prev_frame.segmentation_raw[obj_id]

        # Update the target model with merged masks and data from the previous frame
        skip_update = False
        if self.frame_num > 2:
            if self.params.get('update_target_model', True):

                # Crop the segmentation mask for the previous search area
                prev_seg_prob_crop, _ = sample_patch(
                    prev_seg_prob, self.prev_pos, self.prev_scale * self.prev_im_size, self.prev_im_size,
                    mode=self.params.get('border_mode', 'replicate'), max_scale_change=self.params.get('patch_max_scale_change'),
                    is_mask=True)  # is_mask=True disables replication padding

                #bb = self.spatial_tracker.get_target_bbox(prev_seg_prob_crop[0,0], tlhw=True)
                # nnn = (prev_seg_prob[0,0] > 0.5).sum().item()
                # print(nnn)
                # if nnn < 8000:
                #     skip_update = True

                # Update the target model
                if not skip_update:
                    self.update_target_model(self.prev_feat_tm, prev_seg_prob_crop.clone())

        # Estimate target bbox
        tg_prob = prev_seg_prob[0, 0]  # Extract from first score channel
        if not skip_update:
            self.spatial_tracker.update_target_bbox_estimate(tg_prob)

        # Predict segmentation for the current frame

        # Get image patches and extract features
        pos, scale, im_size = self.spatial_tracker.get_centered_sample_pos()  # From target bbox estimate
        sample_coords, im_patches = self.extract_patches(im, pos, scale, im_size)
        feat_bbone = self.extract_backbone_features(im_patches)
        feat_tm = self.extract_target_model_features(feat_bbone)

        # Save data for merging, for the next frame update
        self.prev_pos, self.prev_scale, self.prev_im_size = self.spatial_tracker.get_centered_sample_pos()
        sample_pos, sample_scale = self.spatial_tracker.get_sample_location(sample_coords)
        self.prev_feat_tm = feat_tm

        # Segment the target
        seg_scores, mask_enc_pred = self.segment_target(feat_tm, feat_bbone, im_patches.shape[-2:], get_mask_enc=True)  # Raw scores, logits
        seg_scores = uncrop(seg_scores, sample_pos, sample_scale, im.shape[-2:], outside_value=-100.0)  # Uncrop to full image

        seg_prob = torch.sigmoid(seg_scores)    # Probability of being target/distractor at each pixel
        seg_mask = (seg_prob > 0.5).float()   # Binary segmentation mask

        # Get target box from the predicted segmentation
        tg_prob = seg_prob[0]  # Extract from first score channel
        new_bbox = torch.cat(self.spatial_tracker.get_target_bbox(tg_prob, tlhw=True))

        if hasattr(self, 'vis'):
            self.vis.imshow(im, "image")
            m = seg_mask
            im = torch.stack((m[1], m[0], torch.zeros_like(m[0])), dim=-3)
            self.vis.imshow(im, "mask (green), distractor (red)")
            self.vis.show_rawseg(seg_scores[0], title="predicted target")
            self.vis.show_rawseg(seg_scores[1], title="predicted distractors")
            self.vis.show_enclb(mask_enc_pred, title="predicted encoded mask")
            pass

        assert self.object_id is not None  # Multi-object mode

        # Save output

        frame.segmentation[obj_id] = seg_mask.view(*seg_mask.shape[-3:]).cpu()
        frame.target_bbox[obj_id] = new_bbox.tolist()
        frame.segmentation_raw[obj_id] = seg_scores.cpu()

        return frame

    def update_distractors_wta_clean(self, L, S, I):
        """
        :param L:  Label map (byte-image with object ids)
        :param S:  Target segmentation probabilities (0.0 - 1.0)
        :param I:  List of object ids, matching order in S
        """

        p_max = S.max(dim=0, keepdim=True).values
        p_min = S.min(dim=0, keepdim=True).values

        D = []
        If = (L != 0).float()
        for k, i in enumerate(I):
            Id = (L != i).float() * If
            d_i = Id * p_max + (1 - If) * p_min()
            D.append(d_i)

        return D

    def update_distractors_passthrough(self, object_ids, seg_probs: torch.Tensor, dis_scores: torch.Tensor, merged_labels):
        return dis_scores

    def update_distractors_zero(self, object_ids, seg_probs: torch.Tensor, dis_scores: torch.Tensor, merged_labels):
        return torch.zeros_like(dis_scores)

    def update_distractors_wta(self, object_ids, seg_probs: torch.Tensor, dis_scores: torch.Tensor, merged_labels):
        """ Generate distractors with a winner-takes-all approach.

        For target i, let the distractors be max(seg_scores[j]) for all j != i,
        in the area marked as occupied by any target in merged_labels.
        The distractor background will be min(seg_scores[k]) for all k,
        in the area marked as background in merged_labels.

        :param object_ids:  List of object ids, WITHOUT the background
        :param seg_probs:  Target segmentation probabilities (0.0 - 1.0), INCLUDING the background in [...,0,:,:]
        :param dis_scores:  Distractor scores, (0.0 - 1.0) INCLUDING the background in [...,0,:,:]
        :param merged_labels:  Label map (byte-image with object ids)
        """
        if dis_scores is None:
            return None

        dt_probs = torch.zeros_like(seg_probs)
        if self.ignore_distractors:
            global warn_distractors_disabled
            if warn_distractors_disabled:
                print("distractors are disabled")
                warn_distractors_disabled = False
            return dt_probs

        lb = torch.from_numpy(merged_labels).to(seg_probs.device)
        fg = (lb != 0)
        bg = (~fg).float()

        max_prob = seg_probs[1:].max(dim=0, keepdim=True).values
        min_prob = seg_probs[1:].min(dim=0, keepdim=True).values

        for i, obj_id in enumerate(object_ids):
            dt = ((lb != int(obj_id)) & fg).unsqueeze(0).float()  # Distractor mask
            dt_probs[i+1] = max_prob * dt + min_prob * bg

        return dt_probs

    def _merge_sigmoid(self, object_ids, seg_scores):

        seg_scores, dis_scores = torch.split(seg_scores, 1, dim=-3)

        # Obtain seg. probability and scores for background label
        eps = 1e-7
        seg_prob = torch.sigmoid(seg_scores)
        bg_prob = torch.prod(1 - seg_prob, dim=0, keepdim=True).clamp(eps, 1.0 - eps)
        bg_score = (bg_prob / (1.0 - bg_prob)).log()

        seg_scores_all = torch.cat((bg_score, seg_scores), dim=0)
        out = []
        for s in seg_scores_all:
            s_out = 1.0 / (seg_scores_all - s.unsqueeze(0)).exp().sum(dim=0)
            out.append(s_out)
        seg_probs_all = torch.stack(out, dim=0)

        dis_probs_all = torch.sigmoid(torch.cat((bg_score, dis_scores), dim=0))

        # Obtain segmentation labels
        obj_ids_all = np.array([0, *map(int, object_ids)], dtype=np.uint8)
        merged_labels = obj_ids_all[seg_probs_all.argmax(dim=0).cpu()]

        # Update distractors and re-join target and distractor segments
        dis_probs_all = self.update_distractors(object_ids, seg_probs_all, dis_probs_all, merged_labels)
        raw_segs = odict()
        for i, obj_id in enumerate(object_ids):
            raw_segs[obj_id] = torch.cat((seg_probs_all[i + 1], dis_probs_all[i + 1]), dim=-3).cpu()

        return merged_labels, raw_segs

    def merge(self, frame, targets):
        """ Merges the predictions of individual targets. Note: Use this as a static method ... """

        tg_channel = 0

        object_ids = list(frame.segmentation.keys())

        fg = None  # All foreground objects - a 0/1 mask
        for target in frame.segmentation.values():
            tg = target[tg_channel].cpu()
            if not torch.is_tensor(tg):
                tg = torch.from_numpy(tg)
            if fg is None:
                fg = tg
            else:
                fg = torch.max(tg, fg)

        # Merge segmentation scores using the soft-aggregation approach from RGMP

        # Collect segmentation scores
        seg_scores = []
        for obj_id in object_ids:
            if obj_id not in frame.segmentation_raw:
                # This is the first frame for this target

                # Get ground-truth target and generate distractor
                tg = frame.segmentation[obj_id][tg_channel].cpu()
                dt = fg * (1 - tg)
                # Convert to logits, i.e raw segmentations
                tg = (tg - 0.5) * 200.0  # (100 to target, -100 to background)
                dt = (dt - 0.5) * 200.0

                s = torch.stack((tg, dt))
            else:
                # Not the first frame for this target.
                s = frame.segmentation_raw[obj_id]

            seg_scores.append(s.reshape(1, *s.shape[-3:]))
        seg_scores = torch.stack(seg_scores).float()

        have_distractors = seg_scores.shape[-3] == 2
        assert have_distractors

        merged_labels, raw_segs = self._merge_sigmoid(object_ids, seg_scores)

        # Get target bounding boxes

        merged_boxes = {}
        for obj_id, seg_prob in raw_segs.items():
            merged_boxes[obj_id] = torch.cat(targets[obj_id].spatial_tracker.get_target_bbox(seg_prob, tlhw=True)).tolist()

        frame.labels = merged_labels
        frame.segmentation_raw = raw_segs
        frame.target_bbox = merged_boxes

        return frame

    # def uncrop(self, s_crop, sample_pos, sample_scale, im_size, outside_value=-100.0):
    #     """ Obtain segmentation scores for the full image using the scores for the search region crop. This is done by
    #         assigning a low score (outside_value=-100) for image regions outside the search region """
    #
    #     # Resize the segmentation scores to match the image scale
    #     s = F.interpolate(s_crop, scale_factor=sample_scale.item(), mode='bilinear', align_corners=True, recompute_scale_factor=True)
    #     (sc, sh, sw), dev = s.shape[-3:], s.device
    #     s = s.view(*s.shape[-3:])
    #
    #     # Regions outside search area get very low score
    #     s_im = torch.ones((sc, *im_size), dtype=s.dtype, device=dev) * outside_value
    #
    #     # Find the co-ordinates of the search region in the image scale
    #     r1 = int(sample_pos[0].item() - 0.5*s.shape[-2])
    #     c1 = int(sample_pos[1].item() - 0.5*s.shape[-1])
    #
    #     r2 = r1 + s.shape[-2]
    #     c2 = c1 + s.shape[-1]
    #
    #     r1_pad = max(0, -r1)
    #     c1_pad = max(0, -c1)
    #
    #     r2_pad = max(r2 - im_size[-2], 0)
    #     c2_pad = max(c2 - im_size[-1], 0)
    #
    #     # Copy the scores for the search region
    #     s_im[:, r1 + r1_pad:r2 - r2_pad, c1 + c1_pad:c2 - c2_pad] = s[:, r1_pad:sh - r2_pad, c1_pad:sw - c2_pad]
    #
    #     return s_im

    def segment_target(self, feat_tm, feat_bbone, im_size, gt_mask=None, get_mask_enc=False):
        with torch.no_grad():
            segmentation_scores, mask_encoding_pred = self.net.segment_target(
                self.target_filter, feat_tm, feat_bbone, im_size, gt_mask=gt_mask)

        if get_mask_enc:
            return segmentation_scores, mask_encoding_pred
        return segmentation_scores

    def extract_patches(self, im: torch.Tensor, pos: torch.Tensor, scale, sz: torch.Tensor, gt_mask=None):
        im_patches, patch_coords = sample_patch_multiscale(im, pos, scale.unsqueeze(0), sz,
                                                           mode=self.params.get('border_mode', 'replicate'),
                                                           max_scale_change=self.params.get('patch_max_scale_change', None))
        if gt_mask is not None:
            gt_mask, _ = sample_patch_multiscale(gt_mask, pos, scale.unsqueeze(0), sz,
                                                 mode=self.params.get('border_mode', 'replicate'),
                                                 max_scale_change=self.params.get('patch_max_scale_change', None))
            return patch_coords[0], im_patches, gt_mask

        return patch_coords[0], im_patches

    def extract_backbone_features(self, im_patches):
        with torch.no_grad():
            return self.net.extract_backbone(im_patches)

    def extract_target_model_features(self, backbone_feat):
        """ Extract features input to the target model"""
        with torch.no_grad():
            return self.net.extract_target_model_features(backbone_feat)

    def generate_init_samples(self, im: torch.Tensor, init_mask):
        """ Generate initial training sample."""

        # Locate the target
        pos, scale, size, transforms = self.spatial_tracker.generate_init_samples_params(im, init_mask)

        # Extract image patches
        im_patches = sample_patch_transformed(im, pos, scale, size, transforms)
        init_masks = sample_patch_transformed(init_mask, pos, scale, size, transforms, is_mask=True)
        init_masks = init_masks.to(self.device)
        self.transforms = transforms

        # Extract backbone features
        feat_bbone = self.extract_backbone_features(im_patches)

        return im_patches, feat_bbone, init_masks

    @staticmethod
    def _visualize_mask_enc(mask_enc):

        from davos.lib.debug import DebugVis
        vis = DebugVis()

        channels = list(range(0, 32))
        remove = [2, 4, 12, 14, 22, 23, 25, 26, 27, 28, 30, 31]
        for v in remove:
            channels.remove(v)
        c = mask_enc.squeeze()[channels]
        vis.show_enclb(c)



    def init_target_model(self, feat_bbone, masks):
        # Get target model features
        feat_tm = self.extract_target_model_features(feat_bbone)

        # Set sizes
        self.feature_sz = torch.Tensor(list(feat_tm.shape[-2:]))
        ksz = self.net.target_model.filter_size
        self.kernel_size = torch.Tensor(_pair(ksz))
        self.output_sz = self.feature_sz + (self.kernel_size + 1) % 2

        self.spatial_tracker.set_tm_params(self.kernel_size, self.feature_sz)

        # Set number of iterations
        num_iter = self.params.get('net_opt_iter', None)

        visualize = hasattr(self, 'vis')

        # Encode the masks and train the target model
        with torch.no_grad():
            mask_enc, mask_ws = self.net.label_encoder(masks, feat_tm.unsqueeze(1))
            self.target_filter, _, losses = self.net.target_model.get_filter(
                feat_tm.unsqueeze(1), mask_enc, mask_ws, num_iter=num_iter, compute_losses=visualize)

            if visualize:
                test_enc, test_pred = self.apply_target_model(
                    feat_tm.unsqueeze(1), self.target_filter,
                    decode=True, feat_bbone=feat_bbone, im_size=masks.shape[-2:]
                )
                self.vis.current_value_range = (-100, 100)
                self.vis.show_rawseg(test_pred[0, 0], title="predicted target")
                self.vis.show_rawseg(test_pred[0, 1], title="predicted distractors")
                self.vis.show_enclb(mask_enc, title="encoded training mask")
                self.vis.show_enclb(test_enc, title="predicted encoded mask")
                # self.vis.lineplot(torch.stack(losses), title="losses")

        # Init memory
        if self.params.get('update_target_model', True):
            self.memory.initialize(TensorList([feat_tm]), labels=masks.view(-1, *masks.shape[-3:]))

    def apply_target_model(self, feat_tm, tm_filter, feat_bbone=None, im_size=None, decode=False):
        assert tm_filter.dim() == 5  # seq, filters, ch, h, w
        feat_tm = feat_tm.view(1, 1, *feat_tm.shape[-3:])
        mask_pred_enc = self.net.target_model.apply_target_model(tm_filter, feat_tm)
        if decode:
            mask_pred, decoder_feat = self.net.decoder(mask_pred_enc, feat_bbone, im_size)
            return mask_pred_enc, mask_pred
        return mask_pred_enc

    def update_target_model(self, train_x, mask, learning_rate=None):

        # Update the tracker memory
        lr = self.params.learning_rate if learning_rate is None else learning_rate
        if self.frame_num % self.params.get('train_sample_interval', 1) == 0:
            self.memory.update(TensorList([train_x]), mask, lr)

        # Decide the number of iterations to run
        num_iter = 0
        if (self.frame_num - 1) % self.params.train_skipping == 0:
            num_iter = self.params.get('net_opt_update_iter', None)

        if num_iter > 0:
            with torch.no_grad():
                # Read memory, encode masks
                samples, masks, sample_weights = self.memory.read()
                mask_enc, mask_sw = self.net.label_encoder(masks.unsqueeze(1), samples.unsqueeze(1))

                if mask_sw is not None:
                    # mask_sw provides spatial weights, while sample_weights contains temporal weights.
                    sample_weights = mask_sw * sample_weights.view(-1, 1, 1, 1, 1)

                # Optimize the target model filter
                target_filter, _, losses = self.net.target_model.filter_optimizer(
                    TensorList([self.target_filter]), num_iter=num_iter,
                    feat=samples.unsqueeze(1), label=mask_enc, sample_weight=sample_weights)

                self.target_filter = target_filter[0]

