from torch import nn
from torch.nn.modules.utils import _pair
from davos.eval.lib.preprocessing import sample_patch_multiscale, sample_patch_transformed, sample_patch
# noinspection PyUnresolvedReferences
from davos.lib.debug import DebugVis
from .common import *


def get_tracker_class():
    return LWL


class LWL(nn.Module):
    """ The LWL tracker """

    def __init__(self, object_id, params):
        super().__init__()
        self.params = params

        self.object_id = object_id
        self.net = self.params.net
        self.device = next(self.net.parameters()).device
        self.spatial_tracker = SpatialTracker(params)
        self.memory = Memory(params)

    def initialize(self, frame: Frame, prev_frame, bbox=None):

        self.cached_object_ids = None
        self.frame_num = 1

        mask = (frame.labels == self.object_id)
        mask = mask.view(1, 1, *mask.shape[-2:]).float()
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

        if self.frame_num > 2:
            if self.params.get('update_target_model', True):

                # Crop the segmentation mask for the previous search area
                prev_seg_prob_crop, _ = sample_patch(
                    prev_seg_prob.unsqueeze(0), self.prev_pos, self.prev_scale * self.prev_im_size, self.prev_im_size,
                    mode=self.params.get('border_mode', 'replicate'), max_scale_change=self.params.get('patch_max_scale_change'),
                    is_mask=True)  # is_mask=True disables replication padding
                prev_seg_prob_crop = prev_seg_prob_crop.squeeze(0)
                # Update the target model
                self.update_target_model(self.prev_feat_tm, prev_seg_prob_crop.clone())

        # Estimate target bbox
        tg_prob = prev_seg_prob[tg_channel]  # Target probability
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
        seg_scores = self.uncrop(seg_scores, sample_pos, sample_scale, im.shape[-2:], outside_value=-100.0)  # Uncrop to full image

        seg_prob = torch.sigmoid(seg_scores)  # Probability of being target/distractor at each pixel
        seg_mask = (seg_prob > 0.5).float()   # Binary segmentation mask

        # Get target box from the predicted segmentation
        tg_prob = seg_prob[tg_channel]  # Target probability
        new_bbox = torch.cat(self.spatial_tracker.get_target_bbox(tg_prob, tlhw=True))

        if hasattr(self, 'vis'):
            # FIXME: Update to one-hot encoding
            self.vis.imshow(im, "image")
            m = seg_mask
            im = torch.stack((m[1], m[0], torch.zeros_like(m[0])), dim=-3)
            self.vis.imshow(im, "mask (green), distractor (red)")
            self.vis.show_rawseg(seg_scores[0], title="predicted target")
            self.vis.show_rawseg(seg_scores[1], title="predicted distractors")
            self.vis.show_enclb(mask_enc_pred, title="predicted encoded mask")
            pass

        # Save output

        frame.segmentation[obj_id] = seg_mask.cpu()
        frame.target_bbox[obj_id] = new_bbox.tolist()
        frame.segmentation_raw[obj_id] = seg_scores.cpu()

        return frame

    def _merge_sigmoid(self, object_ids, seg_scores):

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

        # Obtain segmentation labels
        obj_ids_all = np.array([0, *map(int, object_ids)], dtype=np.uint8)
        merged_labels = obj_ids_all[seg_probs_all.argmax(dim=0).cpu()]

        raw_segs = {obj_id: seg_probs_all[i + 1].cpu() for i, obj_id in enumerate(object_ids)}

        return merged_labels, raw_segs

    def merge(self, frame, targets):
        """ Merges the predictions of individual targets. Note: Use this as a static method ... """

        tg_channel = 0

        object_ids = list(frame.segmentation.keys())

        # Merge segmentation scores using the soft-aggregation approach from RGMP

        # Collect segmentation scores
        seg_scores = []
        for obj_id in object_ids:
            if obj_id not in frame.segmentation_raw:
                # This is the first frame for this target

                # Get ground-truth target
                tg = frame.segmentation[obj_id][tg_channel]
                # Convert to logits, i.e raw segmentations
                tg = (tg - 0.5) * 200.0  # (100 to target, -100 to background)
                s = tg
            else:
                # Not the first frame for this target.
                s = frame.segmentation_raw[obj_id]

            seg_scores.append(s.view(1, *s.shape[-2:]).to(frame.image.device))
        seg_scores = torch.stack(seg_scores).float()

        # seg_scores.shape = (num_targets, height, width)

        merged_labels, raw_segs = self._merge_sigmoid(object_ids, seg_scores)

        # Get target bounding boxes

        merged_boxes = {}
        for obj_id, seg_prob in raw_segs.items():
            tg_prob = seg_prob[tg_channel]
            merged_boxes[obj_id] = torch.cat(targets[obj_id].spatial_tracker.get_target_bbox(tg_prob, tlhw=True)).tolist()

        frame.labels = merged_labels
        frame.segmentation_raw = raw_segs
        frame.target_bbox = merged_boxes

        return frame

    def uncrop(self, s_crop, sample_pos, sample_scale, im_size, outside_value=-100.0):
        """ Obtain segmentation scores for the full image using the scores for the search region crop. This is done by
            assigning a low score (outside_value=-100) for image regions outside the search region """

        # Resize the segmentation scores to match the image scale
        s = F.interpolate(s_crop, scale_factor=sample_scale.item(), mode='bilinear', align_corners=True, recompute_scale_factor=True)
        (sc, sh, sw), dev = s.shape[-3:], s.device
        s = s.view(*s.shape[-3:])

        # Regions outside search area get very low score
        s_im = torch.ones((sc, *im_size), dtype=s.dtype, device=dev) * outside_value

        # Find the co-ordinates of the search region in the image scale
        r1 = int(sample_pos[0].item() - 0.5*s.shape[-2])
        c1 = int(sample_pos[1].item() - 0.5*s.shape[-1])

        r2 = r1 + s.shape[-2]
        c2 = c1 + s.shape[-1]

        r1_pad = max(0, -r1)
        c1_pad = max(0, -c1)

        r2_pad = max(r2 - im_size[-2], 0)
        c2_pad = max(c2 - im_size[-1], 0)

        # Copy the scores for the search region
        s_im[:, r1 + r1_pad:r2 - r2_pad, c1 + c1_pad:c2 - c2_pad] = s[:, r1_pad:sh - r2_pad, c1_pad:sw - c2_pad]

        return s_im

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

