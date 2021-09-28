import math
from dataclasses import dataclass
from typing import Union, Tuple

import numpy as np
import torch
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
import cv2

from davos.eval.lib import augmentation
from davos.lib import TensorList


def warp_affine(src, H, dsize, mode='bicubic'):
    """ Warp a single image or a batch of images with an affine transform.
    :param src:    Input to warp, shape = (..., H, W)
    :param H:      Forward warp affine transform
    :param dsize:  Output (height, width) in pixels
    :param mode:   Interpolation mode. Default 'bicubic'
    :return:  Warped output
    """

    device = src.device
    shape = src.shape[:-2]  # All but the spatial dimensions
    src = src.view(-1, *src.shape[-2:]).detach().cpu().numpy()

    H = H[:2].float().numpy()
    dsize = int(dsize[1]), int(dsize[0])
    flags = dict(bicubic=cv2.INTER_CUBIC, bilinear=cv2.INTER_LINEAR, nearest=cv2.INTER_NEAREST)[mode]
    dst = []
    for im in list(src):
        im = cv2.warpAffine(im, H, dsize, flags=flags)
        dst.append(torch.from_numpy(im).to(device))

    dst = torch.stack(dst)
    dst = dst.view(*shape, *dst.shape[-2:])

    return dst


@dataclass
class Segment:

    pos: Union[torch.Tensor, Tuple[float, float]]  # Center position in full image (y,x)
    scale: float  # patch-to-full-image scale change
    patch_size: Union[torch.Size, Tuple[int, int]]
    im_size: Union[torch.Size, Tuple[int, int]]  # full-image size

    im_patch: torch.tensor = None
    feat_bbone: torch.tensor = None
    mask_enc: torch.tensor = None
    scores: torch.tensor = None

    def crop_transform(self):
        """
        :return:  The crop affine transform.
        """
        sy, sx = _pair(self.scale.tolist() if torch.is_tensor(self.scale) else self.scale)
        crop_h, crop_w = int(sy * self.patch_size[0]), int(sx * self.patch_size[1])  # Rescaled size, with the equivalent of "recompute_scale_factor=True"

        # Find the top left corner of the search region in the image scale
        y1 = np.floor(self.pos[0] - 0.5 * crop_h)
        x1 = np.floor(self.pos[1] - 0.5 * crop_w)
        H_crop = _scale(1./sx, 1./sy) @ _translate(-x1, -y1)

        return H_crop


class Frame:
    """ Information exchange object for input and tracker states that might be shared between targets """

    image: torch.Tensor
    labels: torch.Tensor

    def __init__(self, image=None, labels=None, device='cpu'):

        if image is not None:
            image = image.to(device)
        if labels is not None:
            labels = labels.to(device)

        self.image = image
        self.labels = labels
        self.target_bbox = dict()
        self.segmentation_raw = dict()
        self.segmentation = dict()

        # Intermediate data for MDLWL.track2()
        self.seg_scores = dict()
        self.segments = dict()


def _scale(sx, sy=None):
    sy = sx if sy is None else sy
    return torch.tensor([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])

def _translate(dx, dy):
    return torch.tensor([[1, 0, dx], [0, 1, dy], [0, 0, 1]])


def _crop_params(sample_pos, sample_scale, crop_size, im_size):
    """ Obtain full image and scaled-crop coordinate ranges, from the search region parameters and crop size. """

    sample_scale = sample_scale.tolist() if torch.is_tensor(sample_scale) else sample_scale
    sample_scale = _pair(sample_scale)
    # Rescaled size, with the equivalent of "recompute_scale_factor=True"
    s_size = [int(float(s) * sz) for s, sz in zip(sample_scale, tuple(crop_size))]

    # Find the coordinates of the search region in the image scale
    sample_pos = sample_pos.tolist()
    r1 = int(sample_pos[0] - 0.5 * s_size[-2])
    c1 = int(sample_pos[1] - 0.5 * s_size[-1])

    r2 = r1 + s_size[-2]
    c2 = c1 + s_size[-1]

    r1_pad = max(0, -r1)
    c1_pad = max(0, -c1)

    r2_pad = max(r2 - im_size[-2], 0)
    c2_pad = max(c2 - im_size[-1], 0)

    im_y = slice(r1 + r1_pad, r2 - r2_pad)
    im_x = slice(c1 + c1_pad, c2 - c2_pad)

    s_y = slice(r1_pad, s_size[-2] - r2_pad)
    s_x = slice(c1_pad, s_size[-1] - c2_pad)

    return s_size, im_y, im_x, s_y, s_x


def _crop_transform(sample_pos, sample_scale, patch_size, im_size):
    """
    :param sample_pos:  Center position in full image
    :param sample_scale:  Patch inverse scale change relative to full image (i.e image -> patch scale change)
    :param patch_size:  Size of the final patch (h, w)
    :param im_size:  Size of the full image (h, w)
    :return:
    """
    S = sample_scale
    sample_scale = sample_scale.tolist() if torch.is_tensor(sample_scale) else sample_scale
    sample_scale = _pair(sample_scale)
    # Rescaled size, with the equivalent of "recompute_scale_factor=True"
    crop_size = [int(float(s) * sz) for s, sz in zip(sample_scale, tuple(patch_size))]

    # Find the top left corner of the search region in the image scale
    sample_pos = sample_pos.tolist()
    y1 = int(sample_pos[0] - 0.5 * crop_size[-2])
    x1 = int(sample_pos[1] - 0.5 * crop_size[-1])

    sx = patch_size[-2] / crop_size[-2]
    sy = patch_size[-1] / crop_size[-1]

    H_crop = _scale(sx, sy) @ _translate(-x1, -y1)

    return H_crop


def uncrop(s_crop, sample_pos, sample_scale, im_size, outside_value=-100.0, crop_size=None):
    """ Obtain segmentation scores for the full image using the scores for the search region crop.
        Image regions outside the search region are assigned low scores (outside_value=-100) """
    # crop_size: needed by crop, ignored by uncrop"

    # Resize the segmentation scores to match the image scale
    s = F.interpolate(s_crop, scale_factor=sample_scale.item(), mode='bilinear', align_corners=True, recompute_scale_factor=True)
    s = s.view(*s.shape[-3:])

    s_im = torch.ones((s.shape[-3], *im_size), dtype=s.dtype, device=s.device) * outside_value
    s_size, im_y, im_x, s_y, s_x = _crop_params(sample_pos, sample_scale, s_crop.shape[-2:], im_size)
    s_im[:, im_y, im_x] = s[:, s_y, s_x]  # Copy the scores for the search region
    return s_im


def crop(s_im, sample_pos, sample_scale, crop_size, outside_value=-100.0, im_size=None):
    # im_size: needed by uncrop, ignored by crop

    s_size, im_y, im_x, s_y, s_x = _crop_params(sample_pos, sample_scale, crop_size, s_im.shape[-2:])
    s = torch.ones((s_im.shape[-3], *s_size), dtype=s_im.dtype, device=s_im.device) * outside_value
    s[:, s_y, s_x] = s_im[:, im_y, im_x]
    s_crop = F.interpolate(s.unsqueeze(0), crop_size, mode='bilinear', align_corners=True).squeeze(0)

    return s_crop


def remap_distractor(dt: Segment, tg: Segment, empty, enc_stride):
    """ Remap a distractor segment to the same image patch as a target segment
    :param dt:  distractor to remap
    :param empty:  Encoded background mask, i.e label encoder output with all-zeros input mask
    :param tg:  target to remap to
    :param enc_stride: label encoder stride (int) e.g 8 or 16
    :return:
    """
    # Geometric transform from distractor encoded-mask space to target encoded-mask space
    dt_H_uncrop = dt.crop_transform().inverse()
    tg_H_crop = tg.crop_transform()

    S_dec = _scale(enc_stride)  # mask_enc -> seg_score scale change
    H_im_dt = dt_H_uncrop @ S_dec  # dt_mask_enc -> (decoder) -> dt_seg_score_crop -> (uncrop) -> seg_score
    H_tg_im = S_dec.inverse() @ tg_H_crop  # seg_score -> (crop) tg_seg_score_crop -> (label encoder) -> tg_mask_enc

    H = H_tg_im @ H_im_dt  # The full transform from dt_mask_enc coordinates to tg_mask_enc coordinates

    # Warp the distractor into the target crop space

    m_dt = dt.mask_enc
    sz = empty.shape[-2:]
    alpha = torch.ones((*m_dt.shape[:-3], 1, *m_dt.shape[-2:]), device=m_dt.device)
    m = warp_affine(m_dt, H, sz, mode='bilinear')
    a = warp_affine(alpha, H, sz, mode='bilinear')
    m = m * a + (1 - a) * empty

    return m


# Memory and spatial tracker classes, adapted from the original LWL code without major functional changes.

class Memory:

    def __init__(self, params):
        self.params = params
        self._capacity = self.params.sample_memory_size
        self._learning_rate = self.params.learning_rate

    def initialize(self, feats: TensorList, labels):
        """ Initialize the sample memory used to update the target model """
        assert labels.dim() == 4
        assert feats[0].dim() == 4

        # Initialize first-frame spatial training samples
        self.num_init_samples = feats.size(0)
        init_sample_weights = TensorList([x.new_ones(1) / x.shape[0] for x in feats])

        # Sample counters and weights for spatial
        self.num_stored_samples = self.num_init_samples.copy()
        self.previous_replace_ind = [None] * len(self.num_stored_samples)
        self.sample_weights = TensorList([x.new_zeros(self._capacity) for x in feats])
        for sw, init_sw, num in zip(self.sample_weights, init_sample_weights, self.num_init_samples):
            sw[:num] = init_sw

        # Initialize memory
        self.features = TensorList([x.new_zeros(self._capacity, *x.shape[-3:]) for x in feats])
        self.labels = labels.new_zeros(self._capacity, *labels.shape[-3:])
        self.labels[:labels.shape[0], ...] = labels

        for ts, x in zip(self.features, feats):
            ts[:x.shape[0]] = x

    def update(self, feats: TensorList, labels, learning_rate=None):
        """ Add a new sample to the memory. If the memory is full, an old sample are removed"""
        # Update weights and get replace ind
        replace_ind = self._update_sample_weights(self.sample_weights, self.previous_replace_ind, self.num_stored_samples, self.num_init_samples, learning_rate)
        self.previous_replace_ind = replace_ind

        # Update sample and label memory
        for train_samp, x, ii in zip(self.features, feats, replace_ind):
            train_samp[ii:ii + 1] = x

        ii = replace_ind[0]
        self.labels[ii:ii + 1] = labels

        self.num_stored_samples = [n + 1 if n < self._capacity else n for n in self.num_stored_samples]

    def _update_sample_weights(self, sample_weights, previous_replace_ind, num_stored_samples, num_init_samples, learning_rate=None):
        """ Update weights and get index to replace """
        replace_ind = []
        for sw, prev_ind, num_samp, num_init in zip(sample_weights, previous_replace_ind, num_stored_samples, num_init_samples):
            lr = learning_rate
            if lr is None:
                lr = self.params.learning_rate

            init_samp_weight = self.params.get('init_samples_minimum_weight', None)
            if init_samp_weight == 0:
                init_samp_weight = None
            s_ind = 0 if init_samp_weight is None else num_init

            if num_samp == 0 or lr == 1:
                sw[:] = 0
                sw[0] = 1
                r_ind = 0
            else:
                # Get index to replace
                if num_samp < sw.shape[0]:
                    r_ind = num_samp
                else:
                    _, r_ind = torch.min(sw[s_ind:], 0)
                    r_ind = r_ind.item() + s_ind

                # Update weights
                if prev_ind is None:
                    if self.params.get('lower_init_weight', False):
                        sw[r_ind] = 1
                    else:
                        sw /= 1 - lr
                        sw[r_ind] = lr
                else:
                    sw[r_ind] = sw[prev_ind] / (1 - lr)

            sw /= sw.sum()
            if init_samp_weight is not None and sw[:num_init].sum() < init_samp_weight:
                sw /= init_samp_weight + sw[num_init:].sum()
                sw[:num_init] = init_samp_weight / num_init

            replace_ind.append(r_ind)

        return replace_ind

    def read(self):

        feats = self.features[0][:self.num_stored_samples[0], ...]
        labels = self.labels[:self.num_stored_samples[0], ...]
        weights = self.sample_weights[0][:self.num_stored_samples[0]]

        return feats, labels, weights


class SpatialTracker:

    def __init__(self, params):
        self.params = params

    def initialize(self, bbox):

        # Set target center and target size
        self.pos = torch.Tensor([bbox[1] + (bbox[3] - 1)/2, bbox[0] + (bbox[2] - 1)/2])
        self.target_sz = torch.Tensor([bbox[3], bbox[2]])

        # Set sizes
        sz = self.params.image_sample_size
        self.img_sample_sz = torch.Tensor([sz, sz] if isinstance(sz, int) else sz)
        self.img_support_sz = self.img_sample_sz

        # Set search area
        search_area = torch.prod(self.target_sz * self.params.search_area_scale).item()
        self.target_scale = math.sqrt(search_area) / self.img_sample_sz.prod().sqrt()

        # Target size in base scale
        self.base_target_sz = self.target_sz / self.target_scale

    def set_tm_params(self, kernel_size, feature_sz):
        self.kernel_size = kernel_size
        self.feature_sz = feature_sz

    def get_sample_location(self, sample_coord):
        """Get the location of the extracted sample."""
        sample_coord = sample_coord.float()
        sample_pos = 0.5*(sample_coord[:2] + sample_coord[2:] - 1)
        sample_scales = ((sample_coord[2:] - sample_coord[:2]) / self.img_sample_sz).prod().sqrt()
        return sample_pos, sample_scales

    def get_centered_sample_pos(self):
        """Get the center position for the new sample. Make sure the target is correctly centered."""
        pos = self.pos + ((self.feature_sz + self.kernel_size) % 2) * self.target_scale * \
              self.img_support_sz / (2*self.feature_sz)
        return pos, self.target_scale, self.img_sample_sz

    def clip_scale_change(self, new_target_scale):
        """ Limit scale change """
        if not isinstance(self.params.get('max_scale_change'), (tuple, list)):
            max_scale_change = (self.params.get('max_scale_change'), self.params.get('max_scale_change'))
        else:
            max_scale_change = self.params.get('max_scale_change')

        scale_change = new_target_scale / self.target_scale

        if scale_change < max_scale_change[0]:
            new_target_scale = self.target_scale * max_scale_change[0]
        elif scale_change > max_scale_change[1]:
            new_target_scale = self.target_scale * max_scale_change[1]

        return new_target_scale

    def get_target_bbox(self, seg_prob_im, tlhw=False):
        """ Estimate target bounding box tlhw=False: (center, size), tlhw=True:(top-left,size) using the predicted segmentation probabilities """

        dev = seg_prob_im.device

        # If predicted mask area is too small, target might be occluded. In this case, just return prev. box
        if seg_prob_im.sum() < self.params.get('min_mask_area', -10):
            return self.pos, self.target_sz

        yrange = torch.arange(seg_prob_im.shape[-2], dtype=torch.float32, device=dev)
        xrange = torch.arange(seg_prob_im.shape[-1], dtype=torch.float32, device=dev)

        if self.params.get('seg_to_bb_mode') == 'var':
            # Target center is the center of mass of the predicted per-pixel seg. probability scores
            prob_sum = seg_prob_im.sum()
            e_y = torch.sum(seg_prob_im.sum(dim=-1) * yrange) / prob_sum
            e_x = torch.sum(seg_prob_im.sum(dim=-2) * xrange) / prob_sum

            # Target size is obtained using the variance of the seg. probability scores
            e_h = torch.sum(seg_prob_im.sum(dim=-1) * (yrange - e_y) ** 2) / prob_sum
            e_w = torch.sum(seg_prob_im.sum(dim=-2) * (xrange - e_x) ** 2) / prob_sum

            sz_factor = self.params.get('seg_to_bb_sz_factor', 4)
            cc, sz = torch.Tensor([e_y, e_x]), torch.Tensor([e_h.sqrt() * sz_factor, e_w.sqrt() * sz_factor])
            if not tlhw:
                return cc, sz
            else:
                tl, sz = cc[[1, 0]] - (sz[[1, 0]] - 1) / 2, sz[[1, 0]]
                return tl, sz
        else:
            raise Exception('Unknown seg_to_bb_mode mode {}'.format(self.params.get('seg_to_bb_mode')))

    def update_target_bbox_estimate(self, seg_prob_im):
        self.pos, self.target_sz = self.get_target_bbox(seg_prob_im)
        new_target_scale = torch.sqrt(self.target_sz.prod() / self.base_target_sz.prod())

        if self.params.get('max_scale_change') is not None:
            # Do not allow drastic scale change, as this might be caused
            # due to occlusions or incorrect mask prediction
            new_target_scale = self.clip_scale_change(new_target_scale)

        # Update target size and scale using the filtered target size
        self.target_scale = new_target_scale
        self.target_sz = self.base_target_sz * self.target_scale

    def generate_init_samples_params(self, im: torch.Tensor, init_mask):
        """ Generate initial training sample."""

        mode = self.params.get('border_mode', 'replicate')
        if 'inside' in mode:
            # Get new sample size if forced inside the image
            im_sz = torch.Tensor([*im.shape[-2:]])
            sample_sz = self.target_scale * self.img_sample_sz
            shrink_factor = (sample_sz.float() / im_sz)
            if mode == 'inside':
                shrink_factor = shrink_factor.max()
            elif mode == 'inside_major':
                shrink_factor = shrink_factor.min()
            shrink_factor.clamp_(min=1, max=self.params.get('patch_max_scale_change', None))
            sample_sz = (sample_sz.float() / shrink_factor)
            init_sample_scale = (sample_sz / self.img_sample_sz).prod().sqrt()
            tl = self.pos - (sample_sz - 1) / 2
            br = self.pos + sample_sz / 2 + 1
            global_shift = - ((-tl).clamp(0) - (br - im_sz).clamp(0)) / init_sample_scale
        else:
            init_sample_scale = self.target_scale
            global_shift = torch.zeros(2)

        init_sample_pos = self.pos.round()

        # Compute augmentation size
        aug_expansion_factor = 2.0
        aug_expansion_sz = self.img_sample_sz.clone()
        aug_output_sz = None
        if aug_expansion_factor is not None and aug_expansion_factor != 1:
            aug_expansion_sz = (self.img_sample_sz * aug_expansion_factor).long()
            aug_expansion_sz += (aug_expansion_sz - self.img_sample_sz.long()) % 2
            aug_expansion_sz = aug_expansion_sz.float()
            aug_output_sz = self.img_sample_sz.long().tolist()

        # Can be extended to include data augmentation on the initial frame
        transforms = [augmentation.Identity(aug_output_sz, global_shift.long().tolist())]

        return init_sample_pos, init_sample_scale, aug_expansion_sz, transforms
