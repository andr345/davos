import torch
import types
from torch import nn

import os

from davos.config import env_settings
from davos.sys import loading as ltr_loading


def load_network(path, **kwargs):
    """Load network for tracking.
    args:
        path - Path to network.
        **kwargs - Additional key-word arguments that are sent to davos.sys.loading.load_network.
    """
    kwargs['backbone_pretrained'] = False
    net, _ = ltr_loading.load_network(path, **kwargs)
    return net


class NetWrapper:
    """Used for wrapping networks in pytracking.
    Network modules and functions can be accessed directly as if they were members of this class."""
    _rec_iter=0
    def __init__(self, net_path, use_gpu=True, initialize=False, **kwargs):
        self.net_path = net_path
        self.use_gpu = use_gpu
        self.net = None
        self.net_kwargs = kwargs
        self.net_is_initialized = False
        if initialize:
            self.initialize()

    def __getattr__(self, name):
        if self._rec_iter > 0:
            self._rec_iter = 0
            return None
        self._rec_iter += 1
        try:
            ret_val = getattr(self.net, name)
        except Exception as e:
            self._rec_iter = 0
            raise e
        self._rec_iter = 0
        return ret_val

    def load_network(self):
        self.net = load_network(self.net_path, **self.net_kwargs)
        if self.use_gpu:
            self.cuda()
        self.eval()

    def initialize(self, force=False):
        if self.net_is_initialized and not force:
            return
        self.load_network()
        self.net_is_initialized = True


class ColorNormalizer(nn.Module):

    def __init__(self, means, stds, maxval=1.0, flip_bgr=False):
        super().__init__()

        self.flip_bgr = flip_bgr

        stds = torch.Tensor(stds).view(1, -1, 1, 1)
        means = torch.Tensor(means).view(1, -1, 1, 1)
        self.register_buffer('gain', 1 / maxval / stds)
        self.register_buffer('bias', -means / stds)

    def forward(self, im):
        im = im.to(self.gain.device)
        if self.flip_bgr:
            im = im.flip(dims=(-3,))
        im = im * self.gain + self.bias
        return im


def load_net_with_backbone(net_path, use_gpu=True, image_format='rgb',
                           mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), **kwargs):

    net = load_network(net_path, **kwargs)
    maxval = 255.0 if image_format in ['rgb', 'bgr'] else 1.0
    flip_channels = image_format in ['bgr', 'bgr255']

    net.preprocess_image = torch.jit.script(ColorNormalizer(mean, std, maxval, flip_channels))

    def extract_backbone(self, im):
        im = self.preprocess_image(im)
        im = self.extract_backbone_features(im)
        return im

    def initialize(self):
        pass

    net.extract_backbone = types.MethodType(extract_backbone, net)
    net.initialize = types.MethodType(initialize, net)

    if use_gpu:
        net = net.cuda()
    net.eval()

    return net


class NetWithBackbone(NetWrapper):
    """Wraps a network with a common backbone.
    Assumes the network have a 'extract_backbone_features(image)' function."""

    def __init__(self, net_path, use_gpu=True, initialize=False, image_format='rgb',
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), **kwargs):
        super().__init__(net_path, use_gpu, initialize, **kwargs)

        self.image_format = image_format
        self._mean = torch.Tensor(mean).view(1, -1, 1, 1)
        self._std = torch.Tensor(std).view(1, -1, 1, 1)

        maxval = 255.0 if self.image_format in ['rgb', 'bgr'] else 1.0
        flip_channels = self.image_format in ['bgr', 'bgr255']
        self.normalize = ColorNormalizer(mean, std, maxval, flip_channels)

    def initialize(self, image_format='rgb', mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super().initialize()
        device = next(self.net.parameters()).device
        self.normalize = self.normalize.to(device)

    def preprocess_image(self, im: torch.Tensor):
        """Normalize the image with the mean and standard deviation used by the network."""

        if self.image_format in ['rgb', 'bgr']:
            im = im/255

        if self.image_format in ['bgr', 'bgr255']:
            im = im[:, [2, 1, 0], :, :]
        im -= self._mean
        im /= self._std

        if self.use_gpu:
            im = im.cuda()

        return im

    def extract_backbone(self, im: torch.Tensor):
        """Extract backbone features from the network.
        Expects a float tensor image with pixel range [0, 255]."""
        # im = self.preprocess_image(im)
        im = self.normalize(im)  # and upload
        return self.net.extract_backbone_features(im)
