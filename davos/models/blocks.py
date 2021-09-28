from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.resnet import BasicBlock


def conv_block(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=True,
               batch_norm=True, relu=True, padding_mode='zeros'):
    layers = []
    assert padding_mode == 'zeros' or padding_mode == 'replicate'

    if padding_mode == 'replicate' and padding > 0:
        assert isinstance(padding, int)
        layers.append(nn.ReflectionPad2d(padding))
        padding = 0

    layers.append(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=bias))
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_planes))
    if relu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class ChannelPool2d(nn.Module):

    def __init__(self, in_channels, out_channels, method='max'):
        super().__init__()

        pool_size = in_channels // out_channels
        assert in_channels % out_channels == 0

        if method == "avg":
            self.pool = self.avg_pool2d
        elif method == "max":
            self.pool = self.max_pool2d
        elif method == 'minmax':
            assert in_channels % (pool_size * 2) == 0
            pool_size = pool_size * 2
            self.pool = self.minmax_pool2d
        else:
            raise NotImplementedError

        self.pool_size = pool_size


    def avg_pool2d(self, x, p):
        x = F.avg_pool3d(x, kernel_size=(p, 1, 1))
        return x

    def max_pool2d(self, x, p):
        x = F.max_pool3d(x, kernel_size=(p, 1, 1))
        return x

    def minmax_pool2d(self, x, p):
        x_min = -F.max_pool3d(-x, kernel_size=(p, 1, 1))
        x_max = F.max_pool3d(x, kernel_size=(p, 1, 1))
        x = torch.cat((x_max, x_min), dim=1)
        return x

    def forward(self, x):
        p = self.pool_size
        n, c, h, w = x.shape
        x = x.view(n, c // p, p, h, w)
        x = self.pool(x, p)
        x = x.view(n, -1, h, w)
        return x


class InstanceL2Norm(nn.Module):
    """Instance L2 normalization.
    """
    def __init__(self, size_average=True, eps=1e-5, scale=1.0):
        super().__init__()
        self.size_average = size_average
        self.eps = eps
        self.scale = scale

    def forward(self, input):
        if self.size_average:
            return input * (self.scale * ((input.shape[1] * input.shape[2] * input.shape[3]) / (
                        torch.sum((input * input).view(input.shape[0], 1, 1, -1), dim=3, keepdim=True) + self.eps)).sqrt())
        else:
            return input * (self.scale / (torch.sum((input * input).view(input.shape[0], 1, 1, -1), dim=3, keepdim=True) + self.eps).sqrt())


def residual_basic_block(feature_dim=256, num_blocks=1, num_post_blocks=0, l2norm=True, final_conv=False, norm_scale=1.0, out_dim=None,
                         interp_cat=False, final_relu=False, init_pool=False, final_proj='conv', dropout_module=nn.Dropout2d, dropout_p=0.0):
    """Construct a network block based on the BasicBlock used in ResNet 18 and 34."""
    if out_dim is None:
        out_dim = feature_dim
    feat_layers = []
    if interp_cat:
        feat_layers.append(InterpCat())
    if init_pool:
        feat_layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    for i in range(num_blocks):
        odim = feature_dim if i < num_blocks - 1 + int(final_conv) else out_dim
        feat_layers.append(BasicBlock(feature_dim, odim))

    if final_conv:
        if dropout_p > 0.0:
            feat_layers.append(dropout_module(p=dropout_p))
        if final_proj == 'conv':
            feat_layers.append(nn.Conv2d(feature_dim, out_dim, kernel_size=3, padding=1, bias=False))
        else:
            feat_layers.append(ChannelPool2d(feature_dim, out_dim, method=final_proj))
            feat_layers.append(nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False))
        if final_relu:
            feat_layers.append(nn.ReLU(inplace=True))
    if l2norm:
        feat_layers.append(InstanceL2Norm(scale=norm_scale))

    for i in range(num_post_blocks):
        feat_layers.append(BasicBlock(out_dim, out_dim))

    return nn.Sequential(*feat_layers)


class ResBlockMultilayer(nn.Module):

    def __init__(self, feature_dims=(512, 512), num_blocks=1, l2norm=True, final_conv=False, norm_scale=1.0,
                 out_dim=None, sizing_layer='layer2', layer_names=('layer2', 'layer3'), final_relu=False,
                 final_proj='conv', init_pool=False):
        super().__init__()

        feature_dim = sum(feature_dims)
        out_dim = feature_dim if out_dim is None else out_dim
        self.layer_names = layer_names
        self.ref_tensor = self.layer_names.index(sizing_layer)

        feat_layers = []
        if init_pool:
            feat_layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        for i in range(num_blocks):
            odim = feature_dim if i < num_blocks - 1 + int(final_conv) else out_dim
            feat_layers.append(BasicBlock(feature_dim, odim))
        if final_conv:
            if final_proj == 'conv':
                feat_layers.append(nn.Conv2d(feature_dim, out_dim, kernel_size=3, padding=1, bias=False))
            else:
                feat_layers.append(ChannelPool2d(feature_dim, out_dim, method=final_proj))
                feat_layers.append(nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False))
            if final_relu:
                feat_layers.append(nn.ReLU(inplace=True))
        if l2norm:
            feat_layers.append(InstanceL2Norm(scale=norm_scale))

        self.layers = nn.Sequential(*feat_layers)

    def forward(self, x):
        if len(self.layer_names) > 1:
            x = adaptive_cat([x[L] for L in self.layer_names], ref_tensor=self.ref_tensor, dim=1)
        x = self.layers(x)
        return x


def interpolate(x, sz):
    """Interpolate 4D tensor x to size sz."""
    sz = sz.tolist() if torch.is_tensor(sz) else sz
    return F.interpolate(x, sz, mode='bilinear', align_corners=False) if x.shape[-2:] != sz else x


class InterpCat(nn.Module):
    """Interpolate and concatenate features of different resolutions."""

    def forward(self, input):
        if isinstance(input, (dict, OrderedDict)):
            input = list(input.values())

        output_shape = None
        for x in input:
            if output_shape is None or output_shape[0] > x.shape[-2]:
                output_shape = x.shape[-2:]

        return torch.cat([interpolate(x, output_shape) for x in input], dim=-3)


def adaptive_cat(seq, dim=0, ref_tensor=0):
    sz = seq[ref_tensor].shape[-2:]
    t = torch.cat([interpolate(t, sz) for t in seq], dim=dim)
    return t
