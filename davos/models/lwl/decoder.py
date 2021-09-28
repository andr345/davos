from __future__ import division
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
import torch.nn.functional as F

from davos.models.backbone.resnet import BasicBlock
from davos.models.blocks import ChannelPool2d
from davos.models.lwl.utils import adaptive_cat, interpolate

from davos.lib import TensorDict
from davos.models.lwl.upsampler import ProjectUp, VOSRaftUpsampler, VOSPRaftUpsampler


def conv(ic, oc, ksize, bias=True, dilation=1, stride=1):
    padding = _pair(int(ksize // 2 + ((ksize - 1) * (dilation - 1)) / 2))
    return nn.Conv2d(ic, oc, ksize, padding=padding, bias=bias, dilation=dilation, stride=stride)


def relu(negative_slope=0.0, inplace=False):
    return nn.LeakyReLU(negative_slope, inplace=inplace)


class TSE(nn.Module):
    def __init__(self, fc, ic, oc, reduction_method='conv', dropout_module=nn.Dropout2d, dropout_p=0.0, variant='3*conv'):
        super().__init__()

        reduce = []

        if dropout_p > 0.0:
            reduce += [dropout_module(p=dropout_p)]

        # Reduce number of feature dimensions
        if reduction_method == 'conv':
            reduce += [conv(fc, oc, 1), relu(), conv(oc, oc, 1)]
        elif reduction_method in ('avg', 'max', 'minmax'):
            pool = ChannelPool2d(fc, oc, method=reduction_method)
            reduce += [pool, conv(oc, oc, 1), relu(), conv(oc, oc, 1)]
        else:
            raise ValueError(f"reduction_method={reduction_method} is not defined")

        self.reduce = nn.Sequential(*reduce)

        nc = ic + oc
        if variant == '3*conv':
            self.transform = nn.Sequential(conv(nc, nc, 3), relu(),
                                           conv(nc, nc, 3), relu(),
                                           conv(nc, oc, 3), relu())
        elif variant == 'resblock':
            self.transform = nn.Sequential(BasicBlock(nc, nc),
                                           conv(nc, oc, 3), relu())
        elif variant == '2*resblock':
            self.transform = nn.Sequential(BasicBlock(nc, nc),
                                           BasicBlock(nc, nc),
                                           conv(nc, oc, 3), relu())
        else:
            raise NotImplementedError

    def forward(self, ft, score, x=None):
        h = self.reduce(ft)
        hpool = F.adaptive_avg_pool2d(h, (1, 1)) if x is None else x
        h = adaptive_cat((h, score), dim=1, ref_tensor=0)
        h = self.transform(h)
        return h, hpool


class CAB(nn.Module):
    def __init__(self, oc, deepest):
        super().__init__()

        self.convreluconv = nn.Sequential(conv(2 * oc, oc, 1), relu(), conv(oc, oc, 1))
        self.deepest = deepest

    def forward(self, deeper, shallower):

        shallow_pool = F.adaptive_avg_pool2d(shallower, (1, 1))
        deeper_pool = deeper if self.deepest else F.adaptive_avg_pool2d(deeper, (1, 1))
        global_pool = (shallow_pool, deeper_pool)
        global_pool = torch.cat(global_pool, dim=1)
        conv_1x1 = self.convreluconv(global_pool)
        inputs = shallower * torch.sigmoid(conv_1x1)
        out = inputs + interpolate(deeper, inputs.shape[-2:])

        return out


class RRB(nn.Module):
    def __init__(self, oc, k=3, use_bn=False):
        super().__init__()

        self.conv1x1 = conv(oc, oc, 1)
        if use_bn:
            self.bblock = nn.Sequential(conv(oc, oc, k), nn.BatchNorm2d(oc), relu(), conv(oc, oc, k, bias=False))
        else:
            self.bblock = nn.Sequential(conv(oc, oc, k), relu(), conv(oc, oc, k, bias=False))  # Basic block

    def forward(self, x):
        h = self.conv1x1(x)
        return F.relu(h + self.bblock(h))


class Upsampler(nn.Module):
    def __init__(self, in_channels=64, mid_channels=64, out_channels=1):
        super().__init__()

        self.conv1 = conv(in_channels, mid_channels // 2, 3)
        self.conv2 = conv(mid_channels // 2, out_channels, 3)

    def forward(self, x, image_size):
        x = F.interpolate(x, (2*x.shape[-2], 2*x.shape[-1]), mode='bicubic', align_corners=False)
        x = F.relu(self.conv1(x))
        x = F.interpolate(x, image_size[-2:], mode='bicubic', align_corners=False)
        x = self.conv2(x)
        return x


class LWTLDecoder(nn.Module):

    def append_coord_map(self, x):
        if self.coord_map is None:
            return x
        i = tuple(x.shape[-2:]) + (x.device,)
        if i not in self.coord_maps:
            self.coord_maps[i] = self.coord_map(i[:2]).to(x.device)
        cm = self.coord_maps[i]
        cm = cm.expand(x.shape[0], *cm.shape[1:])

        return torch.cat((x, cm), dim=1)

    """ Decoder module """
    def __init__(self, in_channels=1, out_channels=32, ft_channels=None, label_channels=1, feat_proj='conv',
                 coord_map=None, use_bn=False, dropout_p=0.0, tse_variant='3*conv'):

        super().__init__()

        assert ft_channels is not None
        self.ft_channels = ft_channels
        self.label_channels = label_channels

        cmc = coord_map.num_channels if coord_map is not None else 0
        self.coord_map = coord_map
        self.coord_maps = TensorDict()

        self.TSE = nn.ModuleDict()
        self.RRB1 = nn.ModuleDict()
        self.CAB = nn.ModuleDict()
        self.RRB2 = nn.ModuleDict()
        self.proj = nn.ModuleDict()

        ic = in_channels

        oc = {'layer1': 1, 'layer2': 2, 'layer3': 2, 'layer4': 4}
        out_feature_channels = {}

        if 'layer4' in ft_channels.keys():
            last_layer = 'layer4'
        else:
            last_layer = 'layer3'

        prev_layer = None

        for L, fc in self.ft_channels.items():

            if L == 'conv1':
                continue

            nc = dict(layer1=1, layer2=2, layer3=2, layer4=4)[L] * out_channels  # Number of layer channels
            assert oc[L]*out_channels == nc

            if not L == last_layer:
                self.proj[L] = nn.Sequential(conv(oc[prev_layer]*out_channels + cmc, oc[L]*out_channels, 1), relu())

            self.TSE[L] = TSE(fc + cmc, ic, oc[L] * out_channels, reduction_method=feat_proj, dropout_p=dropout_p, variant=tse_variant)
            self.RRB1[L] = RRB(oc[L] * out_channels + cmc, use_bn=use_bn)
            self.CAB[L] = CAB(oc[L] * out_channels + cmc, L == last_layer)
            self.RRB2[L] = RRB(oc[L] * out_channels + cmc, use_bn=use_bn)
            out_feature_channels['{}_dec'.format(L)] = oc[L]*out_channels
            prev_layer = L

        self.project = Upsampler(out_channels + cmc, out_channels, label_channels)
        self._out_feature_channels = out_feature_channels
        self.refine = None

    def out_feature_channels(self):
        return self._out_feature_channels

    def forward(self, scores, features, image_size, output_layers=(), num_objects=None):
        if num_objects is None:
            assert scores.dim() == 5  # frames, seq, ch, h, w
        else:
            assert scores.dim() == 6  # frames, seq, obj, ch, h, w
        outputs = OrderedDict()

        scores = scores.view(-1, *scores.shape[-3:])

        x = None
        for i, L in enumerate(self.ft_channels):
            if L == 'conv1':
                continue

            ft = features[L]
            ft = self.append_coord_map(ft)

            s = interpolate(scores, ft.shape[-2:])


            if not x is None:
                x = self.proj[L](x)

            h, hpool = self.TSE[L](ft, s, x)
            h = self.append_coord_map(h)
            hpool = self.append_coord_map(hpool)

            h = self.RRB1[L](h)
            h = self.CAB[L](hpool, h)
            x = self.RRB2[L](h)

            if '{}_dec'.format(L) in output_layers:
                outputs['{}_dec'.format(L)] = x

        #x = self.project(x, image_size)

        if isinstance(self.project, Upsampler):
            logits = self.project(x, image_size)
        elif isinstance(self.project, ProjectUp):
            logits = self.project(x, features)
        elif isinstance(self.project, (VOSRaftUpsampler, VOSPRaftUpsampler)):
            logits = self.project(x, features)

        return logits, outputs
