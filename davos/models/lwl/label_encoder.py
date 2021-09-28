import torch
import torch.nn as nn
import math
from davos.models.backbone.resnet import BasicBlock, get_out_channels
from ..blocks import conv_block
from davos.models.lwl.utils import interpolate


class ResidualDS16SW(nn.Module):
    """ Outputs the few-shot learner label and spatial importance weights given the segmentation mask """
    def __init__(self, in_channels, layer_dims, use_bn=True):
        super().__init__()
        ic = in_channels
        self.conv_block = conv_block(ic, layer_dims[0], kernel_size=3, stride=2, padding=1, batch_norm=use_bn)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ds1 = nn.Conv2d(layer_dims[0], layer_dims[1], kernel_size=3, padding=1, stride=2)
        self.res1 = BasicBlock(layer_dims[0], layer_dims[1], stride=2, downsample=ds1, use_bn=use_bn)

        ds2 = nn.Conv2d(layer_dims[1], layer_dims[2], kernel_size=3, padding=1, stride=2)
        self.res2 = BasicBlock(layer_dims[1], layer_dims[2], stride=2, downsample=ds2, use_bn=use_bn)

        self.label_pred = conv_block(layer_dims[2], layer_dims[3], kernel_size=3, stride=1, padding=1,
                                     relu=True, batch_norm=use_bn)

        self.samp_w_pred = nn.Conv2d(layer_dims[2], layer_dims[3], kernel_size=3, padding=1, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.samp_w_pred.weight.data.fill_(0)
        self.samp_w_pred.bias.data.fill_(1)

    def zero(self, shape, device):
        return self(torch.zeros(shape, device=device))

    def forward(self, label_mask, feat_clf=None, feat_bbone=None):

        F, S = label_mask.shape[:2]
        label_mask = label_mask.view(-1, *label_mask.shape[-3:])

        out = self.pool(self.conv_block(label_mask))
        out = self.res2(self.res1(out))

        label_enc = self.label_pred(out)
        sample_w = self.samp_w_pred(out)

        label_enc = label_enc.view(F, S, *label_enc.shape[-3:])
        sample_w = sample_w.view(F, S, *sample_w.shape[-3:])

        return label_enc, sample_w


class ResidualDS16FeatSWBoxCatMultiBlock(nn.Module):
    def __init__(self, layer_dims, feat_dim,  use_final_relu=True, use_gauss=True, use_bn=True,
                 non_default_init=True, init_bn=1, gauss_scale=0.25, final_bn=True):
        super().__init__()
        in_layer_dim = (feat_dim+1,) + tuple(list(layer_dims)[:-2])
        out_layer_dim = tuple(list(layer_dims)[:-1])
        self.use_gauss = use_gauss
        res = []
        for in_d, out_d in zip(in_layer_dim, out_layer_dim):
            ds = nn.Conv2d(in_d, out_d, kernel_size=3, padding=1, stride=1)
            res.append(BasicBlock(in_d, out_d, stride=1, downsample=ds, use_bn=use_bn))

        self.res = nn.Sequential(*res)
        self.label_pred = conv_block(layer_dims[-2], layer_dims[-1], kernel_size=3, stride=1, padding=1,
                                     relu=use_final_relu, batch_norm=final_bn)
        self.gauss_scale = gauss_scale
        if non_default_init:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(init_bn)
                    m.bias.data.zero_()

    def bbox_to_mask(self, bbox, sz):
        mask = torch.zeros((bbox.shape[0],1,*sz), dtype=torch.float32, device=bbox.device)
        for i, bb in enumerate(bbox):
            x1, y1, w, h = list(map(int, bb))
            x1 = int(x1+0.5)
            y1 = int(y1+0.5)
            h = int(h+0.5)
            w = int(w+0.5)
            mask[i, :, y1:(y1+h), x1:(x1+w)] = 1.0
        return mask

    def bbox_to_gauss(self, bbox, sz):
        mask = torch.zeros((bbox.shape[0],1,*sz), dtype=torch.float32, device=bbox.device)
        x_max, y_max = sz[-1], sz[-2]
        for i, bb in enumerate(bbox):
            x1, y1, w, h = list(map(int, bb))
            cx, cy = x1+w/2, y1+h/2
            xcoords = torch.arange(0, x_max).unsqueeze(dim=0).to(bbox.device).float()
            ycoords = torch.arange(0, y_max).unsqueeze(dim=0).T.to(bbox.device).float()
            d_xcoords = xcoords - cx
            d_ycoords = ycoords - cy
            dtotsqr = d_xcoords**2/(self.gauss_scale*w)**2 + d_ycoords**2/(self.gauss_scale*h)**2
            mask[i,0] = torch.exp(-0.5*dtotsqr)
        return mask

    def forward(self, bb, feat, sz):

        if self.use_gauss:
            label_mask = self.bbox_to_gauss(bb, sz[-2:])
        else:
            label_mask = self.bbox_to_mask(bb, sz[-2:])

        label_shape = label_mask.shape
        label_mask = label_mask.view(-1, 1, *label_mask.shape[-2:])

        feat = feat.view(-1, *feat.shape[-3:])
        feat_mask_enc = torch.cat([feat, interpolate(label_mask, feat.shape[-2:])], dim=1)
        out = self.res(feat_mask_enc)

        label_enc = self.label_pred(out)

        label_enc = label_enc.view(label_shape[0], label_shape[1], *label_enc.shape[-3:])

        return label_enc


class ResidualDS8SW(nn.Module):
    def __init__(self, layer_dims, in_channels=1, use_bn=True, out_stride=16, disable_sw=False):
        super().__init__()

        ic = in_channels

        s1, s2, s3 = {4: (2, 1, 1), 8: (2, 2, 1), 16: (2, 2, 2)}[out_stride]
        # s3 = 1 if out_stride in (4, 8) else 2

        self.conv_block = conv_block(ic, layer_dims[0], kernel_size=3, stride=s1, padding=1, batch_norm=use_bn)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ds1 = nn.Conv2d(layer_dims[0], layer_dims[1], kernel_size=3, padding=1, stride=s2)
        self.res1 = BasicBlock(layer_dims[0], layer_dims[1], stride=s2, downsample=ds1, use_bn=use_bn)

        ds2 = nn.Conv2d(layer_dims[1], layer_dims[2], kernel_size=3, padding=1, stride=s3)
        self.res2 = BasicBlock(layer_dims[1], layer_dims[2], stride=s3, downsample=ds2, use_bn=use_bn)

        self.label_pred = conv_block(layer_dims[2], layer_dims[3], kernel_size=3, stride=1, padding=1,
                                     relu=True, batch_norm=use_bn)

        self.disable_sw = disable_sw
        if not disable_sw:
            self.samp_w_pred = nn.Conv2d(layer_dims[2], layer_dims[3], kernel_size=3, padding=1, stride=1)

        self.out_channels = get_out_channels(self.label_pred)
        self.out_stride = out_stride

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if not disable_sw:
            self.samp_w_pred.weight.data.fill_(0)
            self.samp_w_pred.bias.data.fill_(1)

    def zero(self, shape, device):
        return self(torch.zeros(shape, device=device))

    def forward(self, label_mask, feat_clf=None, feat_bbone=None):

        F, S = label_mask.shape[:2]  # shape = (frames, seq, c, h, w)
        label_mask = label_mask.view(-1, *label_mask.shape[-3:])

        out = self.pool(self.conv_block(label_mask))
        out = self.res2(self.res1(out))

        label_enc = self.label_pred(out)
        label_enc = label_enc.view(F, S, *label_enc.shape[-3:])

        if not self.disable_sw:
            sample_w = self.samp_w_pred(out)
            sample_w = sample_w.view(F, S, *sample_w.shape[-3:])
            return label_enc, sample_w
        else:
            return label_enc, torch.ones(F, S, label_enc.shape[-3], 1, 1, device=label_enc.device)
