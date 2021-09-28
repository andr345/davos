import torch
import torch.nn as nn
import torch.nn.functional as F

from .interp_weights_est import Simple, UNet
from .nconv_modules import NConvUNet
from .pac_upsampler import PacJointUpsample, DJIF

num_channels = {'image': 3, 'conv1': 64, 'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048,
                     'decoder': 64, 'tm': 512, 'tms': 512+16}


def get_feat_num_channels(feat_name):
    return num_channels[feat_name]


def get_upsampler(enc_feat_layer, in_ch=1, upsampler_name='nconvupsampler', final_upsampling_scale=4):

    #upsampler_name = args.final_upsampling.lower()
    guidance_ch = get_feat_num_channels(enc_feat_layer)

    if upsampler_name == 'nconvupsampler':
        # Define the normalized convolution interpolation network
        #TODO: Normalized convolution network only supports in_ch of 1, add support for more channels using groups.
        interpolation_net = NConvUNet(in_ch=1, channels_multiplier=2,
                                      num_downsampling=1,
                                      encoder_filter_sz=5,
                                      decoder_filter_sz=3,
                                      out_filter_sz=1,
                                      use_bias=False,
                                      data_pooling='conf_based',
                                      shared_encoder=True,
                                      use_double_conv=False,
                                      pos_fn='SoftPlus', groups=1)

        # Define the weights estimation network
        weights_est_net = None

        # Add the number of input channels at the beginning of num_ch
        num_channels = [64, 32]
        final_upsampling_use_data_for_guidance = True
        if final_upsampling_use_data_for_guidance:
            num_channels.insert(0, guidance_ch + in_ch)
        else:
            num_channels.insert(0, guidance_ch)

        #if args.weights_est_net.lower() == 'simple':
        weights_est_net = Simple(num_ch=num_channels, out_ch=in_ch,
                                 filter_sz=[3,3,1], dilation=[1,1,1],
                                 final_act=nn.Sigmoid())
        #elif args.weights_est_net.lower() == 'unet':
        #weights_est_net = UNet(num_ch=num_channels, out_ch=in_ch, final_act=nn.Sigmoid())

        # Define the NConv upsampler
        upsampler = NConvUpsampler(enc_feat_layer=enc_feat_layer, scale=4, interpolation_net=interpolation_net,
                                   weights_est_net=weights_est_net,
                                   use_data_for_guidance=final_upsampling_use_data_for_guidance,
                                   channels_to_batch=True,
                                   use_residuals=False,
                                   est_on_high_res=False)

    elif upsampler_name == 'bilinear':
        upsampler = Bilinear(final_upsampling_scale)

    elif upsampler_name == 'pacjointupsamplefull':
        upsampler = PacJointUpsampleFull(scale=final_upsampling_scale, in_ch=1, guidance_ch=guidance_ch)

    elif upsampler_name == 'djiforiginal':
        upsampler = DjifOriginal(scale=final_upsampling_scale, in_ch=1, guidance_ch=guidance_ch)
    elif upsampler_name == 'raftupsampler':
        upsampler = RaftUpsampler(scale=final_upsampling_scale, in_ch=in_ch, guidance_ch=guidance_ch)
    else:
        raise NotImplementedError('Upsampler `{}` is not implemented!'.format(upsampler_name))

    return upsampler


class ProjectUp(nn.Module):
    def __init__(self, guidance, in_ch=64, out_ch=1):
        super().__init__()
        self.project = nn.Sequential(conv(in_ch, in_ch // 2, 3),
                                     nn.ReLU(inplace=True),
                                     conv(in_ch // 2, out_ch, 3))
        self.up = get_upsampler(guidance, in_ch=2)

    def forward(self, x, features):
        features['decoder'] = x
        x_proj = self.project(x)
        return self.up(x_proj, features)


class NConvUpsampler(torch.nn.Module):
    def __init__(self, enc_feat_layer, scale=None, size=None, interpolation_net=None, weights_est_net=None, use_data_for_guidance=True,
                 channels_to_batch=True, use_residuals=False, est_on_high_res=False, use_decoder_for_guidance=False,
):
        """
        An upsampling layer using Normalized CNN with an input weights estimation network.
        Either `scale` or `size` needs to be specified.

        Args:
            scale: The uspampling factor.
            size: The desired size of the output.
            interpolation_net: Interpolation network. Needs to be an object of `NConvUNetFull`.
            weights_est_net: Weights estimation network.
            use_data_for_guidance: Either to use the low-resolution data as input to the weights estimation network with
                                   the guidance data.
            channels_to_batch: Either to reshape data tensor to B*Cx1xHxW before performing interpolation.
        """
        super().__init__()
        self.__name__ = 'NConvUpsampler'
        # Check the validity of arguments
        if scale is None and size is None:
            raise ValueError('Either scale or size needs to be set!')
        elif scale is not None and size is not None:
            raise ValueError('You can set either scale or size at a time!')
        elif scale is not None and size is None:
            if isinstance(scale, tuple):
                self.scaleW = float(scale[1])
                self.scaleH = float(scale[0])
            elif isinstance(scale, int):
                self.scaleW = self.scaleH = float(scale)
            else:
                raise ValueError('Scale value can be tuple or integer only!')
        elif scale is None and size is not None:
            if isinstance(size, tuple):
                self.osize = size
                self.scaleW = self.scaleH = None
            else:
                raise ValueError('Size has to be a tuple!')

        # Interpolation network must be provided and from `NConv` family
        if interpolation_net is None:
            raise ValueError('An interpolation network mush be provided!')
        else:
            assert 'NConv' in interpolation_net.__name__, 'Only `NConv` interpolaion networks are supported!'
            self.interpolation_net = interpolation_net
            # Get the number of data input channels
            self.data_ich = self.interpolation_net._modules['nconv_in'].in_channels

        if weights_est_net is None:  # No weights estimation network provided, use binary weights mask
            self.weights_est_net = self.get_binary_weights
            self.guidance_ich = self.data_ich
        else:
            self.weights_est_net = weights_est_net
            # Get the number of guidance input channels
            self.guidance_ich = self.weights_est_net.in_ch

        self.use_data_for_guidance = use_data_for_guidance
        self.channels_to_batch = channels_to_batch
        self.use_residuals = use_residuals
        self.est_on_high_res = est_on_high_res
        self.enc_feat_layer = enc_feat_layer
        self.use_decoder_for_guidance = use_decoder_for_guidance

        # If data is used for guidance, check that the guidance network has the right number of in_channels
        if self.use_data_for_guidance:
            assert(self.guidance_ich >= self.data_ich)

        print('NC parameters: ', sum(p.numel() for p in self.parameters() if p.requires_grad))


    @staticmethod
    def get_binary_weights(t):
        return (t > 0).float()

    def forward(self, x_lowres, features):

        x_guidance = features[self.enc_feat_layer]
        x_highres = self.get_out_tensor(x_lowres)

        # Prepare guidance data
        if self.est_on_high_res:
            x_data_for_guidance = x_highres
        else:
            x_guidance = F.interpolate(x_guidance, x_lowres.size()[2:], mode='area')  # Downsample the guidance
            x_data_for_guidance = x_lowres

        # Feed guidance data to weights estimation network
        if self.use_data_for_guidance:
            w_lowres = self.weights_est_net(torch.cat((x_data_for_guidance, x_guidance), 1))
        else:
            w_lowres = self.weights_est_net(x_guidance)

        if self.est_on_high_res:
            w_highres = w_lowres
        else:
            w_highres = self.get_out_tensor(w_lowres)

        ib, ic, oh, ow = x_highres.shape

        # Perform interpolation using NConv
        if self.channels_to_batch:
            output, _ = self.interpolation_net((x_highres.view(ib * ic, 1, oh, ow), w_highres.view(ib * ic, 1, oh, ow)))
        else:
            output, _ = self.interpolation_net((x_highres, w_highres))

        output = output.view(ib, ic, oh, ow)

        if self.use_residuals:
            output[x_highres > 0] = x_highres[x_highres > 0]

        return output

    def get_out_tensor(self, inp):
        b, ic, ih, iw = inp.shape

        if self.scaleH is None and self.scaleW is None:  # Size was provided
            oh = self.osize[0]
            ow = self.osize[1]

            # Calculate the scaling factor
            sH = oh / ih
            sW = ow / iw
        else:
            sH = int(self.scaleH)
            sW = int(self.scaleW)
            oh = round(ih * sH)
            ow = round(iw * sW)

        out_t = torch.zeros((b, ic, oh, ow), dtype=inp.dtype).to(inp.device)
        """
        ix = torch.arange(iw).to(inp.device).float()
        iy = torch.arange(ih).to(inp.device).float()

        ox = torch.round(ix * sW).long()
        oy = torch.round(iy * sH).long()

        gy, gx = torch.meshgrid([oy, ox])

        out_t[:, :, gy+4, gx+4] = inp
        """

        out_t[:, :, sH // 2::sH, sW // 2::sW] = inp

        return out_t


class Bilinear(nn.Module):
    def __init__(self, scale=None):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)

    def forward(self, x, *argv):
        return self.up(x)


class PacJointUpsampleFull(nn.Module):
    def __init__(self, scale=None, in_ch=1, guidance_ch=3):
        super().__init__()

        self.up = PacJointUpsample(factor=scale, channels=in_ch, guide_channels=guidance_ch)

    def forward(self, x, guide):
        return self.up(x, guide)


class DjifOriginal(nn.Module):
    def __init__(self, scale=None, in_ch=1, guidance_ch=3):
        super().__init__()

        self.up = DJIF(factor=scale, channels=in_ch, guide_channels=guidance_ch)

    def forward(self, x, guide):
        return self.up(x, guide)


class RaftUpsampler(nn.Module):
    def __init__(self, scale=None, in_ch=1, guidance_ch=3):
        super(RaftUpsampler, self).__init__()
        self.scale = scale
        self.win = 5
        self.mask = nn.Sequential(
            nn.Conv2d(guidance_ch, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, scale*scale*self.win*self.win, 1, padding=0))

    def forward(self, flow, guidance):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        guidance = F.interpolate(guidance, (H,W), mode='area')  # Downsample the guidance
        mask = self.mask(guidance)

        mask = mask.view(N, 1, self.win**2, self.scale, self.scale, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(flow, [self.win,self.win], padding=1)
        up_flow = up_flow.view(N, 2, self.win**2, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, self.scale*H, self.scale*W)


def conv(ic, oc, ksize, bias=True, dilation=1, stride=1):
    return nn.Conv2d(ic, oc, ksize, padding=ksize // 2, bias=bias, dilation=dilation, stride=stride)


class VOSRaftUpsampler(nn.Module):
    def __init__(self, enc_feat_layer, in_channels=64, out_channels=64):
        super(VOSRaftUpsampler, self).__init__()

        self.scale = 4
        self.win = 3
        self.enc_feat_layer = enc_feat_layer
        self.out_channels= out_channels

        self.get_seg_mask = nn.Sequential(
            conv(in_channels, in_channels // 2, 3),
            nn.ReLU(inplace=True),
            conv(in_channels // 2, out_channels, 3)
        )

        guidance_ich = get_feat_num_channels(enc_feat_layer)+256

        self.guidance_weights = nn.Sequential(
            conv(guidance_ich, 256, 3),
            nn.ReLU(inplace=True),
            conv(256, self.scale * self.scale * self.win ** 2, 1),
            #conv(256, 128, 3),
            #nn.ReLU(inplace=True),
            #conv(128, self.scale * self.scale * self.win ** 2, 1)
        )

        print('RAFT parameters: ', sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x, features):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        features['decoder'] = x
        seg_mask = self.get_seg_mask(x)

        guidance_feat = features[self.enc_feat_layer]

        # Conv1
        #conv1_ft = F.interpolate(features['conv1'], x.size()[2:], mode='area')  # Downsample the guidance
        #guidance_feat = torch.cat((features[self.enc_feat_layer], conv1_ft), 1)

        # Layer 1
        guidance_feat = torch.cat((features[self.enc_feat_layer], features['layer1']), 1)

        weights = self.guidance_weights(guidance_feat)

        N, _, H, W = x.shape

        weights = weights.view(N, 1, self.win**2, self.scale, self.scale, H, W)
        weights = torch.softmax(weights, dim=2)

        seg_mask = F.unfold(seg_mask, [self.win, self.win], padding=self.win//2)
        seg_mask = seg_mask.view(N, self.out_channels, self.win**2, 1, 1, H, W)

        seg_mask = torch.sum(weights * seg_mask, dim=2)
        seg_mask = seg_mask.permute(0, 1, 4, 2, 5, 3)
        seg_mask = seg_mask.reshape(N, self.out_channels, self.scale*H, self.scale*W)
        return seg_mask


class VOSPRaftUpsampler(nn.Module):
    def __init__(self, enc_feat_layer, in_channels=64, out_channels=64):
        super().__init__()

        self.scale = 4
        self.win = 3

        self.enc_feate_layer = enc_feat_layer

        self.project1 = nn.Sequential(
            conv(in_channels, in_channels // 2, 3),
            nn.ReLU(inplace=True),
            conv(in_channels // 2, 16, 3),
            nn.ReLU(inplace=True),
        )

        self.project2 = nn.Sequential(
            conv(16, 8, 5),
            nn.ReLU(inplace=True),
            conv(8, out_channels, 3),
        )

        guidance_ich = get_feat_num_channels(enc_feat_layer)
        if self.enc_feate_layer == 'image':
            self.guidance_weights = nn.Sequential(
                conv(guidance_ich, 32, 3),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
                conv(32, self.scale * self.scale * 9, 1),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        elif self.enc_feate_layer in ['layer1', 'decoder']:
            self.guidance_weights = nn.Sequential(
                conv(guidance_ich, guidance_ich//2, 3),
                nn.ReLU(inplace=True),
                conv(guidance_ich//2, self.scale*self.scale*self.win**2, 1)
            )

        print('pRAFT parameters: ', sum(p.numel() for p in self.parameters() if p.requires_grad))


    def forward(self, x, features):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = x.shape
        x_proj = self.project1(x)  # N x 16 x H x W

        if self.enc_feate_layer == 'decoder':
            guidance_feat = x
        else:
            guidance_feat = features[self.enc_feate_layer]

        weights = self.guidance_weights(guidance_feat)

        weights = weights.view(N, 1, self.win**2, self.scale, self.scale, H, W)
        weights = torch.softmax(weights, dim=2)
        x_proj = F.unfold(x_proj, [self.win, self.win], padding=self.win//2)
        x_proj = x_proj.view(N, 16, self.win**2, 1, 1, H, W)

        x_proj = torch.sum(weights * x_proj, dim=2)

        x_proj = x_proj.permute(0, 1, 4, 2, 5, 3).reshape(N, 16, self.scale*H, self.scale*W)

        x_out = self.project2(x_proj)

        return x_out


class CatUpsampler(nn.Module):
    def __init__(self, enc_feat_layer, in_channels=64, out_channels=64):
        super().__init__()
        self.enc_feate_layer = enc_feat_layer
        guidance_ich = get_feat_num_channels(enc_feat_layer)
        self.conv1 = conv(in_channels+guidance_ich, out_channels // 2, 3)
        self.conv2 = conv(out_channels // 2, 1, 3)

        print('CatUpsampler: ', sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x, features, image_size):
        guidance_feat = features[self.enc_feate_layer]
        x = F.interpolate(x, (2*x.shape[-2], 2*x.shape[-1]), mode='bicubic', align_corners=False)
        x = F.relu(self.conv1(torch.cat((x, guidance_feat), 1)))
        x = F.interpolate(x, image_size[-2:], mode='bicubic', align_corners=False)
        x = self.conv2(x)
        return x

if __name__ == '__main__':
    pass
