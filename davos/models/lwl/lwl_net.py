import math
import torch
import torch.nn as nn
from collections import OrderedDict

import davos.models.blocks
import davos.models.lwl.linear_filter as target_clf
import davos.models.lwl.initializer as seg_initializer
import davos.models.lwl.label_encoder as seg_label_encoder
import davos.models.lwl.loss_residual_modules as loss_residual_modules
import davos.models.lwl.decoder as lwtl_decoder
import davos.models.backbone as backbones
import davos.models.backbone.resnet_mrcnn as mrcnn_backbones
from davos.models import steepestdescent
from davos.sys.model_constructor import model_constructor
from davos.lib import TensorList, TensorDict


class Memory:

    def __init__(self):

        self.mask_enc = []
        self.mask_sw = []
        self.feat_tm = []

    def append(self, mask_enc, mask_sw, feat_clf):

        self.feat_tm.append(feat_clf)
        self.mask_enc.append(mask_enc)
        self.mask_sw.append(mask_sw)


    def feats(self):
        return torch.cat(self.feat_tm, dim=0)

    def masks(self):
        return torch.cat(self.mask_enc, dim=0)

    def sw(self):
        return torch.cat(self.mask_sw, dim=0)


class LWLNet(nn.Module):
    def __init__(self, feature_extractor, target_model, decoder, target_model_input_layer, decoder_input_layers, label_encoder=None):
        super().__init__()

        self.feature_extractor = feature_extractor      # Backbone feature extractor F
        self.target_model = target_model                # Target model and the few-shot learner
        self.decoder = decoder                          # Segmentation Decoder
        self.label_encoder = label_encoder              # Few-shot label generator and weight predictor

        self.target_model_input_layer = (target_model_input_layer,) if isinstance(target_model_input_layer,
                                                                                  str) else target_model_input_layer
        self.decoder_input_layers = decoder_input_layers
        self.output_layers = sorted(list(set(self.target_model_input_layer + self.decoder_input_layers)))

    def _optimize_filter(self, memory, tm_filter=None, num_iter=None, **kwargs):
        tm_filter, _, _ = self.target_model.filter_optimizer(
            TensorList([tm_filter]), feat=memory.feats(), label=memory.masks(), sample_weight=memory.sw(), num_iter=num_iter, **kwargs)
        return tm_filter[0]

    def forward(self, train_imgs, test_imgs, train_masks, test_masks, num_refinement_iter=2, return_debug=False):

        num_seqs = train_imgs.shape[1]
        num_train = train_imgs.shape[0]  # number of training frames
        num_test = test_imgs.shape[0]  # number of test frames

        def to_batch(t, reshape=False):
            return t.reshape(-1, *t.shape[-3:]) if reshape else t.view(-1, *t.shape[-3:])
        
        def to_seq(t, frames, sequences):
            if isinstance(t, TensorDict):
                t2 = TensorDict()
                for k, v in t.items():
                    t2[k] = to_seq(v, frames, sequences)
                return t2
            elif t is None:
                return None
            return t.view(frames, sequences, *t.shape[-3:])

        memory = Memory()

        # Extract features
        train_feat_bbone = self.extract_backbone_features(to_batch(train_imgs))
        train_feat_tm = self.extract_target_model_features(train_feat_bbone)  # seq*frames, channels, height, width
        train_feat_tm = to_seq(train_feat_tm, num_train, num_seqs)

        test_feat_bbone = self.extract_backbone_features(to_batch(test_imgs, reshape=True))
        test_feat_tm = self.extract_target_model_features(test_feat_bbone)  # seq*frames, channels, height, width
        test_feat_tm = to_seq(test_feat_tm, num_test, num_seqs)

        # Get few-shot learner label and spatial importance weights
        mask_enc, mask_sw = self.label_encoder(train_masks, train_feat_tm)
        memory.append(mask_enc, mask_sw, train_feat_tm)

        # Obtain the target module parameters using the few-shot learner
        tm_filter, filter_iter, _ = self.target_model.get_filter(train_feat_tm, mask_enc, mask_sw)

        mask_preds_all = []
        mask_preds_enc_all = []

        # Iterate over the test sequence
        for i in range(num_test):
            # Features for the current frame
            test_feat_it = test_feat_tm[i:i+1, ...]
            # test_masks_it = test_masks.view(num_test, num_seqs, 1, *test_masks.shape[-2:])[i:i+1, ...]
            test_feat_bbone_it = {k: to_seq(v, num_test, num_seqs)[i, ...] for k, v in test_feat_bbone.items()}
            # Apply the target model to obtain mask encodings.
            mask_pred_enc = [self.target_model.apply_target_model(f, test_feat_it) for f in filter_iter]
            mask_pred_enc_it = mask_pred_enc[-1]

            # Run decoder to obtain the segmentation mask
            mask_pred, decoder_feat = self.decoder(mask_pred_enc_it, test_feat_bbone_it, test_imgs.shape[-2:])
            mask_pred = mask_pred.view(1, num_seqs, *mask_pred.shape[-3:])
            mask_preds_all.append(mask_pred)

            # Obtain label encoding for the predicted mask in the previous frame
            mask_pred_prob = torch.sigmoid(mask_pred.clone().detach())  # Convert the segmentation scores to probability
            mask_enc, mask_sw = self.label_encoder(mask_pred_prob, test_feat_it)

            if return_debug:
                mask_preds_enc_all.append(mask_pred_enc_it.detach())

            # Extend the training set
            memory.append(mask_enc, mask_sw, test_feat_it)

            # Update the target model
            if (i < (num_test - 1)) and (num_refinement_iter > 0):
                tm_filter = self._optimize_filter(memory, tm_filter, num_iter=num_refinement_iter)

        mask_preds_all = torch.cat(mask_preds_all, dim=0)

        if return_debug:
            # Important: Make sure batch dim=1
            debug_info = dict(
                mask_pred_enc=mask_preds_enc_all[0],
                mask_pred=mask_preds_all[0].unsqueeze(0),
            )
            return mask_preds_all, debug_info

        return mask_preds_all

    def segment_target(self, target_filter, test_feat_tm, test_feat, im_size, gt_mask=None):
        # Classification features
        assert target_filter.dim() == 5     # seq, filters, ch, h, w
        test_feat_tm = test_feat_tm.view(1, 1, *test_feat_tm.shape[-3:])
        if gt_mask is None:
            mask_pred_enc = self.target_model.apply_target_model(target_filter, test_feat_tm)
        else:
            mask_pred_enc = self.label_encoder(gt_mask, test_feat_tm)[0]
        mask_pred, decoder_feat = self.decoder(mask_pred_enc, test_feat, im_size)
        return mask_pred, mask_pred_enc

    def get_backbone_target_model_features(self, backbone_feat):
        # Get the backbone feature block which is input to the target model
        feat = OrderedDict({l: backbone_feat[l] for l in self.target_model_input_layer})
        if len(self.target_model_input_layer) == 1:
            return feat[self.target_model_input_layer[0]]
        return feat

    def extract_target_model_features(self, backbone_feat):
        return self.target_model.extract_target_model_features(self.get_backbone_target_model_features(backbone_feat))

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.output_layers
        return self.feature_extractor(im, layers)


@model_constructor
def steepest_descent_resnet50(filter_size=1, num_filters=1, optim_iter=3, optim_init_reg=0.01,
                              backbone_pretrained=False, clf_feat_blocks=1, clf_feat_post_blocks=0,
                              clf_feat_norm=True, final_conv=False, clf_feat_dropout_p=0.0,
                              out_feature_dim=512,
                              target_model_input_layer='layer3',
                              decoder_input_layers=("layer4", "layer3", "layer2", "layer1",),
                              dec_dropout_p=0.0,
                              detach_length=float('Inf'),
                              label_channels=1,
                              label_encoder_dims=(1, 1),
                              frozen_backbone_layers=(),
                              decoder_mdim=64, filter_groups=1,
                              decoder_type='rofl',
                              decoder_tse_variant='3*conv',
                              use_bn_in_label_enc=True,
                              label_encoder='res_ds16_sw',
                              clf_feat_extractor='resblock',
                              clf_proj='conv', dec_proj='conv',
                              dilation_factors=None,
                              backbone_type='imagenet',
                              basenet=None,
                              upsampler=None,
                              freeze_vos=False):

    # backbone feature extractor F
    if backbone_type == 'imagenet':
        backbone_net = backbones.resnet50(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)
    elif backbone_type == 'mrcnn':
        backbone_net = mrcnn_backbones.resnet50(pretrained=False, frozen_layers=frozen_backbone_layers)
    elif backbone_type == 'resnest-mrcnn':
        from davos.models.backbone.resnest import resnest50
        backbone_net = resnest50(pretrained=False, frozen_layers=frozen_backbone_layers)
    else:
        raise ValueError

    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    layer_channels = backbone_net.out_feature_channels()

    # Extracts features input to the target model
    if clf_feat_extractor == 'resblock':
        target_model_feature_extractor = davos.models.blocks.residual_basic_block(
            feature_dim=layer_channels[target_model_input_layer],
            num_blocks=clf_feat_blocks, num_post_blocks=clf_feat_post_blocks, l2norm=clf_feat_norm,
            final_conv=final_conv, final_proj=clf_proj, dropout_p=clf_feat_dropout_p,
            norm_scale=norm_scale, out_dim=out_feature_dim)

    elif isinstance(clf_feat_extractor, dict) and clf_feat_extractor['type'] == 'resblock_multilayer':
        p = clf_feat_extractor
        target_model_feature_extractor = davos.models.blocks.ResBlockMultilayer(
            feature_dims=[layer_channels[L] for L in p['layers']], layer_names=p['layers'], sizing_layer=p['sizing_layer'],
            num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
            final_conv=final_conv, final_proj=clf_proj,
            norm_scale=norm_scale, out_dim=out_feature_dim)

    else:
        raise ValueError

    # Few-shot label generator and weight predictor

    if label_encoder == 'res_ds16_sw':
        label_encoder = seg_label_encoder.ResidualDS16SW(in_channels=label_channels,
                                                         layer_dims=label_encoder_dims + (num_filters,),
                                                         use_bn=use_bn_in_label_enc)

    elif isinstance(label_encoder, dict) and label_encoder['type'] == 'res_ds8_sw':
        p = label_encoder
        label_encoder = seg_label_encoder.ResidualDS8SW(layer_dims=label_encoder_dims + (num_filters,),
                                                        in_channels=label_channels,
                                                        use_bn=use_bn_in_label_enc,
                                                        out_stride=p.get('out_stride', 8),
                                                        disable_sw=p.get('disable_sw', False))
    else:
        raise ValueError

    # Global context encoder

    # Predicts initial target model parameters
    initializer = seg_initializer.FilterInitializerZero(filter_size=filter_size, num_filters=num_filters,
                                                        feature_dim=out_feature_dim, filter_groups=filter_groups)

    # Computes few-shot learning loss
    residual_module = loss_residual_modules.LWLResidual(
        init_filter_reg=optim_init_reg, filter_dilation_factors=dilation_factors)

    # Iteratively updates the target model parameters by minimizing the few-shot learning loss
    optimizer = steepestdescent.GNSteepestDescent(residual_module=residual_module, num_iter=optim_iter, detach_length=detach_length,
                                                  residual_batch_dim=1, compute_losses=False)

    # Target model and few-shot learner
    target_model = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                           filter_optimizer=optimizer, feature_extractor=target_model_feature_extractor,
                                           filter_dilation_factors=dilation_factors)

    # Decoder
    decoder_input_layers_channels = {L: layer_channels[L] for L in decoder_input_layers}

    assert decoder_type == 'rofl'
    decoder = lwtl_decoder.LWTLDecoder(in_channels=num_filters, out_channels=decoder_mdim,
                                       ft_channels=decoder_input_layers_channels,
                                       label_channels=label_channels,
                                       feat_proj=dec_proj,
                                       use_bn=True,
                                       dropout_p=dec_dropout_p,
                                       tse_variant=decoder_tse_variant)

    net = LWLNet(feature_extractor=backbone_net, target_model=target_model, decoder=decoder, label_encoder=label_encoder,
                 target_model_input_layer=target_model_input_layer, decoder_input_layers=decoder_input_layers)

    if basenet is not None:
        try:
            net.load_state_dict(basenet.state_dict())
            model_loaded = True
        except:
            print('Basenet loading failed! Trying Basenet+upsampler!')
            model_loaded = False

    if upsampler is not None:
        upsampler, guidance = upsampler.split('-')

        if upsampler == 'lts':
            from davos.models.lwl.decoder import LTS
            net.decoder.refine = LTS()
        if upsampler == 'nc':
            from davos.models.lwl.upsampler import ProjectUp
            net.decoder.project = ProjectUp(guidance, out_ch=label_channels)
        elif upsampler == 'raft':
            from davos.models.lwl.upsampler import VOSRaftUpsampler
            net.decoder.project = VOSRaftUpsampler(guidance, out_channels=label_channels)
        elif upsampler == 'praft':
            from davos.models.lwl.upsampler import VOSPRaftUpsampler
            net.decoder.project = VOSPRaftUpsampler(guidance, out_channels=label_channels)

    if freeze_vos:
        for name, param in net.named_parameters():
            if not any(module in name for module in ['decoder.refine', 'decoder.project']):
                param.requires_grad = False

    print('Trainable parameters: ', sum(p.numel() for p in net.parameters() if p.requires_grad))

    return net
