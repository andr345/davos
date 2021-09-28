import torch
import torch.nn as nn

from davos.lib.measures import davis_jaccard_measure
from davos.lib import TensorDict


class BaseActor:
    """ Base class for actor. The actor class handles the passing of the data through the network
    and calculation the loss"""
    def __init__(self, net, objective):
        """
        args:
            net - The network to train
            objective - The loss function
        """
        self.net = net
        self.objective = objective

    def __call__(self, data: TensorDict):
        """ Called in each training iteration. Should pass in input data through the network, calculate the loss, and
        return the training stats for the input data
        args:
            data - A TensorDict containing all the necessary data blocks.

        returns:
            loss    - loss for the input data
            stats   - a dict containing detailed losses
        """
        raise NotImplementedError

    def to(self, device):
        """ Move the network to device
        args:
            device - device to use. 'cpu' or 'cuda'
        """
        self.net.to(device)

    def train(self, mode=True):
        """ Set whether the network is in train mode.
        args:
            mode (True) - Bool specifying whether in training mode.
        """
        self.net.train(mode)

    def eval(self):
        """ Set network to eval mode"""
        self.train(False)


class LWLActor(BaseActor):
    """Actor for training the LWL network."""
    def __init__(self, net, objective, loss_weight=None,
                 num_refinement_iter=3,
                 disable_backbone_bn=False,
                 disable_all_bn=False,
                 disable_detach=False):
        """
        args:
            net - The network model to train
            objective - Loss functions
            loss_weight - Weights for each training loss
            num_refinement_iter - Number of update iterations N^{train}_{update} used to update the target model in
                                  each frame
            disable_backbone_bn - If True, all batch norm layers in the backbone feature extractor are disabled, i.e.
                                  set to eval mode.
            disable_all_bn - If True, all the batch norm layers in network are disabled, i.e. set to eval mode.
        """
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'segm': 1.0}
        self.loss_weight = loss_weight

        self.num_refinement_iter = num_refinement_iter
        self.disable_backbone_bn = disable_backbone_bn
        self.disable_all_bn = disable_all_bn
        self.disable_detach = disable_detach

    def train(self, mode=True):
        """ Set whether the network is in train mode.
        args:
            mode (True) - Bool specifying whether in training mode.
        """
        self.net.train(mode)

        if self.disable_all_bn:
            self.net.eval()
        elif self.disable_backbone_bn:
            for m in self.net.feature_extractor.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_masks',
                    'test_masks'
        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        if self.disable_detach:
            self.net.disable_detach = True

        segm_pred = self.net(train_imgs=data['train_images'],
                             test_imgs=data['test_images'],
                             train_masks=data['train_masks'],
                             test_masks=data['test_masks'],
                             num_refinement_iter=self.num_refinement_iter)

        acc = 0
        cnt = 0

        segm_pred = segm_pred.view(-1, 1, *segm_pred.shape[-2:])
        gt_segm = data['test_masks']
        gt_segm = gt_segm.view(-1, 1, *gt_segm.shape[-2:])

        loss_segm = self.loss_weight['segm'] * self.objective['segm'](segm_pred, gt_segm)

        acc_l = [davis_jaccard_measure(torch.sigmoid(rm.detach()).cpu().numpy() > 0.5, lb.cpu().numpy()) for
                 rm, lb in zip(segm_pred.view(-1, *segm_pred.shape[-2:]), gt_segm.view(-1, *segm_pred.shape[-2:]))]
        acc += sum(acc_l)
        cnt += len(acc_l)

        loss = loss_segm

        if torch.isinf(loss) or torch.isnan(loss):
            raise Exception('ERROR: Loss was nan or inf!!!')

        # Log stats
        stats = {'Loss/total': loss.item(),
                 'Loss/segm': loss_segm.item(),
                 'Stats/acc': acc / cnt}

        return loss, stats


class DLWLActor(BaseActor):

    def __init__(self, net, objective, loss_weight=None,
                 num_refinement_iter=3,
                 disable_backbone_bn=False,
                 disable_all_bn=False,
                 hard_dloss=False):

        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'segm': 1.0}
        self.loss_weight = loss_weight

        self.num_refinement_iter = num_refinement_iter
        self.disable_backbone_bn = disable_backbone_bn
        self.disable_all_bn = disable_all_bn
        self.hard_dloss = hard_dloss

        # start_visdom = 'SLURM_JOB_ID' not in os.environ

        self.batch_count = 0
        # self.vis_batch_interval = 1
        # self.vis = DebugVis(start_visdom=start_visdom)
        # self.imshow = self.vis.imshow if start_visdom else self.vis.imwrite

    # def _value_range(self, t):
    #     return max(abs(t.min().item()), abs(t.max().item()))
    #
    # def visualize(self, mask_gt, mask_pred_enc, mask_pred, **kwargs):
    #     if not hasattr(self, 'vis'):
    #         return
    #
    #     self.batch_count += 1
    #     if self.batch_count < self.vis_batch_interval:
    #         return
    #     self.batch_count = 0
    #
    #     b = 0
    #     m_gt =  mask_gt[0, b]
    #     m_enc = mask_pred_enc[0, b]
    #     m_prd = mask_pred[0, b]
    #
    #     cmap = self.vis.default_cmap
    #     imshow = self.vis.imshow
    #
    #     # gt test mask
    #     z = torch.zeros_like(m_gt[0])
    #     d = m_gt[1] if m_gt.shape[0] == 2 else z
    #     m_gt = torch.stack((d, m_gt[0], z)) * 255
    #     imshow(m_gt, "gt.png")
    #
    #     # encoded predicted label (scaled to full color range)
    #     m_enc = self.vis.make_grid(m_enc.unsqueeze(1), padding=0)
    #     im = cmap(m_enc / self._value_range(m_enc)) * 255
    #     imshow(im, "gt_enc.png")
    #
    #     # predicted target and distractor - raw labels
    #     a = self._value_range(m_prd)
    #     imshow(cmap(m_prd[0] / a) * 255, "pred_target.png")
    #     if m_prd.shape[0] == 2:
    #         imshow(cmap(m_prd[1] / a) * 255, "pred_distr.png")
    #
    #     # predicted target labels
    #     im = torch.sigmoid(m_prd)
    #     z = torch.zeros_like(im[0])
    #     d = im[1] if im.shape[0] == 2 else z
    #
    #     im = torch.stack((d, im[0], z)) * 255
    #     imshow(im, "pred_prob.png")

    def train(self, mode=True):

        self.net.train(mode)

        if self.disable_all_bn:
            self.net.eval()
        elif self.disable_backbone_bn:
            for m in self.net.feature_extractor.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def encode_segments(self, data):

        train_masks = data['train_masks']
        test_masks = data['test_masks']
        train_dm = data.get('train_distractor_masks', None)
        test_dm = data.get('test_distractor_masks', None)

        have_distractors = train_dm is not None
        nc = getattr(self.net.decoder, 'label_channels', 1)

        if have_distractors:
            assert nc == 2
            train_masks = torch.cat([train_masks, train_dm], dim=-3)
            test_masks = torch.cat([test_masks, test_dm], dim=-3)

        if train_masks.shape[-3] < nc:
            (f, b, c, h, w), d = train_masks.shape, train_masks.device
            train_masks = torch.cat([train_masks, torch.zeros((f, b, nc-c, h, w), device=d)], dim=-3)
            (f, b, c, h, w), d = test_masks.shape, test_masks.device
            test_masks = torch.cat([test_masks, torch.zeros((f, b, nc-c, h, w), device=d)], dim=-3)

        return train_masks, test_masks, have_distractors, nc

    def __call__(self, data):

        train_masks, test_masks, have_distractors, nc = self.encode_segments(data)

        segm_pred, debug_info = self.net(train_imgs=data['train_images'],
                                         test_imgs=data['test_images'],
                                         train_masks=train_masks,
                                         test_masks=test_masks,
                                         num_refinement_iter=self.num_refinement_iter, return_debug=True)

        return test_masks, segm_pred, debug_info, have_distractors, nc

    def loss(self, data, results):
        test_masks, segm_pred, debug_info, have_distractors, nc = results
        segm_gt = test_masks

        #self.visualize(mask_gt=test_masks, **debug_info)

        if have_distractors and not self.hard_dloss:

            # Soft distractor loss: Disable the distractor loss outside the target area if no gt distractor exists.
            # At the same time, require the distractor activations to be zero inside the target area.

            tc, dc = slice(0, 1), slice(1, 2)  # Target channel, distractor channel (first number inside slice tuple)
            F, B = segm_gt.shape[:2]  # number of test frames, batch size
            d = data['have_distractors'].repeat(F, 1).float().view(F, B, 1, 1, 1)
            enable = segm_gt[:, :, tc] * (1 - d) + d
            segm_pred[:, :, dc] = segm_pred[:, :, dc] * enable  # shape = [F=3,B,2,H,W]
            segm_gt[:, :, dc] = segm_gt[:, :, dc] * d

        loss = 0
        if 'segm' in self.objective:
            objective = self.objective['segm']

            if objective.balancing is None:
                pr = segm_pred.view(-1, 1, *segm_pred.shape[-2:])  # 3*B*2 x 1 X H x W
                gt = segm_gt.view(-1, 1, *segm_gt.shape[-2:])      # 3*B*2 x 1 X H x W
            else:
                pr = segm_pred.view(-1, 2, *segm_pred.shape[-2:])  # 3*B x 2 X H x W
                gt = segm_gt.view(-1, 2, *segm_gt.shape[-2:])      # 3*B x 2 X H x W

            loss_segm = self.loss_weight['segm'] * objective(pr, gt)
            loss += loss_segm

        # Remove distractors, if any

        segm_pred = segm_pred.view(-1, *segm_pred.shape[-3:])
        segm_gt = segm_gt.view(-1, *segm_gt.shape[-3:])
        segm_pred = segm_pred[:, :1, :, :]
        segm_gt = segm_gt[:, :1, :, :]

        acc = 0
        n = segm_gt.shape[0]
        for pr, gt in zip(torch.sigmoid(segm_pred.detach()) > 0.5, segm_gt > 0.5):
            acc += davis_jaccard_measure(pr[0], gt[0])  # targets but not distractors

        if not torch.isfinite(loss):
            raise RuntimeError('ERROR: NaN or inf loss detected')

        stats = {'Stats/acc': (acc / n).detach(),
                 'Loss/total': loss.detach()}

        if 'segm' in self.objective:
            stats['Loss/segm'] = loss_segm.detach()

        return loss, stats
