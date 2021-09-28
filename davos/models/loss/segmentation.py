import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import davos.models.loss.lovasz_loss as lovasz_loss


class LovaszSegLoss(nn.Module):
    def __init__(self, classes=(1,), per_image=True, with_sigmoid=True, balancing=None):
        """
        :param classes:  'all' for all, 'present' for classes present in labels, or a list of classes to average.
        :type classes: Union[str, List[int]]
        :param per_image: Compute the loss per image instead of per batch
        :type per_image: bool
        :param balancing: Method to balance target and distractor pixels so that large distractor surface areas (pixels) are discounted.
        :type balancing: str
        """
        super().__init__()

        self.classes = classes
        self.per_image = per_image
        self.with_sigmoid = with_sigmoid
        self.balancing = balancing

        if balancing is None:
            self._forward = self._unbalanced_fw
        elif balancing == 'n1/n2':
            self._forward = self._n1n2_balanced_fw
        elif balancing == 'n2/n1':
            self._forward = self._n2n1_balanced_fw
        else:
            raise NotImplementedError(f"balancing = {balancing}")

    def _n2n1_balanced_fw(self, pred, gt):

        pred = torch.sigmoid(pred) if self.with_sigmoid else pred

        pred1 = pred[:, 0:1].contiguous()  # Target objects
        gt1 = gt[:, 0:1].contiguous()  #
        n1 = gt1.sum()

        pred2 = pred[:, 1:2].contiguous()  # Target distractors
        gt2 = gt[:, 1:2].contiguous()
        n2 = gt2.sum()

        L1 = lovasz_loss.lovasz_softmax(probas=pred1, labels=gt1, per_image=self.per_image, classes=self.classes)
        L2 = lovasz_loss.lovasz_softmax(probas=pred2, labels=gt2, per_image=self.per_image, classes=self.classes)
        w = (n2 / n1).clamp(0.0, 1.0)
        L = L1 + w * L2

        return L

    def _n1n2_balanced_fw(self, pred, gt):

        pred = torch.sigmoid(pred) if self.with_sigmoid else pred

        pred1 = pred[:, 0:1].contiguous()  # Target objects
        gt1 = gt[:, 0:1].contiguous()  #
        n1 = gt1.sum()

        pred2 = pred[:, 1:2].contiguous()  # Target distractors
        gt2 = gt[:, 1:2].contiguous()
        n2 = gt2.sum()

        L1 = lovasz_loss.lovasz_softmax(probas=pred1, labels=gt1, per_image=self.per_image, classes=self.classes)
        L2 = lovasz_loss.lovasz_softmax(probas=pred2, labels=gt2, per_image=self.per_image, classes=self.classes)
        w = (n1 / n2).clamp(0.0, 1.0)
        #print(f"L1={L1.item():.2f}, n1={n1.item()}, L2={L2.item():.2f}, n2={n2.item()}, n1/n2={n1/n2:.2f}, w={w:.2f}")

        L = L1 + w * L2
        return L

    def draw_balancing_functions(self):
        import matplotlib.pyplot as plt
        import numpy as np
        n2 = np.arange(0, 10, step=0.01)
        n1 = np.ones_like(n2)
        plt.plot(n2, (n2 / n1).clip(0.0, 1.0))
        plt.plot(n2, (n1 / n2).clip(0.0, 1.0))
        plt.legend(["min(n2/n2,1.0)", "(n1/n2).clip(0,1)"])
        plt.title("w(n1, n2) s.t. L = L1 + w(n1, n2) * L2,\nL1 = target loss, L2 = distractor loss")
        plt.xlabel("n2/n1 ratio")
        plt.show()

    def _unbalanced_fw(self, input, target):
        input = torch.sigmoid(input) if self.with_sigmoid else input
        return lovasz_loss.lovasz_softmax(probas=input, labels=target, per_image=self.per_image, classes=self.classes)

    def forward(self, pred, gt):
        return self._forward(pred, gt)


def one_hot(labels: torch.Tensor,
            num_classes: int,
            device=None,
            dtype=None,
            eps=1e-6) -> torch.Tensor:
    r"""Converts an integer label x-D tensor to a one-hot (x+1)-D tensor.
    Args:
        labels (torch.Tensor) : tensor with labels of shape :math:`(N, *)`,
                                where N is batch size. Each value is an integer
                                representing correct classification.
        num_classes (int): number of classes in labels.
        device (Optional[torch.device]): the desired device of returned tensor.
         Default: if None, uses the current device for the default tensor type
         (see torch.set_default_tensor_type()). device will be the CPU for CPU
         tensor types and the current CUDA device for CUDA tensor types.
        dtype (Optional[torch.dtype]): the desired data type of returned
         tensor. Default: if None, infers data type from values.
    Returns:
        torch.Tensor: the labels in one hot tensor of shape :math:`(N, C, *)`,
    Examples::
        #>>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
        #>>> kornia.losses.one_hot(labels, num_classes=3)
        tensor([[[[1., 0.],
                  [0., 1.]],
                 [[0., 1.],
                  [0., 0.]],
                 [[0., 0.],
                  [1., 0.]]]]
    """
    if not torch.is_tensor(labels):
        raise TypeError("Input labels type is not a torch.Tensor. Got {}"
                        .format(type(labels)))
    if not labels.dtype == torch.int64:
        raise ValueError(
            "labels must be of the same dtype torch.int64. Got: {}".format(
                labels.dtype))
    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one."
                         " Got: {}".format(num_classes))
    shape = labels.shape
    one_hot = torch.zeros((shape[0], num_classes, shape[1], shape[2])).to(device)
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps
