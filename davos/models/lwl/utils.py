import torch
import torch.nn.functional as F


def adaptive_cat(seq, dim=0, ref_tensor=0, mode='bilinear'):
    sz = seq[ref_tensor].shape[-2:]
    t = torch.cat([interpolate(t, sz, mode=mode) for t in seq], dim=dim)
    return t


def interpolate(t, sz, mode='bilinear'):
    sz = sz.tolist() if torch.is_tensor(sz) else sz
    align = {} if mode == 'nearest' else dict(align_corners=False)
    return F.interpolate(t, sz, mode=mode, **align) if t.shape[-2:] != sz else t


class ModuleWrapper:
    """ A wrapper for hiding modules from PyTorch, so that the same module can be used in multiple places.
    and yet saved only once in per checkpoint, or not at all. """

    # https://stackoverflow.com/questions/1466676/create-a-wrapper-class-to-call-a-pre-and-post-function-around-existing-functions

    def __init__(self, wrapped_module):
        self.__wrapped_module__ = wrapped_module

    def __getattr__(self, attr):
        orig_attr = self.__wrapped_module__.__getattribute__(attr)
        if callable(orig_attr):
            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                # prevent wrapped_class from becoming unwrapped
                if result == self.__wrapped_module__:
                    return self
                return result

            return hooked
        else:
            return orig_attr

    def __call__(self, *args, **kwargs):
        return self.__wrapped_module__(*args, **kwargs)

