import numpy as np
from PIL import Image
import torch
from ._color import davis_palette

def imread(f):
    im = np.array(Image.open(f))
    im = np.atleast_3d(im).transpose(2, 0, 1)
    im = torch.from_numpy(im)
    return im

def imwrite(f, im, mode=None, color_palette=None):

    im = im.detach().cpu().numpy() if torch.is_tensor(im) else im
    im = im.reshape(-1, *im.shape[-2:]).transpose(1, 2, 0)

    if mode is not None:
        pass
    elif im.shape[-1] == 3:
        mode = 'RGB'
        im = im.astype(np.uint8)
    elif im.shape[-1] == 1:
        if im.dtype == np.float32:
            mode = 'L'
        elif im.dtype == np.uint8:
            mode = 'P'

    if mode is None:
        raise ValueError

    im = Image.fromarray(im.squeeze(), mode)

    if mode == 'P':
        if color_palette is None:
            color_palette = davis_palette
        im.putpalette(color_palette.ravel())

    im.save(f)
    return im
