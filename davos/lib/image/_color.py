import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mplc
import io


davis_palette = np.repeat(np.expand_dims(np.arange(0,256), 1), 3, 1).astype(np.uint8)
davis_palette[:22, :] = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                         [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                         [64, 0, 0], [191, 0, 0], [64, 128, 0], [191, 128, 0],
                         [64, 0, 128], [191, 0, 128], [64, 128, 128], [191, 128, 128],
                         [0, 64, 0], [128, 64, 0], [0, 191, 0], [128, 191, 0],
                         [0, 64, 128], [128, 64, 128]]


def indexed_to_rgb(im, color_palette=None):
    """
    :param im:  Image, shape (H,W)
    :param color_palette:
    :return:
    """

    if color_palette is None:
        color_palette = davis_palette

    if torch.is_tensor(im):
        p = torch.from_numpy(color_palette).to(im.device)
        im = p[im.squeeze(0).long()].permute(2, 0, 1)
    else:
        im = Image.fromarray(im.squeeze(), 'P')
        im.putpalette(color_palette.ravel())
        im = np.array(im.convert('RGB'))

    return im


class Colormap(nn.Module):

    def __init__(self, name, values, colors, palette_size):
        super().__init__()

        self.name = name
        self.size = palette_size
        self.low = values[0]
        self.high = values[-1]
        self.scale = self.size / (self.high - self.low)
        self.mpl_registered = False

        x = np.arange(self.low, self.high, step=1/self.scale)
        palette = np.stack((np.interp(x=x, xp=values, fp=colors[:, 0]),
                            np.interp(x=x, xp=values, fp=colors[:, 1]),
                            np.interp(x=x, xp=values, fp=colors[:, 2])), axis=-1)
        palette = torch.from_numpy(palette.astype(np.float32))

        self.register_buffer("palette", palette)

    def forward(self, x):
        x = self.palette[torch.round((x - self.low) * self.scale).clamp(0, self.size-1).long()]
        return x.transpose(-1, -2).transpose(-2, -3)

    def reverse_(self):
        self.palette = self.palette.flip(dims=(0,))

    def mpl_colormap(self):
        return mplc.LinearSegmentedColormap.from_list(self.name, self.palette.cpu().numpy(), self.size)

    def mpl_normalizer(self):
        return mplc.Normalize(self.low, self.high, clip=True)

    def mpl_scalar_mappable(self):
        return plt.cm.ScalarMappable(cmap=self.mpl_colormap(), norm=self.mpl_normalizer())

    def colorbar_image(self, width, height, orientation='vertical', **savefig_kwargs):

        fig = plt.figure(frameon=False)
        DPI = float(fig.get_dpi())
        fig.set_size_inches(width / DPI, height / DPI)
        ax = fig.gca()
        sm = self.mpl_scalar_mappable()
        cb = fig.colorbar(sm, orientation=orientation, cax=ax)
        ticklabs = cb.ax.get_yticklabels()
        cb.ax.set_yticklabels(ticklabs, fontsize=1000)

        io_buf = io.BytesIO()
        fig.savefig(io_buf, format='png')
        io_buf.seek(0)
        im = np.array(Image.open(io_buf)).transpose(2, 0, 1)
        io_buf.close()

        return im

    def extra_repr(self) -> str:
        return self.name

#def get_colorbar(colormap: Colormap):

def get_colormap(name, size, low, high):

    if name == 'fire_ice':
        # Based on https://www.mathworks.com/matlabcentral/fileexchange/24870-fireice-hot-cold-colormap
        # Copyright (c) 2009, Joseph Kirk, License: BSD 2-clause
        # Piecewise linear interpolation settings:
        colors = np.array(((0.75, 1, 1), (0, 1, 1), (0, 0, 1), (0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 0.75)))
        values = np.array((-3, -2, -1, 0, 1, 2, 3), dtype=np.float32) / 6 + 0.5  # scaled to [0., 1.]
        values = values * (high - low) + low  # scaled to [low, high]
    elif name == 'bright_fire_ice':  # Fire-ice with color near zero.
        colors = np.array(((0.75, 1, 1), (0, 1, 1), (0, 0, 1), (0, 0, 0.5),
                           (0.5, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 0.75)))
        values = np.array((-3, -2, -1, -0.003, 0.003, 1, 2, 3), dtype=np.float32) / 8 + 0.5  # scaled to [0., 1.]
        values = values * (high - low) + low  # scaled to [low, high]
    elif name.startswith("mpl:"):
        colors = plt.get_cmap(name[4:], lut=size)
        values = np.arange(0, size) / (size-1)
        if isinstance(colors, mplc.ListedColormap):
            colors = colors.colors[:, :3]
        elif isinstance(colors, mplc.LinearSegmentedColormap):
            colors = colors(values)
        else:
            raise NotImplementedError
        values = values * (high - low) + low
    else:
        raise ValueError("Undefined colormap %s" % name)

    return Colormap(name, values, colors, size)
