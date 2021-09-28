import os
import inspect
import typing
from pathlib import Path
import math
import torch
import numpy as np
from .image import get_colormap, indexed_to_rgb, imwrite, davis_palette, Colormap
from davos.config import env_settings


class DebugVis:
    def __init__(self, raw_segment_range=(-100, 100), start_visdom=True):

        self.visdom = None
        if start_visdom:
            self.start_visdom()

        self.raw_segment_colormap = "fire_ice"
        self.raw_segment_range = raw_segment_range
        self.raw_segment_cmap = get_colormap(self.raw_segment_colormap, 255, *raw_segment_range)

        self.default_cmap = get_colormap("fire_ice", 255, -1, 1)
        self.env = env_settings()
        self.name = os.environ.get("LTR_RUN_NAME", str(os.getpid()))
        self.bbox_palette = torch.from_numpy(davis_palette).view(256, 3, 1, 1)

    def start_visdom(self):
        from visdom import Visdom
        self.visdom = Visdom(raise_exceptions=False)

    def make_grid(self, t, nrow=None, padding=2):
        t = t.view(-1, *t.shape[-3:])
        n, c, h, w = t.shape
        nrows = math.ceil(math.sqrt(n)) if nrow is None else nrow
        ncols = math.ceil(n / nrows)
        dx = w + padding
        dy = h + padding
        x, y = 0, 0

        grid = torch.zeros((c, dy * nrows - padding, dx * ncols - padding), device=t.device, dtype=t.dtype)
        for im in list(t):
            grid[:, y:y + h, x:x + w] = im
            x += dx
            if x >= (dx * ncols - padding):
                x = 0
                y += dy

        return grid

    def show_colorbar(self, colormap, width=200, height=600, dpi=200, orientation='vertical', title="", win="colorbar"):
        """ show_colorbar(cmap, 200, 600, orientation='vertical', dpi=200, bbox_inches='tight', pad_inches=0.2)
        """
        im = colormap.colorbar_image(width, height, orientation, dpi=dpi, bbox_inches='tight', pad_inches=0.2)
        self.imshow(im, title=title, win=win)

    def colorize_rawseg(self, seg, value_range=None, colormap=None):
        if isinstance(value_range, (int, float)):
            value_range = (-abs(value_range), abs(value_range))

        if value_range is None:
            value_range = self.raw_segment_range
        if colormap is None:
            colormap = self.raw_segment_colormap

        # FIXME: This part is quite buggy. Is colormap a string or a Colormap object? When to pass-through the colormap parameter, when to change it?
        if not isinstance(colormap, Colormap):
            if colormap != self.raw_segment_colormap or value_range != self.raw_segment_range:
                cmap = get_colormap(colormap, 255, *value_range).to(seg.device)
            else:
                cmap = self.raw_segment_cmap
        else:
            cmap = colormap

        cmap = cmap.to(seg.device)

        im = cmap(seg.squeeze())
        return im

    def colorize(self, t, cmap):
        cmap = cmap.to(t.device)
        im = cmap(t.squeeze())
        return im, cmap

    def show_rawseg(self, seg, title="raw segment", value_range=None, **kwargs):
        if value_range is None:
            value_range = self.current_value_range
        self.current_value_range = value_range
        if isinstance(seg, (tuple, list)):
            im = [self.colorize_rawseg(x, value_range) for x in seg]
        else:
            im = self.colorize_rawseg(seg, value_range)
        self.imshow(im, title=title, **kwargs)

    def segshow(self, image, labels, alpha=0.5, title="image+labels", **kwargs):
        if isinstance(image, (tuple, list)):
            assert len(image) == len(labels)
            image = [self.overlay_labels(im, lb, alpha) for im, lb in zip(image, labels)]
        else:
            image = self.overlay_labels(image, labels, alpha)
        self.imshow(image, title=title, **kwargs)

    def overlay_labels(self, image, labels, alpha):
        return indexed_to_rgb(labels.squeeze()) * alpha + image * (1 - alpha * (labels != 0))

    @staticmethod
    def fill_rect_(im, x1, x2, y1, y2, color):

        H, W = im.shape[-2:]

        y1, y2 = min(y1, y2), max(y1, y2)
        if y2 < 0 or y1 >= H:
            return

        x1, x2 = min(x1, x2), max(x1, x2)
        if x2 < 0 or x1 >= W:
            return

        y1, y2 = max(0, y1), min(H - 1, y2)
        x1, x2 = max(0, x1), min(W - 1, x2)

        im[:, y1:y2 + 1, x1:x2 + 1] = color

    @classmethod
    def draw_bbox_(cls, im, bbox, color, thickness=1):

        x1, y1, w, h = list(map(lambda x: int(x + 0.5), bbox))
        x2 = x1 + w
        y2 = y1 + h
        t = thickness // 2

        cls.fill_rect_(im, x1 - t, x2 + t, y1 - t, y1 + t, color)  # top
        cls.fill_rect_(im, x1 - t, x2 + t, y2 - t, y2 + t, color)  # bottom
        cls.fill_rect_(im, x1 - t, x1 + t, y1 - t, y2 + t, color)  # left
        cls.fill_rect_(im, x2 - t, x2 + t, y1 - t, y2 + t, color)  # right

        return im

    def draw_bboxes_(self, image, bboxes):
        for obj_id, bbox in bboxes.items():
            self.draw_bbox_(image, bbox, self.bbox_palette[obj_id], thickness=3)

        return im

    def image_grid(self, tensor, nrow=8, padding=2):
        """
        Given a 4D tensor of shape (B x C x H x W),
        or a list of images all of the same size,
        makes a grid of images of size (B / nrow, nrow).

        This is a modified from `make_grid()`
        https://github.com/pytorch/vision/blob/master/torchvision/utils.py
        """

        # If list of images, convert to a 4D tensor
        if isinstance(tensor, list):
            tensor = np.stack(tensor, 0)

        if tensor.ndim == 2:  # single image H x W
            tensor = np.expand_dims(tensor, 0)
        if tensor.ndim == 3:  # single image
            if tensor.shape[0] == 1:  # if single-channel, convert to 3-channel
                tensor = np.repeat(tensor, 3, 0)
            return tensor

        if tensor.ndim == 4 and tensor.shape[1] == 1:  # single-channel images
            tensor = np.repeat(tensor, 3, 1)

        # make 4D tensor of images into a grid
        nmaps = tensor.shape[0]
        xmaps = min(nrow, nmaps)
        ymaps = int(math.ceil(float(nmaps) / xmaps))
        height = int(tensor.shape[2] + 2 * padding)
        width = int(tensor.shape[3] + 2 * padding)

        grid = np.ones([3, height * ymaps, width * xmaps])
        k = 0
        for y in range(ymaps):
            for x in range(xmaps):
                if k >= nmaps:
                    break
                h_start = y * height + 1 + padding
                h_end = h_start + tensor.shape[2]
                w_start = x * width + 1 + padding
                w_end = w_start + tensor.shape[3]
                grid[:, h_start:h_end, w_start:w_end] = tensor[k]
                k += 1
        return grid

    def imshow(self, im, title="image", win=None, nrow=None):
        if self.visdom is None:
            return
        win = title if win is None else win
        if isinstance(im, (tuple, list)):
            if nrow is None:
                nrow = math.ceil(math.sqrt(len(im)))
            self.visdom.images(torch.stack(im), nrow, win=win, opts=dict(title=title))
        else:
            self.visdom.image(im.squeeze(), win=win, opts=dict(title=title))

    def show_enclb(self, lbenc, **kwargs):

        a = max(abs(lbenc.min().item()), abs(lbenc.max().item()))
        cmap = get_colormap("fire_ice", 255, -a, a).to(lbenc.device)
        self.imshow(list(cmap(lbenc).squeeze()), **kwargs)

    def imwrite(self, im, filename):
        """ Save image to ramdisk
        :param im:  Image (tensor) to save.
        :param filename:  The name of the file. Not a path.
        :return:
        """
        path = self.env.ramdisk_path / self.name
        path.mkdir(exist_ok=True, parents=True)
        imwrite(path / filename, im)

    def lbshow(self, lb, title="labels", **kwargs):
        if isinstance(lb, (tuple, list)):
            lb = [indexed_to_rgb(x.squeeze()) for x in lb]
        else:
            lb = indexed_to_rgb(lb.squeeze())
        if lb.shape[-1] == 3:
            lb = lb.transpose(2, 0, 1)
        self.imshow(lb, title=title, **kwargs)

    def lineplot(self, Y, X=None, title="image", win=None):
        if self.visdom is None:
            return
        if X is None:
            X = torch.arange(1, Y.shape[0] + 1)
        win = title if win is None else win
        self.visdom.line(Y, X, win=win, opts=dict(title=title))
