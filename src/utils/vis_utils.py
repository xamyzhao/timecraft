import cv2
import math
import numpy as np
import textwrap

from PIL import Image, ImageDraw, ImageFont

def pad_or_crop_to_shape(
        I,
        out_shape,
        border_color=(255, 255, 255)):

    if not isinstance(border_color, tuple):
        n_chans = I.shape[-1]
        border_color = tuple([border_color] * n_chans)

    # an out_shape with a dimension value of None means just don't crop or pad in that dim
    border_size = [out_shape[d] - I.shape[d] if out_shape[d] is not None else 0 for d in range(2)]
    #print('Padding or cropping with border: {}'.format(border_size))
    if not border_size[0] == 0:
        top_border = abs(int(math.floor(border_size[0] / 2.)))
        bottom_border = abs(int(math.ceil(border_size[0] / 2.)))

        if border_size[0] > 0:
            # pad with rows on top and bottom
            I = np.concatenate([
                np.ones((top_border,) + I.shape[1:], dtype=I.dtype) * border_color,
                I,
                np.ones((bottom_border,) + I.shape[1:], dtype=I.dtype) * border_color
            ], axis=0)

        elif border_size[0] < 0:
            # crop from top and bottom
            I = I[top_border:-bottom_border]

    if not border_size[1] == 0:
        left_border = abs(int(math.floor(border_size[1] / 2.)))
        right_border = abs(int(math.ceil(border_size[1] / 2.)))

        if border_size[1] > 0:
            # pad with cols on left and right
            I = np.concatenate([
                np.ones((I.shape[0], left_border) + I.shape[2:], dtype=I.dtype) * border_color,
                I,
                np.ones((I.shape[0], right_border) + I.shape[2:], dtype=I.dtype) * border_color,
            ], axis=1)
        elif border_size[1] < 0:
            # crop left and right sides
            I = I[:, left_border: -right_border]

    return I


def concatenate_with_pad(ims_list, pad_to_im_idx=None, axis=None, pad_val=0.):
    padded_ims_list = pad_images_to_size(ims_list, pad_to_im_idx, ignore_axes=axis, pad_val=pad_val)
    return np.concatenate(padded_ims_list, axis=axis)


def pad_images_to_size(ims_list, pad_to_im_idx=None, ignore_axes=None, pad_val=0.):
    if pad_to_im_idx is not None:
        pad_to_shape = ims_list[pad_to_im_idx].shape
    else:
        im_shapes = np.reshape([im.shape for im in ims_list], (len(ims_list), -1))
        pad_to_shape = np.max(im_shapes, axis=0).tolist()

    if ignore_axes is not None:
        if not isinstance(ignore_axes, list):
            ignore_axes = [ignore_axes]
        for a in ignore_axes:
            pad_to_shape[a] = None

    ims_list = [pad_or_crop_to_shape(im, pad_to_shape, border_color=pad_val) \
        for i, im in enumerate(ims_list)]
    return ims_list


def normalize(X):
    if not X.dtype == np.float32 and not X.dtype == np.float64:
        X = X.astype(np.float32) / 255.
    if X is None:
        return None
    return np.clip(X * 2.0 - 1.0, -1., 1.)


def inverse_normalize(X):
    return np.clip((X + 1.0) * 0.5, 0., 1.)


def visualize_video(frames, border_size=1, max_n_ims_per_row=10, normalized=False):
    h, w, c, T = frames.shape[-4:]

    n_rows = int(np.ceil(T / float(max_n_ims_per_row)))

    output_rows = []
    for ri in range(n_rows):
        row_ims = [
            frames[..., t]
            for t in range(ri * max_n_ims_per_row, min(T, (ri + 1) * max_n_ims_per_row))]

        if normalized:
            row_ims = [inverse_normalize(im) for im in row_ims]

        if border_size:
            row_ims = [
                cv2.copyMakeBorder(im, border_size, border_size, border_size, border_size,
                                   borderType=cv2.BORDER_CONSTANT, value=0.
                                   ) for im in row_ims]
        row_im = np.concatenate(row_ims, axis=1)

        output_rows.append(row_im)
    return np.concatenate(output_rows, axis=0)
