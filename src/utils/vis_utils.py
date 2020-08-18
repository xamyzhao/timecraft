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


def label_ims(ims_batch, labels=None,
              display_h=128, concat_axis=0, combine_from_axis=0):
    '''
    Displays a batch of matrices as an image.

    :param ims_batch: n_batches x h x w x c array of images.
    :param labels: optional labels. Can be an n_batches length list of tuples, floats or strings
    :param normalize: boolean to normalize any [min, max] to [0, 255]
    :param display_h: integer number of pixels for the height of each image to display
    :return: an image (h' x w' x 3) with elements of the batch organized into rows
    '''

    if len(ims_batch.shape) == 3 and ims_batch.shape[-1] == 3:
        # already an image
        return ims_batch

    batch_size = ims_batch.shape[combine_from_axis]

    if type(labels) == list and len(labels) == 1:
        # only label the first image
        labels = labels + [''] * (batch_size - 1)
    elif labels is not None and not type(labels) == list and not type(labels) == np.ndarray:
        # replicate labels for each row in the batch
        labels = [labels] * batch_size



    # make sure we have a channels dimension
    if len(ims_batch.shape) < 4:
        ims_batch = np.expand_dims(ims_batch, 3)

    ims_batch = inverse_normalize(ims_batch)

    if np.max(ims_batch) <= 1.0:
        ims_batch = ims_batch * 255.0

    ims_batch = np.split(ims_batch, ims_batch.shape[combine_from_axis], axis=combine_from_axis)
    ims_batch = [np.take(im, 0, axis=combine_from_axis) for im in ims_batch]  # remove the extra axis
    h, w = ims_batch[0].shape[:2]
    scale_factor = display_h / float(h)

    out_im = []
    for i in range(batch_size):
        # convert grayscale to rgb if needed
        if len(ims_batch[i].shape) == 2:
            curr_im = np.tile(np.expand_dims(ims_batch[i], axis=-1), (1, 1, 3))
        elif ims_batch[i].shape[-1] == 1:
            curr_im = np.tile(ims_batch[i], (1, 1, 3))
        else:
            curr_im = ims_batch[i]

        # scale to specified display size
        if scale_factor > 2:  # if we are upsampling by a lot, nearest neighbor can look really noisy
            interp = cv2.INTER_NEAREST
        else:
            interp = cv2.INTER_LINEAR

        if not scale_factor == 1:
            curr_im = cv2.resize(curr_im, None, fx=scale_factor, fy=scale_factor, interpolation=interp)

        out_im.append(curr_im)

    out_im = np.concatenate(out_im, axis=concat_axis).astype(np.uint8)

    # draw text labels on images if specified
    font_size = 15
    max_text_width = int(17 * display_h / 128.)  # empirically determined

    if labels is not None and len(labels) > 0:
        im_pil = Image.fromarray(out_im)
        draw = ImageDraw.Draw(im_pil)

        for i in range(batch_size):
            if len(labels) > i:  # if we have a label for this image
                if type(labels[i]) == tuple or type(labels[i]) == list:
                    # format tuple or list nicely
                    formatted_text = ', '.join([
                        labels[i][j].decode('UTF-8') if type(labels[i][j]) == np.unicode_ \
                            else labels[i][j] if type(labels[i][j]) == str \
                            else str(round(labels[i][j], 2)) if isinstance(labels[i][j], float) \
                            else str(labels[i][j]) for j in range(len(labels[i]))])
                elif type(labels[i]) == float or type(labels[i]) == np.float32:
                    formatted_text = str(round(labels[i], 2))  # round floats to 2 digits
                elif isinstance(labels[i], np.ndarray):
                    # assume that this is a 1D array
                    curr_labels = np.squeeze(labels[i]).astype(np.float32)
                    formatted_text = np.array2string(curr_labels, precision=2, separator=',')
                else:
                    formatted_text = '{}'.format(labels[i])

                font = ImageFont.truetype('Ubuntu-M.ttf', font_size)
                # wrap the text so it fits
                formatted_text = textwrap.wrap(formatted_text, width=max_text_width)

                for li, line in enumerate(formatted_text):
                    draw.text((5, i * display_h + 5 + 14 * li), line, font=font, fill=(50, 50, 255))

        out_im = np.asarray(im_pil)

    return out_im