import time

import numpy as np

import scipy.ndimage as spnd

from src.utils import utils

def crop_center(frames, frame_ids, target_shape, vid_name=None, _print=print):
    top = int(np.floor((frames.shape[1] - target_shape[0]) / 2.))
    bottom = top + target_shape[0]

    left = int(np.floor((frames.shape[2] - target_shape[1]) / 2.))
    right = left + target_shape[1]
    _print('Cropping vid {} frames about center from {} to {}'.format(
        vid_name,
        (left, top), (right, bottom)))

    frames = frames[:, top:bottom, left:right].astype(np.uint8)
    _print('Cropped frames to {}'.format(frames.shape))
    return frames, frame_ids


def crop_rand(frames, target_shape, n_crops,
              frame_ids=None, vid_name=None, _print=print, verbose=True, im_shapes=None,
              border_color=(0, 0, 0), do_scale_before_crop=False):
    cropped_frames = []

    if (frames.shape[1] < target_shape[0] or frames.shape[2] < target_shape[1]) and do_scale_before_crop:
        # print('Scaling up frames {} before cropping to {}'.format(frames.shape, target_shape))
        # scale up to a size so that we dont need to pad when we crop (no borders, yay!)
        scale_x = target_shape[0] / float(frames.shape[1])
        scale_y = target_shape[1] / float(frames.shape[2])
        frames = utils.resize_batch(frames, max(scale_x, scale_y))

    nr, nc, n_chans = frames[0].shape

    if verbose:
        _print('Cropping vid {} frames ({}) randomly x {}'.format(
            vid_name, frames.shape, n_crops
        ))

    if im_shapes is not None:  # only crop up to where the image goes, the rest might be border
        nr = np.min(im_shapes[:, 0])
        nc = np.min(im_shapes[:, 1])
    top_min = 0
    left_min = 0
    top_max = (nr - target_shape[0])
    left_max = (nc - target_shape[1])
    cropped_frame_ids = []
    for ci in range(n_crops):

        top = int(round(np.random.rand(1)[0] * (top_max - top_min) + top_min))
        left = int(round(np.random.rand(1)[0] * (left_max - left_min) + left_min))

        bottom = top + target_shape[0]
        right = left + target_shape[1]
        if frames.shape[1] > target_shape[0] and frames.shape[2] > target_shape[1]:
            cropped = frames[:, top:bottom, left:right]
        elif frames.shape[1] > target_shape[0]:
            cropped = frames[:, top:bottom, :]
        elif frames.shape[2] > target_shape[1]:
            cropped = frames[:, :, left:right]
        else:
            cropped = frames  # frames are smaller, need to pad

        if not np.all(cropped.shape[1:3] == np.asarray(target_shape[:2])):
            cropped = np.concatenate([
                utils.pad_or_crop_to_shape(cropped_frame,
                                           target_shape, border_color=border_color)[np.newaxis]
                for cropped_frame in cropped], axis=0).astype(np.uint8)

        cropped_frames.append(cropped)
        cropped_frame_ids.append(frame_ids)
    if verbose:
        _print('Cropped frames to {}'.format([
            cf.shape for cf in cropped_frames[:min(len(cropped_frames), 5)]]))
    return cropped_frames, cropped_frame_ids


def crop_overlapping(frames, target_shape, frame_ids=None, vid_name=None, _print=print):
    cropped_frames = []

    nr, nc, n_chans = frames[0].shape
    n_rows = int(np.ceil(nr / float(target_shape[0])))
    n_cols = int(np.ceil(nc / float(target_shape[1])))

    top_starts = np.linspace(0, nr - target_shape[0], num=n_rows, dtype=int)
    left_starts = np.linspace(0, nc - target_shape[1], num=n_cols, dtype=int)

    _print('Cropping vid {} frames ({}) with {} overlapping {} blocks: {}, {}'.format(
        vid_name, frames.shape, len(list(top_starts)) * len(list(left_starts)), target_shape,
        top_starts, left_starts
    ))

    cropped_frame_ids = []
    for top in top_starts:
        for left in left_starts:
            bottom = top + target_shape[0]
            right = left + target_shape[1]
            cropped = frames[:, top:bottom, left:right]

            if not np.all(cropped.shape[1:3] == np.asarray(target_shape[:2])):
                cropped = np.concatenate([
                    utils.pad_or_crop_to_shape(cropped_frame, target_shape)[np.newaxis]
                    for cropped_frame in cropped], axis=0).astype(np.uint8)

            cropped_frames.append(cropped)
            cropped_frame_ids.append(frame_ids)

    _print('Cropped frames to {}'.format([cf.shape for cf in cropped_frames[:min(len(cropped_frames), 5)]]))
    return cropped_frames, cropped_frame_ids


def _test_crop_center():
    test_frames = np.ones((2, 10, 10, 1))

    test_frames[:, 2:8, 2:8] = 0

    assert np.sum(test_frames) == 2 * (100 - 36)

    cropped_frames = crop_center(test_frames, target_shape=(6, 6))
    assert np.all(cropped_frames == 0)

    cropped_frames = crop_center(test_frames, (8, 8))
    assert np.sum(cropped_frames) == 2 * (64 - 36)
    print('UNIT TEST: cropping in center produces the correct sizes and sums -- PASSED')


def _test_crop_overlapping():
    batch_size = 2
    test_frames = np.zeros((batch_size, 10, 10, 1))

    test_frames[:, 2:8, 2:8] = 1

    assert np.sum(test_frames) == batch_size * 6 ** 2

    cropped_frames = crop_overlapping(test_frames, (6, 6))
    assert cropped_frames.shape[0] == batch_size * 4
    assert np.all([np.sum(cf) == 4 ** 2 for cf in cropped_frames])

    cropped_frames = crop_overlapping(test_frames, (5, 5))
    assert cropped_frames.shape[0] == batch_size * 4
    assert np.all([np.sum(cf) == 3 ** 2 for cf in cropped_frames])
    print('UNIT TEST: overlapping crops produce the correct sizes and sums -- PASSED')


if __name__ == '__main__':
    _test_crop_center()
    _test_crop_overlapping()
