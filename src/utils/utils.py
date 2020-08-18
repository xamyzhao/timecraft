import os

import cv2
import numpy as np

from keras import losses as keras_losses

from src import metrics


def filenames_to_im_ids(im_files):
    im_files = [os.path.basename(f) for f in im_files]
    if isinstance(im_files[0], int):
        return im_files
    elif 'frame_' in im_files[0]:
        im_file_ids = [int(re.search('(?<=frame_)[0-9]*', f).group(0)) for f in im_files]
        return im_file_ids
    elif 'frame' in im_files[0]:
        im_file_ids = [int(re.search('(?<=frame)[0-9]*', os.path.basename(f)).group(0)) for f in im_files]
        return im_file_ids
    else:
        raise Exception(f'Could not parse im file names {im_files[:3]}')
    return im_file_ids


import skimage


def compute_attention_maps(
        frames=None, deltas=None,
        attn_thresh=0.1,
        attn_erode_sigma=0, attn_dilate_sigma=0,
        convert_lab=False, verbose=True,
        do_pad_start=False
):
    '''
    assume X is n_frames x h x w x 3
    '''
    if frames is not None and (np.max(frames) > 1. or frames.dtype == np.uint8):
        frames = frames.astype(np.float32) / 255.

    if deltas is not None and deltas.dtype == np.uint8:
        deltas = deltas.astype(np.float32) / 255.

    if deltas is None:
        if convert_lab:
            # X = np.concatenate([skimage.color.rgb2hsv(x[...,[2, 1, 0]])[np.newaxis] for x in X], axis=0)
            frames = skimage.color.rgb2lab(frames[..., [2, 1, 0]])
            # X = X[..., [0, 1]] # ignore deltas in value channel
            if verbose:
                print('X min {} max {}'.format(np.min(frames, axis=(0, 1, 2)), np.max(frames, axis=(0, 1, 2))))
            # lab ranges from 0-100, -127 to 127, -127 to 127
            attn_thresh = [35, 10, 10]
        else:
            # rgb to grayscale
            frames = np.mean(frames, axis=-1)[..., np.newaxis]
        if verbose:
            print('Computing deltas on frames of shape {}'.format(frames.shape))
        # compute absolute delta. take absolute difference and average across color channels
        deltas = np.diff(frames, axis=0)

    frame_shape = deltas.shape
    deltas = np.abs(deltas)

    deltas = np.reshape(deltas, (np.prod(deltas.shape[:-1]), -1))
    # print('Postprocessing deltas with thresh {}, erode {}, dilate {}, blur {}'.format(
    #    attn_thresh, attn_erode_sigma, attn_dilate_sigma, attn_blur_sigma))
    attn_thresh = np.asarray(attn_thresh)
    if attn_thresh.size == 1:
        attn_thresh = np.tile(np.reshape(attn_thresh, (1, 1)), (deltas.shape[0], deltas.shape[-1]))
    else:
        attn_thresh = np.tile(np.reshape(attn_thresh, (1, frame_shape[-1])), (deltas.shape[0], 1))

    bin_deltas_flat = np.zeros((deltas.shape[0], 1))  # binary map
    bin_deltas_flat[np.any(deltas > attn_thresh, axis=-1)] = 1.
    deltas = np.reshape(bin_deltas_flat, frame_shape[:-1] + (1,))
    # binarize
    # deltas[deltas > attn_thresh] = 1.
    # deltas[deltas <= attn_thresh] = 0.

    # roll frames into batches so we can blur
    batch_size = 100

    for bi in np.arange(0, deltas.shape[0], step=batch_size):
        if attn_erode_sigma > 0:
            deltas[bi:min(bi + batch_size, deltas.shape[0])] \
                = erode_batch(deltas[bi:min(bi + batch_size, deltas.shape[0])], attn_erode_sigma)
        if attn_dilate_sigma > 0:
            deltas[bi:min(bi + batch_size, deltas.shape[0])] \
                = dilate_batch(deltas[bi:min(bi + batch_size, deltas.shape[0])], attn_dilate_sigma)

    # normalize each frame in each video to 0, 1. Forget about normalizing by sum
    max_deltas = np.max(deltas, axis=(1, 2))
    for fi in range(deltas.shape[0]):
        if max_deltas[fi] > 1e-5:
            deltas[fi] /= max_deltas[fi]
    if do_pad_start:
        # add a 0 attention map to the beginning so that
        # the output volume is the same size as the input frames
        deltas = np.concatenate([
            np.zeros(deltas[[0]].shape),
            deltas
        ], axis=0)

    return deltas


import re
def make_output_dirs(experiment_base_name: str,
                     prompt_delete_existing: bool = True,
                     prompt_update_name: bool = True,
                     exp_root: str = 'C:/experiments/',
                     existing_exp_dir=None,
                     # for unit testing
                     debug_delete_input=None,
                     debug_rename_input=None
                     ):
    '''
    Creates the experiment directory (for storing log files, parameters) as well as subdirectories for
    files, logs and models.

    If a directory already exists for this experiment,

    :param experiment_base_name: desired name for the experiment
    :param prompt_delete_existing: if we find an existing directory with the same name,
        do we tell the user? if not, just continue in the existing directory by default
    :param prompt_update_name: if the new experiment name differs from the existing_exp_dir,
        do we try to rename the existing directory to the new naming scheme?
    :param exp_root: root directory for all experiments
    :param existing_exp_dir: previous directory (if any) of this experiment
    :return:
    '''

    do_rename = False

    if existing_exp_dir is None:
        # brand new experiment
        experiment_name = experiment_base_name
        target_exp_dir = os.path.join(exp_root, experiment_base_name)
    else:  # we are loading from an existing directory
        if re.search('_[0-9]*$', existing_exp_dir) is not None:
            # we are probably trying to load from a directory like experiments/<exp_name>_1,
            #  so we should track the experiment_name with the correct id
            experiment_name = os.path.basename(existing_exp_dir)
            target_exp_dir = os.path.join(exp_root, experiment_name)
        else:
            # we are trying to load from a directory, but the newly provided experiment name doesn't match.
            # this can happen when the naming scheme has changed
            target_exp_dir = os.path.join(exp_root, experiment_base_name)

            # if it has changed, we should prompt to rename the old experiment to the new one
            if prompt_update_name and not os.path.abspath(existing_exp_dir) == os.path.abspath(target_exp_dir):
                target_exp_dir, do_rename = _prompt_rename(
                    existing_exp_dir, target_exp_dir, debug_rename_input)

                if do_rename: # we might have changed the model name to something that exists, so prompt if so
                    print('Renaming {} to {}!'.format(existing_exp_dir, target_exp_dir))
                    prompt_delete_existing = True
            else:
                target_exp_dir = existing_exp_dir # just assume we want to continue in the old one

            experiment_name = os.path.basename(target_exp_dir)

    print('Existing exp dir: {}'.format(existing_exp_dir))
    print('Target exp dir: {}'.format(target_exp_dir))

    figures_dir = os.path.join(target_exp_dir, 'figures')
    logs_dir = os.path.join(target_exp_dir, 'logs')
    models_dir = os.path.join(target_exp_dir, 'models')

    copy_count = 0

    # check all existing dirs with the same prefix (and all suffixes e.g. _1, _2)
    while os.path.isdir(figures_dir) or os.path.isdir(logs_dir) or os.path.isdir(models_dir):
        # list existing files
        if os.path.isdir(figures_dir):
            figure_files = [os.path.join(figures_dir, f) for f in os.listdir(figures_dir) if
                            f.endswith('.jpg') or f.endswith('.png')]
        else:
            figure_files = []

        # check for .log files
        if os.path.isdir(logs_dir):
            log_files = [os.path.join(logs_dir, l) for l in os.listdir(logs_dir) \
                         if os.path.isfile(os.path.join(logs_dir, l))] \
                        + [os.path.join(target_exp_dir, f) for f in os.listdir(target_exp_dir) if f.endswith('.log')]
        else:
            log_files = []

        # check for model files
        if os.path.isdir(models_dir):
            model_files = [os.path.join(models_dir, m) for m in os.listdir(models_dir) \
                           if os.path.isfile(os.path.join(models_dir, m))]
        else:
            model_files = []

        if prompt_delete_existing and (len(figure_files) > 0 or len(log_files) > 0 or len(model_files) > 0):
            # TODO: print some of the latest figures, logs and models so we can see what epoch
            # these experiments trained until
            print(
                'Remove \n\t{} figures from {}\n\t{} logs from {}\n\t{} models from {}?[y]es / [n]o (create new folder) / [C]ontinue existing / remove [m]odels too: [y/n/C/m]'.format(
                    len(figure_files), figures_dir, len(log_files), logs_dir, len(model_files), models_dir))

            if debug_delete_input:
                print('Debug input: {}'.format(debug_delete_input))
                choice = debug_delete_input
            else:
                choice = input().lower()

            remove_choices = ['yes', 'y', 'ye']
            make_new_choices = ['no', 'n']
            continue_choices = ['c', '']
            remove_models_too = ['m']

            if choice in remove_choices:
                for f in figure_files + log_files:
                    print('Removing {}'.format(f))
                    os.remove(f)
            elif choice in remove_models_too:
                for f in figure_files + log_files + model_files:
                    print('Removing {}'.format(f))
                    os.remove(f)
            elif choice in continue_choices:
                print('Continuing in existing folder...')
                break

            elif choice in make_new_choices:
                copy_count += 1
                experiment_name = experiment_base_name + '_{}'.format(copy_count)
                target_exp_dir = os.path.join(exp_root, experiment_name)

                figures_dir = os.path.join(exp_root, experiment_name, 'figures')
                logs_dir = os.path.join(exp_root, experiment_name, 'logs')
                models_dir = os.path.join(exp_root, experiment_name, 'models')
        else:
            break

    if do_rename:
        # simply rename the existing old_exp_dir to exp_dir, rather than creating a new one
        os.rename(existing_exp_dir, target_exp_dir)
    else:
        # create each directory
        if not os.path.isdir(target_exp_dir):
            os.mkdir(target_exp_dir)

    # make subdirectories if they do not exist already
    if not os.path.isdir(figures_dir):
        os.mkdir(figures_dir)
    if not os.path.isdir(logs_dir):
        os.mkdir(logs_dir)
    if not os.path.isdir(models_dir):
        os.mkdir(models_dir)

    return experiment_name, target_exp_dir, figures_dir, logs_dir, models_dir


def _prompt_rename(old_dir, new_dir, debug_input=None):
    print('Rename dir \n{} to \n{} [y/N]?'.format(old_dir, new_dir))

    if debug_input:
        print('Debug input: {}'.format(debug_input))
        choice = debug_input
    else:
        choice = input().lower()

    rename_choices = ['yes', 'y', 'ye']

    if choice in rename_choices:
        return new_dir, True
    else:
        return old_dir, False


def resize_batch(X, scale_factor, interp=cv2.INTER_LINEAR):
    if not isinstance(scale_factor, tuple):
        scale_factor = (scale_factor, scale_factor)

    n = X.shape[0]
    h = X.shape[1]
    w = X.shape[2]
    c = X.shape[3]

    max_chans = 100
    X_temp = np.transpose(X, (1, 2, 3, 0))
    X_temp = np.reshape(X_temp, (h, w, c * n))
    X_resized = []
    # do this in batches
    for bi in range(int(np.ceil(c * n / float(max_chans)))):
        X_batch = X_temp[..., bi * max_chans : min((bi + 1) * max_chans, X_temp.shape[-1])]
        n_batch_chans = X_batch.shape[-1]

        if np.max(X_batch) <= 1.0:
            X_batch = cv2.resize(X_batch * 255, None,
                                 fx=scale_factor[0], fy=scale_factor[1],
                                 interpolation=interp) / 255.
        else:
            X_batch = cv2.resize(X_batch, None, fx=scale_factor[0], fy=scale_factor[1],
                                 interpolation=interp)
        X_resized.append(np.reshape(X_batch, X_batch.shape[:2] + (n_batch_chans,)))
    X_temp = np.concatenate(X_resized, axis=-1)
    h_new = X_temp.shape[0]
    w_new = X_temp.shape[1]
    X_out = np.reshape(X_temp, (h_new, w_new, c, n))
    X_out = np.transpose(X_out, (3, 0, 1, 2))
    return X_out


def compute_frame_delta_maps(
        frames=None, deltas=None,
        delta_intensity_thresh=0.05,
        erode_kernel_size=3, dilate_kernel_size=3,
        verbose=False,
        do_pad_start=False
):
    '''
    assume X is n_frames x h x w x 3
    '''
    if frames is not None and (np.max(frames) > 1. or frames.dtype == np.uint8):
        frames = frames.astype(np.float32) / 255.

    if deltas is not None and deltas.dtype == np.uint8:
        deltas = deltas.astype(np.float32) / 255.

    if deltas is None:
        # rgb to grayscale
        frames = np.mean(frames, axis=-1)[..., np.newaxis]
        if verbose:
            print('Computing deltas on frames of shape {}'.format(frames.shape))
        # compute absolute delta. take absolute difference and average across color channels
        deltas = np.diff(frames, axis=0)

    frame_shape = deltas.shape
    deltas = np.abs(deltas)

    deltas = np.reshape(deltas, (np.prod(deltas.shape[:-1]), -1))
    # print('Postprocessing deltas with thresh {}, erode {}, dilate {}, blur {}'.format(
    #    attn_thresh, attn_erode_sigma, attn_dilate_sigma, attn_blur_sigma))
    delta_intensity_thresh = np.asarray(delta_intensity_thresh)
    if delta_intensity_thresh.size == 1:
        delta_intensity_thresh = np.tile(np.reshape(
            delta_intensity_thresh, (1, 1)), (deltas.shape[0], deltas.shape[-1]))
    else:
        delta_intensity_thresh = np.tile(np.reshape(
            delta_intensity_thresh, (1, frame_shape[-1])), (deltas.shape[0], 1))

    bin_deltas_flat = np.zeros((deltas.shape[0], 1))  # binary map
    bin_deltas_flat[np.any(deltas > delta_intensity_thresh, axis=-1)] = 1.
    deltas = np.reshape(bin_deltas_flat, frame_shape[:-1] + (1,))

    # roll frames into batches so we can blur
    batch_size = 100

    for bi in np.arange(0, deltas.shape[0], step=batch_size):
        if erode_kernel_size > 0:
            deltas[bi:min(bi + batch_size, deltas.shape[0])] \
                = erode_batch(deltas[bi:min(bi + batch_size, deltas.shape[0])], erode_kernel_size)
        if dilate_kernel_size > 0:
            deltas[bi:min(bi + batch_size, deltas.shape[0])] \
                = dilate_batch(deltas[bi:min(bi + batch_size, deltas.shape[0])], dilate_kernel_size)

    # normalize each delta in each video to 0, 1
    max_deltas = np.max(deltas, axis=(1, 2))
    for fi in range(deltas.shape[0]):
        if max_deltas[fi] > 1e-5:
            deltas[fi] /= max_deltas[fi]

    if do_pad_start:
        # add a 0 delta map to the beginning so that
        # the output volume is the same size as the input frames
        deltas = np.concatenate([
            np.zeros(deltas[[0]].shape),
            deltas
        ], axis=0)

    return deltas


def erode_batch(X, ks):
    ks = int(ks)
    n = X.shape[0]
    h = X.shape[1]
    w = X.shape[2]
    c = X.shape[3]
    X_temp = np.transpose(X, (1, 2, 3, 0))
    X_temp = np.reshape(X_temp, (h, w, -1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    #	X_temp = cv2.resize( X_temp, None, fx=scale_factor, fy=scale_factor )
    X_temp = cv2.erode(X_temp, kernel)
    h_new = X_temp.shape[0]
    w_new = X_temp.shape[1]
    X_out = np.reshape(X_temp, (h_new, w_new, c, n))
    X_out = np.transpose(X_out, (3, 0, 1, 2))
    return X_out

def dilate_batch(X, ks):
    ks = int(ks)
    n = X.shape[0]
    h = X.shape[1]
    w = X.shape[2]
    c = X.shape[3]
    X_temp = np.transpose(X, (1, 2, 3, 0))
    X_temp = np.reshape(X_temp, (h, w, -1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    #	X_temp = cv2.resize( X_temp, None, fx=scale_factor, fy=scale_factor )

    X_temp = cv2.dilate(X_temp, kernel)
    h_new = X_temp.shape[0]
    w_new = X_temp.shape[1]
    X_out = np.reshape(X_temp, (h_new, w_new, c, n))
    X_out = np.transpose(X_out, (3, 0, 1, 2))
    return X_out

def parse_loss_name(ln, normalize_input=False, pred_shape=None, logger=None, n_chans=3):
    if ln is None:
        lf = 'mean_squared_error'
    elif ln == 'l2':
        lf = keras_losses.mean_squared_error
    elif ln == 'l1' or 'l1-gan' in ln or 'l1-wgan' in ln:  # we handle the gan loss separately
        lf = keras_losses.mean_absolute_error
    elif 'l2-vgg' in ln:
        vgg_net = metrics.vgg_isola_norm(
            shape=pred_shape,
            normalized_inputs=normalize_input,
            )
        vgg_net.summary(line_length=120, print_fn=logger.debug)

        '''
        # compute vgg features for each frame
        lf = my_metrics.SummedLosses(
            loss_fns=[
                keras_losses.mean_squared_error,
                my_metrics.VggFeatLoss(feat_net=vgg_net).compute_loss],
            loss_weights=[1., 1.],
            ).compute_loss
        '''
        # actually return a list of loss functions
        lf = [keras_losses.mean_squared_error, metrics.VggFeatLoss(feat_net=vgg_net).compute_loss]
        ln = ['l2', 'vgg']
    elif 'l1-vgg' in ln:
        vgg_net = metrics.vgg_isola_norm(
            shape=pred_shape,
            normalized_inputs=normalize_input,
            )
        vgg_net.summary(line_length=120, print_fn=logger.debug)

        '''
        # compute vgg features for each frame
        lf = my_metrics.SummedLosses(
            loss_fns=[
                keras_losses.mean_squared_error,
                my_metrics.VggFeatLoss(feat_net=vgg_net).compute_loss],
            loss_weights=[1., 1.],
            ).compute_loss
        '''
        # actually return a list of loss functions
        lf = [keras_losses.mean_absolute_error, metrics.VggFeatLoss(feat_net=vgg_net).compute_loss]
        ln = ['l1', 'vgg']
    elif 'vgg' in ln:
        vgg_net = metrics.vgg_isola_norm(
            shape=pred_shape,
            normalized_inputs=normalize_input,
            )
        vgg_net.summary(line_length=120, print_fn=logger.debug)

        # compute vgg features for each frame
        lf = [metrics.VggFeatLoss(feat_net=vgg_net).compute_loss]
        ln = ['vgg']
    elif ln == 'bce':
        lf = keras_losses.binary_crossentropy

    if not isinstance(lf, list):
        lf = [lf]
    if not isinstance(ln, list):
        ln = [ln]

    return lf, ln
