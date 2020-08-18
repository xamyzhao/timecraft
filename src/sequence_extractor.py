import functools
import os
import sys
import time

import numpy as np

from src.utils import utils, vis_utils
from src.dataset import dataset_utils, frame_filter_utils

def _apply_cpu_aug(self, firstlast_stack_batch, frames_seq_batch):
    # only augment within the central area
    # TODO: should ideally handle different corners if we are combining multiple datasets
    start_corner_xy = self.datasets[0].vids_train[0]['start_corner_xy']
    end_corner_xy = self.datasets[0].vids_train[0]['end_corner_xy']

    im_area_xs, im_area_ys = np.meshgrid(
        range(int(np.floor(start_corner_xy[0, 0])), int(np.ceil(end_corner_xy[0, 0]))),
        range(int(np.floor(start_corner_xy[0, 1])), int(np.ceil(end_corner_xy[0, 1]))),
    )

    # make these all into one volume so we can aug with the same random sat
    frames_to_aug_flat = np.concatenate([
        np.reshape(f, f.shape[:3] + (-1,))
        for f in [frames_seq_batch, firstlast_stack_batch]], axis=-1)

    im_area = frames_to_aug_flat[:, im_area_ys, im_area_xs]
    # TODO: refactor augmentation so we can apply it with fixed params
    im_area, aug_params = aug_utils.aug_im_batch(
        im_area,
        **self.cpu_aug_params
    )
    frames_to_aug_flat[:, im_area_ys, im_area_xs] = im_area
    frames_seq_batch_flatchans = frames_to_aug_flat[:, :, :, :np.prod(frames_seq_batch.shape[3:])]
    frames_seq_batch = np.reshape(frames_seq_batch_flatchans, frames_seq_batch.shape)

    firstlast_stack_batch_flatchans = frames_to_aug_flat[:, :, :, frames_seq_batch_flatchans.shape[-1]:]
    firstlast_stack_batch = np.reshape(firstlast_stack_batch_flatchans, firstlast_stack_batch.shape)
    return firstlast_stack_batch, frames_seq_batch


def _apply_op_to_im_stack(ims_list, op, do_reshape_to=None, output_shapes=None, axis=-1, use_output_idx=None):
    # assumes each im in ims_list is of size batch_size x h x w x ...
    n_ims_in_stack = len(ims_list)
    orig_im_shapes = [im.shape for im in ims_list]

    if do_reshape_to is not None:
        ims_list = [np.reshape(im, do_reshape_to) for im in ims_list]

    if output_shapes is None:
        output_shapes = orig_im_shapes

    im_sizes = [im.shape[axis] for im in ims_list]
    split_idxs = np.cumsum(im_sizes)

    # stack along axis and apply op
    im_stack = np.concatenate(ims_list, axis=axis)
    out = op(im_stack)

    if use_output_idx is not None:
        ims_out = out[use_output_idx]
    else:
        ims_out = out

    if not isinstance(ims_out, np.ndarray):
        # if our op returned a list (e.g. a list of random crops)
        ims_out_temp = [None] * n_ims_in_stack

        for i, im_out in enumerate(ims_out):
            # split the stack into individual ims
            curr_ims_out = np.split(im_out, split_idxs[:-1], axis=axis) 
            if do_reshape_to is not None:
                # reshape ims to original sizes (e.g. if there was a time dimension)
                curr_ims_out = [np.reshape(im, output_shapes[j]) 
                    for j, im in enumerate(curr_ims_out)]

            # now re-organize these nested lists so that we have im stacks on the outer list
            # and crops on the inner list
            for isi, im_out_temp in enumerate(ims_out_temp):
                if ims_out_temp[isi] is None:
                    ims_out_temp[isi] = [curr_ims_out[isi]]
                else:
                    ims_out_temp[isi].append(curr_ims_out[isi])

        ims_list = ims_out_temp
    else:
        ims_list = np.split(ims_out, split_idxs[:-1], axis=axis)

        if do_reshape_to is not None: # TODO: add a flag for this? this will not be None when we need to reshape the inputs
            # reshape to original sizes
            ims_list = [np.reshape(im, output_shapes[i]) for i, im in enumerate(ims_list)]

    return tuple(ims_list)


def _generate_intermediate_seqs(vids_data_list, seq_infos,
                                batch_size=8, n_pred_frames=1,
                                n_prev_frames=1, n_prev_attns=1,
                                target_type='frame',
                                crop_type='stretch', crop_shape=(50, 50, 3),
                                cpu_aug_params=None, gpu_aug_params=None,
                                # TODO: wrap up additional aug params
                                aug_scale_range=0.5,
                                randomize=False, do_aug=False,
                                do_normalize_frames=False,
                                ):
    n_seqs = len(seq_infos)
    seq_idxs_batch = np.asarray(list(range(batch_size)), dtype=int) - batch_size

    while True:
        if randomize:
            # assumes that we always have more sequences than batch size...
            seq_idxs_batch = np.random.choice(n_seqs, batch_size, replace=True)
        else:
            seq_idxs_batch += batch_size
            seq_idxs_batch[seq_idxs_batch >= n_seqs] -= n_seqs

        seq_infos_batch = [seq_infos[si] for si in seq_idxs_batch]

        # get frames corresponding to sequence information tuples
        frames_seq_batch = []
        firstlast_stack_batch = []
        seq_imfiles_batch = []
        for ei, seq_info in enumerate(seq_infos_batch):
            # decompose into parts
            vid_idx, curr_seq_frame_idxs, firstlast_idxs = seq_info

            if None not in curr_seq_frame_idxs:
                curr_seq_frames = vids_data_list[vid_idx]['frames'][curr_seq_frame_idxs]
            else:
                curr_vid_frames = vids_data_list[vid_idx]['frames']
                # None should only occur in the first location (starter sequence)
                curr_seq_frames = np.concatenate([
                    np.ones((1,) + curr_vid_frames.shape[1:], dtype=np.uint8) * 255 if fi is None \
                        else curr_vid_frames[[fi]] for fi in curr_seq_frame_idxs], axis=0)

            curr_seq_frames = curr_seq_frames[..., np.newaxis, :]  # make into seq_len x h x w x 1 x c
            frames_seq_batch.append(curr_seq_frames)
            # this should already be 1 x h x w x c
            firstlast_stack_batch.append(vids_data_list[vid_idx]['frames'][firstlast_idxs][0])

            seq_imfiles_batch.append(
                ','.join([
                    vids_data_list[vid_idx]['frame_ids'][fi] if fi is not None else 'blank'
                    for fi in curr_seq_frame_idxs])
            )

        frames_seq_batch = np.concatenate(frames_seq_batch, axis=-2).transpose(3, 1, 2, 0,
                                                                               4)  # swap batch and time axes

        firstlast_stack_batch = np.concatenate(
            [frame[np.newaxis] for frame in firstlast_stack_batch], axis=0) / 255.

        frames_seq_batch = np.concatenate(
            [frame[np.newaxis] for frame in frames_seq_batch], axis=0) / 255.

        # color augmentation on the cpu, do this before normalization since we might need to convert color spaces
        if do_aug and cpu_aug_params is not None:
            # firstlast_stack_batch, frames_seq_batch = _apply_cpu_aug(firstlast_stack_batch, frames_seq_batch)
            firstlast_stack_batch, frames_seq_batch = _apply_op_to_im_stack(
                [firstlast_stack_batch, frames_seq_batch],
                do_reshape_to=(batch_size,) + crop_shape[:-1] + (-1,),
                op=functools.partial(aug_utils.aug_im_batch,
                                     **cpu_aug_params
                                     ),
                use_output_idx=0,
            )

        # normalize to range [-1, 1]
        frames_seq_batch = vis_utils.normalize(frames_seq_batch)
        firstlast_stack_batch = vis_utils.normalize(firstlast_stack_batch)

        # now slice into previous (which will be used as inputs) and current (which we want to predict)
        prev_frames_batch = frames_seq_batch[..., :-n_pred_frames, :]
        assert prev_frames_batch.shape[-2] == n_prev_frames
        curr_frames_batch = frames_seq_batch[..., -n_pred_frames:, :]  # last frame


        # flatten time into channels dimensions
        firstlast_stack_batch = np.reshape(
            firstlast_stack_batch, (batch_size,) + crop_shape[:-1] + (-1,))
        prev_frames_batch = np.reshape(
            prev_frames_batch, (batch_size,) + crop_shape[:-1] + (n_prev_frames, -1))
        curr_frames_batch = np.reshape(
            curr_frames_batch, (batch_size,) + crop_shape[:-1] + (n_pred_frames, -1))

        yield firstlast_stack_batch, prev_frames_batch, curr_frames_batch, seq_imfiles_batch

# only used for pretraining
def generate_random_frames_sequences(vids_data_list, seq_infos,
                                      batch_size=8,
                                      n_prev_frames=1, n_prev_attns=1,
                                      target_type='frame',
                                      crop_type='stretch', crop_shape=(50, 50, 3),
                                      cpu_aug_params=None, gpu_aug_params=None,
                                      randomize=False, do_aug=False,
                                      do_normalize_frames=False,
                                      return_ids=False,
                                      ):
    seqs_gen = _generate_intermediate_seqs(
        vids_data_list, seq_infos,
        batch_size, n_pred_frames=1,
        n_prev_frames=n_prev_frames, n_prev_attns=n_prev_attns,
        target_type=target_type,
        crop_type=crop_type, crop_shape=crop_shape,
        cpu_aug_params=cpu_aug_params, gpu_aug_params=gpu_aug_params,
        randomize=randomize,
    )

    while True:
        firstlast_stack_batch, prev_frames_batch, curr_frames_batch, \
        seq_imfiles_batch = next(seqs_gen)

        # make into a list of len n_prev_frames, where each element is a batch of prev frames
        # at times t-P, ..., t-2, t-1
        prev_frames_batch = np.split(prev_frames_batch, prev_frames_batch.shape[-2], axis=-2)
        prev_frames_batch = [f[..., 0, :] for f in prev_frames_batch]
        cond_inputs = [firstlast_stack_batch] + prev_frames_batch

        # get rid of time dimension. in this csae n_pred_frames is always 1
        curr_frames_batch = np.reshape(curr_frames_batch, (batch_size,) + crop_shape)
        ########## compile lists of inputs and targets! ############################
        ae_inputs = cond_inputs + [curr_frames_batch]


        inputs = ae_inputs + cond_inputs

        # TODO: make all basic_network_utils take in an aug matrix. it can be identity if no aug

        if not return_ids:
            yield inputs, curr_frames_batch
        else:
            yield inputs, curr_frames_batch, seq_imfiles_batch



def _extract_sequences_from_vids_list(vid_infos_list,
                                      seq_len,
                                      n_prev_frames=1,
                                      include_attn=False,
                                      attn_params=None,
                                      frame_delta_range=None,
                                      min_attn_area=None, max_attn_area=None,
                                      do_use_seg_end=True,
                                      do_filter_by_prev_attn=True,  # check attention area for frame t-p? the first attention in a sequence
                                      include_nonadj_seqs=True,
                                      include_starter_seq=False,
                                      true_starts_file=None,
                                      only_load_end=False,
                                      min_good_seg_len=None,
                                      exclude_file_pairs_file=None,
                                      _print=print,
                                      ):
    '''
    :param vid_infos_list: list of video dictionaries, with keys frames, im_files, bounds
    :return:
    frames: list with n_vids elements, with each element being n_pairs x 2 x h x w x 3
    bounds: list with n_vids elements, with each element being n_pairs(tiled) x h x w x 3
    ims_list: list with n_vids elements, with each element being a list of n_pairs of "im_file_0, im_file_1" strings
    '''
    if include_attn:
        aux_maps = []
    else:
        aux_maps = None

    if seq_len <= 1:
        # TODO: a bit hacky, but all attns will be 0 if our sequences are len 1
        min_attn_area = None
        max_attn_area = None

    # TODO: need to exclude frames that cross any affine shift boundaries as well
    exclude_frame_ranges = None
    exclude_files = None
    if exclude_file_pairs_file is not None:
        exclude_frame_ranges = {}
        exclude_files = []
        if not isinstance(exclude_file_pairs_file, list):
            exclude_file_pairs_file = [exclude_file_pairs_file]

        for efpf in exclude_file_pairs_file:
            with open(efpf, 'r') as f:
                exclude_file_pairs = f.readlines()
            exclude_file_pairs = [fp.strip() for fp in exclude_file_pairs]
            exclude_files += [fp.split(',')[-1].strip() for fp in exclude_file_pairs]
            for file_pair in exclude_file_pairs:
                vid_name = file_pair.split(',')[0].split('_frame')[0]
                frame_range = utils.filenames_to_im_ids(file_pair.split(','))
                if vid_name in exclude_frame_ranges:
                    exclude_frame_ranges[vid_name].append(frame_range)
                else:
                    # initialize a list of tuples representing start, end of bad segments
                    exclude_frame_ranges[vid_name] = [frame_range]

    # if we recorded which video files actually contained the start of the art in a file,
    # then use this list to filter where we get starter sequences from
    has_true_start_vid_names = None
    if true_starts_file is not None:
        with open(true_starts_file, 'r') as f:
            has_true_start_vid_names = f.readlines()
        has_true_start_vid_names = [vn.split('\n')[0] for vn in has_true_start_vid_names]

    frame_seqs = []
    seqs_im_files = []
    seqs_vidnames = []
    firstlast_frames = []
    all_seqs_frame_idxs = []
    all_seqs_firstlast_idxs = []
    all_seqs_attn_idxs = []
    all_seqs_isdone = []

    # _print('Extracting sequences of len {} from {} vids'.format(seq_len, len(vid_infos_list)))
    # _print(F'Using filtering params: frame_delta_range {frame_delta_range}, '
    #        F'min_attn_area {min_attn_area}, max_attn_area {max_attn_area}')
    # _print(F'include_nonadj_seqs: {include_nonadj_seqs}, include_starter_seq: {include_starter_seq}')
    # simply take all pairs and concatenate them
    for vi, vid_info in enumerate(vid_infos_list):
        _print('Creating sequences for vid {} of {}: {}'.format(vi, len(vid_infos_list), vid_info['vid_name']))
        _print('Frames shape: {}'.format(vid_info['frames'].shape))

        # for each video, we will keep a list of ALL video frames, and ALL attns.
        # however, we will keep track of sequences using indices instead of actual arrays, to save space
        curr_vid_frames = vid_info['frames']

        if len(curr_vid_frames) < 2:
            _print('Skipping video with frames shape {}'.format(curr_vid_frames.shape))
            continue

        if not curr_vid_frames.dtype == np.uint8:
            _print('Converting frames to uint8 for space!')
            if np.max(curr_vid_frames) <= 1:
                curr_vid_frames = (curr_vid_frames * 255).astype(np.uint8)
            else:
                curr_vid_frames = curr_vid_frames.astype(np.uint8)

        curr_vid_framenums = np.asarray(utils.filenames_to_im_ids(vid_info['frame_ids']))
        total_attn_area = curr_vid_frames.shape[1] * curr_vid_frames.shape[2] # compute from frames instead of attn, in case we have already summed attn previously
        assert len(curr_vid_frames) == len(curr_vid_framenums)
        ################ set up our sequences of indices, these will track which sequences we extracted ###########
        # set up attns first in case we need to re-compute them when we get new frame idxs
        # also keep track of attn sequences corresponding to the frame seqs
        if min_attn_area is not None or max_attn_area is not None:
            if 'attn' not in vid_info or vid_info['attn'] is None:
                vid_info['attn'] = utils.compute_frame_delta_maps(
                        frames=curr_vid_frames / 255., do_pad_start=True,
                        )

            _print('Attn maps shape: {}'.format(vid_info['attn'].shape))
            #seqs_attns = [vid_info['attn'][seq] for seq in seqs_frame_idxs]
            curr_vid_attns = vid_info['attn']

            # these indices reference into our curr_vid_attns matrix,
            # which should start out the same size as curr_vid_frames
            seqs_attn_idxs = []
        else:
            curr_vid_attns = None
            seqs_attn_idxs = []

        # these indices reference into our curr_vid_frames matrix
        # adjacent indices will not work if we have a minimum frame delta
        if (frame_delta_range is not None and frame_delta_range[0] > 0) or include_nonadj_seqs:
            _print('Looking for nonadj sequences of len {}'.format(seq_len))
            # if we havent computed this or we need to update it
            if 'has_valid_deltas' not in vid_info or not len(curr_vid_framenums) == vid_info['has_valid_deltas'].shape[0]:
                if 'has_valid_deltas' in vid_info and len(curr_vid_framenums) < vid_info['has_valid_deltas'].shape[0]:
                    # if we pruned the dataset since last time we computed deltas, attn maps will no longer match. so recompute them
                    if len(curr_vid_frames) > 1:
                        curr_base_attns = utils.compute_frame_delta_maps(
                                frames=curr_vid_frames / 255., do_pad_start=True,
                                **attn_params
                                )
                    else:
                        curr_base_attns = np.zeros(vid_info['attn'][[0]].shape)
                else:
                    curr_base_attns = vid_info['attn'][:len(curr_vid_framenums)]

                seqs_frame_idxs, has_valid_deltas = frame_filter_utils.enumerate_nonadj_seqs(
                    curr_vid_framenums, curr_base_attns, # in case we added more earlier
                    frame_delta_range, min_attn_area,
                    max_attn_area * total_attn_area if max_attn_area is not None else None,
                    seq_len=seq_len,
                    greedy=True,
                    vid_name=None,#vid_info['vid_name'] if not 'was_pruned' in vid_info else None # used for caching, but speedup isnt significant enough
                )
                del curr_base_attns
                # include the computed valid deltas matrix in our vid info in case we need to use it later
                #vid_infos_list[vi]['has_valid_deltas'] = has_valid_deltas.astype(np.bool)
                #vid_infos_list[vi]['valid_deltas_framenums'] = curr_vid_framenums[:]

            elif 'has_valid_deltas' in vid_info and len(curr_vid_framenums) == vid_info['has_valid_deltas'].shape[0]:
                seqs_frame_idxs, _ = frame_filter_utils.enumerate_nonadj_seqs(
                    curr_vid_framenums, vid_info['attn'][:len(curr_vid_framenums)], frame_delta_range, min_attn_area,
                    max_attn_area * total_attn_area if max_attn_area is not None else None,
                    seq_len=seq_len,
                    has_valid_deltas=vid_info['has_valid_deltas'],
                    greedy=True,
                )
            '''
            additional_attn_keys = []
            additional_attns = []
            last_attn_idx = vid_info['attn'].shape[0]
            # need to re-compute attentions if we're dealing with mostly non-adjacent sequences
            for si, seq in enumerate(seqs_frame_idxs):
                if np.all(np.diff(seq) == 1):
                    # all frames are exactly adjacent, so we'll just use pre-computed attns
                    seqs_attn_idxs += [seq]
                    continue
                seq_diffs = np.reshape(np.diff(curr_vid_frames[seq] / 255., axis=0),
                               (len(seq) - 1,) + curr_vid_frames.shape[1:])

                curr_additional_seq_attns = utils.compute_attention_maps(
                        deltas=seq_diffs,
                        attn_thresh=attn_params['attn_thresh'],
                        attn_blur_sigma=attn_params['attn_blur_sigma'],
                        attn_dilate_sigma=attn_params['attn_dilate_sigma'],
                        attn_erode_sigma=attn_params['attn_erode_sigma'],
                        )

                curr_seq_attn_idxs = [seq[0]] # just use the attention between fi-1 and fi for the first frame in this additional sequence
                for fi, frame_idx in enumerate(seq[1:]):
                    attn_key = (seq[fi - 1], frame_idx) # this attn is between these frame idxs
                    if attn_key not in additional_attn_keys:
                        additional_attn_keys.append(attn_key)
                        additional_attns.append(curr_additional_seq_attns[[fi]])
                        curr_attn_idx = len(additional_attn_keys) - 1
                    else:
                        curr_attn_idx = additional_attn_keys.index(attn_key)

                    # any new attn maps will be appended to the end of the precomputed adjacent attns, so
                    # we need to be careful to index into them properly
                    curr_seq_attn_idxs.append(last_attn_idx + curr_attn_idx)

                seqs_attn_idxs += [curr_seq_attn_idxs]

            if len(additional_attns) > 0:
                if len(curr_vid_attns.shape) > 2:
                    additional_attns = np.concatenate(additional_attns, axis=0)
                else:
                    # only save areas instead
                    additional_attns = np.sum(
                            np.concatenate(additional_attns, axis=0),
                        axis=(1, 2, 3))[..., np.newaxis]
                curr_vid_attns = np.concatenate([curr_vid_attns, additional_attns], axis=0)
            _print('Added {} new attn maps to {} precomputed maps'.format(len(additional_attns), last_attn_idx))
            '''
            #if len(additional_attns) > 0:
            #    # make sure all of our newly added attention idx sequences reference valid volumes
            #    assert np.all([ai < curr_vid_attns.shape[0] for attn_seq in seqs_attn_idxs[-len(additional_attns):] for ai in attn_seq])

        else:
            _print('Taking only adj sequences of len {}'.format(seq_len))
            seqs_frame_idxs = [list(range(i, i + seq_len))
                               for i in range(vid_info['frames'].shape[0] - seq_len + 1)]
            # these indices reference into our curr_vid_attns matrix,
            # which should start out the same size as curr_vid_frames
            #seqs_attn_idxs = [seq[:] for seq in seqs_frame_idxs]
        _print('Initial seqs list takes up {:.1f} MB'.format(sys.getsizeof(seqs_frame_idxs) / 1e6))
        # keep track of firstlast (or just last) frames for each seq, since this might not be the true first/last of the video
        # if we are doing filtering based on segments
        # these indices reference into our curr_vid_frames matrix
        firstlast_idxs = [-1]
        seqs_firstlast_idxs = [firstlast_idxs for _ in seqs_frame_idxs]

        # first remove sequences that contain unacceptable framenum deltas
        if frame_delta_range is not None:
            seqs_frame_idxs, seqs_firstlast_idxs = frame_filter_utils.apply_filter(
                curr_vid_framenums,
                seqs_frame_idxs, seqs_firstlast_idxs, curr_vid_attns,
                filter_fn=functools.partial(frame_filter_utils.filter_seqs_by_framenum_deltas,
                                            frame_delta_range=frame_delta_range),
                filter_name='frame_delta_range',
                filter_info=frame_delta_range,
                _print=_print
            )

        '''
        if include_nonadj_seqs and seq_len > 1:
            _print('Looking for non-adjacent sequences!')
            ############## now look for additional sequences that we could add #############################
            n_seqs_before = len(seqs_frame_idxs)

            if 'has_valid_deltas' not in vid_info:
                all_additional_seqs, has_valid_deltas = enumerate_nonadj_seqs(
                    curr_vid_framenums, vid_info['attn'][:len(curr_vid_framenums)] if vid_info['attn'] is not None else None, frame_delta_range, min_attn_area,
                    max_attn_area * total_attn_area if max_attn_area is not None else None,
                    seq_len=seq_len
                )
                # include the computed valid deltas matrix in our vid info in case we need to use it later
                vid_infos_list[vi]['has_valid_deltas'] = has_valid_deltas.astype(np.bool)
            else:
                all_additional_seqs, _ = enumerate_nonadj_seqs(
                    curr_vid_framenums, vid_info['attn'][:len(curr_vid_framenums)], frame_delta_range, min_attn_area,
                    max_attn_area * total_attn_area if max_attn_area is not None else None,
                    seq_len=seq_len,
                    has_valid_deltas=vid_info['has_valid_deltas']
                )
            seqs_to_add = []
            seqs_to_add_framenums = [] # for display purposes
            seq_frame_diffs = []  # for computing attn
            for seq in all_additional_seqs:
                if seq not in seqs_frame_idxs:
                    seqs_to_add.append(seq)
                    seqs_to_add_framenums.append([curr_vid_framenums[fi] for fi in seq])
                    if np.max(seq) >= len(curr_vid_framenums):
                        print(seq)
                        print(vid_info['vid_name'])
                        sys.exit()
            seqs_frame_idxs += seqs_to_add
            seqs_firstlast_idxs += [firstlast_idxs for added_seq in seqs_to_add]
            _print('Found {} seqs with non-adj frames: idxs {} framenums {}'.format(
                len(seqs_frame_idxs) - n_seqs_before,
                seqs_to_add[:min(len(seqs_to_add),5)],
                seqs_to_add_framenums[:min(len(seqs_to_add_framenums), 5)]
            ))

            # if include_attn or min_attn_area is not None or max_attn_area is not None:
            #     additional_attn_keys = []
            #     additional_attns = []
            #
            #     # append these new attention maps to the end of our precomputed attn maps for this video.
            #     # make sure we update the attn idxs appropriately
            #     last_attn_idx = curr_vid_attns.shape[0]
            #     # compute attn for the newly added sequences right here
            #     for si, seq in enumerate(seqs_to_add):
            #         seq_diffs = np.reshape(np.diff(curr_vid_frames[seq] / 255., axis=0),
            #                        (len(seq) - 1,) + curr_vid_frames.shape[1:])
            #
            #         curr_additional_seq_attns = utils.compute_attention_maps(
            #                 deltas=seq_diffs,
            #                 attn_thresh=attn_params['attn_thresh'],
            #                 attn_blur_sigma=attn_params['attn_blur_sigma'],
            #                 attn_dilate_sigma=attn_params['attn_dilate_sigma'],
            #                 attn_erode_sigma=attn_params['attn_erode_sigma'],
            #                 )
            #
            #         curr_seq_attn_idxs = [seq[0]] # just use the attention between fi-1 and fi for the first frame in this additional sequence
            #         for fi, frame_idx in enumerate(seq[1:]):
            #             attn_key = (seq[fi - 1], frame_idx) # this attn is between these frame idxs
            #             if attn_key not in additional_attn_keys:
            #                 additional_attn_keys.append(attn_key)
            #                 additional_attns.append(curr_additional_seq_attns[[fi]])
            #                 curr_attn_idx = len(additional_attn_keys) - 1
            #             else:
            #                 curr_attn_idx = additional_attn_keys.index(attn_key)
            #
            #             # any new attn maps will be appended to the end of the precomputed adjacent attns, so
            #             # we need to be careful to index into them properly
            #             curr_seq_attn_idxs.append(last_attn_idx + curr_attn_idx)
            #         seqs_attn_idxs += [curr_seq_attn_idxs]

                # if len(additional_attns) > 0:
                #     if len(curr_vid_attns.shape) > 2:
                #         additional_attns = np.concatenate(additional_attns, axis=0)
                #     else:
                #         # only save areas instead
                #         additional_attns = np.sum(
                #                 np.concatenate(additional_attns, axis=0),
                #             axis=(1, 2, 3))[..., np.newaxis]
                #     curr_vid_attns = np.concatenate([curr_vid_attns, additional_attns], axis=0)
                # _print('Added {} new attn maps to {} precomputed maps'.format(len(additional_attns), last_attn_idx))
                # if len(additional_attns) > 0:
                #     # make sure all of our newly added attention idx sequences reference valid volumes
                #     assert np.all([ai < curr_vid_attns.shape[0] for attn_seq in seqs_attn_idxs[-len(additional_attns):] for ai in attn_seq])

            _print('Added {} seqs with non-adj frames: \nidxs {}...{} \nframenums {}...{}'.format(
                len(seqs_frame_idxs) - n_seqs_before,
                seqs_to_add[:min(len(seqs_to_add),5)],
                seqs_to_add[-min(len(seqs_to_add),5):],
                seqs_to_add_framenums[:min(len(seqs_to_add_framenums), 5)],
                seqs_to_add_framenums[-min(len(seqs_to_add_framenums), 5):]
            ))
        '''
        # remove sequences that contain blank attention maps
        # TODO: this isnt entirely correct, because it doesn't compute attention between non-adjacent frames in the sequences
        if min_attn_area is not None:
            seqs_frame_idxs, seqs_firstlast_idxs = frame_filter_utils.apply_filter(
                curr_vid_framenums, seqs_frame_idxs, seqs_firstlast_idxs, curr_vid_attns,
                filter_fn=functools.partial(frame_filter_utils.filter_by_min_attn_area,
                                            frames=curr_vid_frames,
                                            min_attn_area=min_attn_area,
                                            do_filter_by_prev_attn=do_filter_by_prev_attn,
                                            attn_params=attn_params,
                                            ),
                filter_name='min_attn_area',
                filter_info=min_attn_area,
                _print=_print
            )

        # remove frames with too much attn
        if max_attn_area is not None:
            seqs_frame_idxs, seqs_firstlast_idxs = frame_filter_utils.apply_filter(
                curr_vid_framenums, seqs_frame_idxs, seqs_firstlast_idxs, curr_vid_attns,
                filter_fn=functools.partial(frame_filter_utils.filter_seqs_by_max_attn_area,
                                            max_attn_area=max_attn_area * total_attn_area,
                                            do_filter_by_prev_attn=do_filter_by_prev_attn,
                                            ),
                filter_name='max_attn_area',
                filter_info=max_attn_area,
                _print=_print
            )

        good_segs_bounds = None # TODO: we dont use this right now

        ############ now append starter frames #########################################
        # Add in [0] * n_prev + [1, 2, ...] sequence here so it doesn't get filtered
        # for having 0 time delta or 0 attn delta
        if include_starter_seq and n_prev_frames > 0 \
            and (has_true_start_vid_names is None
                or (has_true_start_vid_names is not None \
                    and dataset_utils.vid_name_to_vid_piece_name(vid_info['vid_name']) in has_true_start_vid_names)):
            _print('Looking for starter seqs...')
            found_good_starter_seq = True
            # if we have split the dataset's videos into segments, we only want to consider "initial" sequences from the very
            # first segment...
            # we should include a sequence that is basically [0] * n_prev_frames + [1, 2, 3...]

            if seq_len - n_prev_frames > 1:
                # search for shorter candidates that we can combine with our blank frames
                all_starter_seq_candidates, _ = frame_filter_utils.enumerate_nonadj_seqs(
                    curr_vid_framenums, vid_info['attn'][:len(curr_vid_framenums)], # in case we added more earlier
                    frame_delta_range, min_attn_area,
                    max_attn_area * total_attn_area if max_attn_area is not None else None,
                    seq_len=seq_len - n_prev_frames,
                    greedy=True,
                )
            else:
                # otherwise just look in the sequences we've found already
                all_starter_seq_candidates = seqs_frame_idxs

            # initialize a guess for the starter sequence
            # TODO: should sum attn
            # stop at the first frame that would count as a "stroke"
            first_nonzero_attn_idx = 0
            if min_attn_area is not None:
                first_nonzero_attn_idx = [i for i in range(vid_info['attn'].shape[0]) if np.sum(vid_info['attn'][i]) >= min_attn_area]

                if len(first_nonzero_attn_idx) > 0:
                    first_nonzero_attn_idx = min(first_nonzero_attn_idx)
                else:
                    first_nonzero_attn_idx = 0 # just try to look for a seq starting with 0, as we did in the old implementation

            starter_seq_candidates = []
            for initial_seq_start_idx in range(first_nonzero_attn_idx): # stop right before it, since that is the one that had the first attn
                #starter_seq_idxs = [None] * n_prev_frames + [initial_seq_start_idx] * (seq_len - n_prev_frames)
                _print('Looking for starter seqs starting with blank, {}, ...'.format(initial_seq_start_idx))
                curr_starter_seq_candidates = [seq for seq in all_starter_seq_candidates if seq[0] == initial_seq_start_idx]
                starter_seq_candidates += curr_starter_seq_candidates
                if len(curr_starter_seq_candidates) > 0:
                    _print('Adding {} candidates'.format(len(curr_starter_seq_candidates)))

            starter_seqs = [[None] * n_prev_frames + seq[:seq_len - n_prev_frames] for seq in starter_seq_candidates]

            # TODO: check that we no longer need this
            '''
            else:
                # see if we already have some sequences that start with the 0th frame
                #starter_seq_candidate_seq_idxs = [si for si, seq in enumerate(seqs_frame_idxs) if seq[0] == initial_seq_start_idx]
                starter_seqs = [[None] * n_prev_frames + seq[:seq_len - n_prev_frames] for seq in seqs_frame_idxs if seq[0] == initial_seq_start_idx]
            '''

            if len(starter_seqs) > 0:
                #starter_seq_candidate_frame_idxs = seqs_frame_idxs[starter_seq_candidate_seq_idxs[0]]
                   # starter_seq[..., -1, :] = curr_vid_frames[0]
                for starter_seq in starter_seqs:
                    is_good_seq = True

                    '''
                    if include_attn:
                        if seq_len > 1:
                            # the attn should not be 0 between an empty frame and the 0th frame, since
                            # the 0th frame will have a canvas, and possibly a sketch and some paint...
                            initial_attn = timelapse_utils.compute_attention_maps(
                                frames=vid_info['frames'][starter_seq[1:]] / 255.,
                                attn_thresh=attn_params['attn_thresh'],
                                attn_blur_sigma=attn_params['attn_blur_sigma'],
                                attn_dilate_sigma=attn_params['attn_dilate_sigma'],
                                attn_erode_sigma=attn_params['attn_erode_sigma'],
                            )
                            # TODO: check this
                            # attns are missing the first entry, so we need to accomodate
                            starter_seq_attns = np.concatenate(
                                [np.zeros(initial_attn[[0]].shape)] * n_prev_frames + [initial_attn], axis=0)
                            #starter_seq_attns = np.concatenate([
                            #    starter_seq_attns, initial_attn,
                            #    starter_seq_candidate_attns[1:len(starter_seq_candidate_attns) - n_prev_frames]], axis=0)
                            starter_seq_attn_idxs = list(range(curr_vid_attns.shape[0], curr_vid_attns.shape[0] + seq_len))
                    '''
                    # now make sure the 0, 1 part of the sequence satisfies our attn filters
                    # TODO: what about frame diff filters? probably irrelevant...we just need a starter sequence

                    # TODO: do we even care about attn filters? maybe not
                    if good_segs_bounds is not None:
                        # if we are using segs other than start to finish, we need to figure out
                        # which seg this initial sequence falls into
                        initial_seg_idx = [si for si, seg_bounds in enumerate(good_segs_bounds) \
                                           if initial_seq_start_idx <= seg_bounds[-1]]
                        if len(initial_seg_idx) == 0: # if the first frame does not fall into a valid segment, then
                            _print('Rejecting starter seq due to not falling in a good segment')
                            is_good_seq = False

                    do_oversample = True # include duplicates
                    if is_good_seq and (not do_oversample and starter_seq not in seqs_frame_idxs or do_oversample):
                        _print('Including starter sequence {}'.format(starter_seq))
                        '''
                        starter_seq_frames = []
                        for si, fi in enumerate(starter_seq):
                            if fi is None:
                                starter_seq_frames.append(np.ones((1,) + curr_vid_frames.shape[1:], dtype=np.uint8) * 255)
                            else:
                                starter_seq_frames.append((curr_vid_frames[[fi]]).astype(np.uint8))
                        starter_seq_frames = np.concatenate([f[..., np.newaxis, :] for f in starter_seq_frames], axis=-2)
                        '''
                        # append starter seq information to the rest of the sequences
                        #curr_vid_seqs_frames.append(starter_seq_frames)

                        # TODO: the ordering here is weird...we add to the entire collection of frames, but add to the current vid's attns and idxs
                        if seqs_attn_idxs is not None:
                            #curr_vid_attns = np.concatenate([curr_vid_attns, starter_seq_attns], axis=0)
                            seqs_attn_idxs.append(starter_seq)

                        # add the seq idxs to the rest, so we can parse them all together later
                        seqs_frame_idxs.append(starter_seq)
                        # just use the true last sequence for now, we wil lupdate it as necessary in the next section
                        seqs_firstlast_idxs.append(firstlast_idxs)


        # filter out sequences that cross affine shift boundaries
        curr_vid_bad_framenums = []
        if exclude_frame_ranges is not None and vid_name_to_vid_base_name(vid_info['vid_name']) in exclude_frame_ranges:
            curr_vid_affine_bounds = exclude_frame_ranges[vid_name_to_vid_base_name(vid_info['vid_name'])]
            curr_vid_bad_framenums = []

            for bounds in curr_vid_affine_bounds:
                if bounds[1] == bounds[0] + 1:
                    # a bit hacky, but save the midpoint as the bad frame. we will exclude any sequences that contain
                    # this midpoint
                    curr_vid_bad_framenums.append((bounds[0] + bounds[1]) / 2.)
                else:
                    curr_vid_bad_framenums += [i for i in range(bounds[0] + 1, bounds[1])]

            keep_seq_idxs = []
            for si, seq in enumerate(seqs_frame_idxs):
                seq_framenums = [curr_vid_framenums[frame_idx] for frame_idx in seq if frame_idx is not None]
                if not np.any([bad_frame >= min(seq_framenums) and bad_frame <= max(seq_framenums)
                               for bad_frame in curr_vid_bad_framenums]):
                    keep_seq_idxs.append(si)

            removed_seq_idxs = [si for si, seq in enumerate(seqs_frame_idxs) if si not in keep_seq_idxs]

            _print('Removed {} of {} seqs due to affine breaks file {}: {}'.format(
                len(seqs_frame_idxs) - len(keep_seq_idxs), len(seqs_frame_idxs), exclude_file_pairs_file,
                [seqs_frame_idxs[rsi] for rsi in removed_seq_idxs[:min(len(removed_seq_idxs), 3)]]
            ))

            seqs_frame_idxs = [seqs_frame_idxs[i] for i in keep_seq_idxs]
            seqs_firstlast_idxs = [seqs_firstlast_idxs[i] for i in keep_seq_idxs]

            if do_use_seg_end:
                # use the end of the aligned segment as the "final" frame rather than the true completed painting.
                # all good segments shoudl fall in between (but not over) bad frames
                good_seg_bounds = [-1] + curr_vid_bad_framenums
                for bfi, bfn in enumerate(good_seg_bounds[:-1]):
                    if bfn > max(curr_vid_framenums):
                        # if we've passed the end of the video already (e.g. if we filtered the frames to the first half)
                        break

                    n_seqs_updated = 0
                    min_seq_framenum = None
                    max_seq_framenum = None
                    for si, seq_frame_idxs in enumerate(seqs_frame_idxs):
                        # take framenums from non-blank frames
                        curr_seq_framenums = np.asarray([curr_vid_framenums[fi] for fi in seq_frame_idxs if fi is not None])

                        # make sure we're not crossing a bad boundary first
                        if min(curr_seq_framenums) > bfn \
                                and max(curr_seq_framenums) < good_seg_bounds[bfi + 1]:
                            # find the nearest "good" frame idx before the next "bad" frame
                            last_good_idx = max([fi for fi, fn in enumerate(curr_vid_framenums) if fn < good_seg_bounds[bfi + 1]])

                            if min_seq_framenum is None or min(curr_seq_framenums) < min_seq_framenum:
                                min_seq_framenum = min(curr_seq_framenums)

                            if max_seq_framenum is None or max(curr_seq_framenums) > max_seq_framenum:
                                max_seq_framenum = max(curr_seq_framenums)
                            n_seqs_updated += 1

                            seqs_firstlast_idxs[si] = [last_good_idx]

                    if n_seqs_updated > 0:
                        _print('Updated last frame of {} seqs (in range {}-{}) to {} to avoid bad frame {}'.format(
                            n_seqs_updated, min_seq_framenum, max_seq_framenum,
                            curr_vid_framenums[last_good_idx], good_seg_bounds[bfi + 1]
                        ))

        vid_info['attn'] = None # save space
        # TODO: just redo this logic later
        # if min_good_seg_len is not None:
        #     if len(curr_vid_bad_framenums) > 1:
        #         print('bad frame nums {}'.format(curr_vid_bad_framenums))
        #         # compute segment lengths in number of frames, as well as frames
        #         continuous_seg_lens = np.diff(curr_vid_bad_framenums)
        #         print('seg lens {}'.format(continuous_seg_lens))
        #         # keep track of frames that fall into bad segs, since we would probably want to remove those
        #         bad_seg_framenums = []
        #         good_segs_bounds = []
        #         for si, seg_len in enumerate(continuous_seg_lens):
        #             # TODO: shoudl we also enforce a min number of frames within a segment?
        #             if seg_len < min_good_seg_len:  # or continous_seg_lens_idxs < 10:
        #                 # this is an ok sequence
        #                 bad_seg_frame_idxs += list(range(curr_bad_frame_idxs[si], curr_bad_frame_idxs[si + 1]))
        #             else:
        #                 good_segs_bounds.append(
        #                     (curr_bad_frame_idxs[si], curr_bad_frame_idxs[si + 1]))
        #
        #     else:
        #         bad_seg_frame_idxs = []
        #         good_segs_bounds = []
        #
        #     # filter out seqs that overlap with bad segments
        #     keep_seq_idxs = [si for si, seq in enumerate(seqs_frame_idxs)
        #                      if not np.any([fi in bad_seg_frame_idxs for fi in seq])]
        #
        #     _print('Removed {} of {} seqs due to min_good_seg_len {}'.format(
        #         len(seqs_frame_idxs) - len(keep_seq_idxs), len(seqs_frame_idxs), min_good_seg_len))
        #     # now filter all of our seqs
        #     seqs_frame_idxs = [seqs_frame_idxs[i] for i in keep_seq_idxs]
        #     seqs_firstlast_idxs = [seqs_firstlast_idxs[i] for i in keep_seq_idxs]
        #     if seqs_attns is not None:
        #         seqs_attns = [seqs_attns[i] for i in keep_seq_idxs]
        #
        #     for segi, seg_bound_idxs in enumerate(good_segs_bounds):
        #         # for seqs that fall within these good bounds, set their firstlast to be the start and end of the seg
        #         seq_idxs_in_seg = [seqi for seqi, seq in enumerate(seqs_frame_idxs)
        #                            if seq[0] >= seg_bound_idxs[0] and seq[-1] <= seg_bound_idxs[-1]]
        #
        #         for seq_idx in seq_idxs_in_seg:
        #             # filter by firstlast_idxs since we might not even take teh first frame
        #             seqs_firstlast_idxs[seq_idx] = [seg_bound_idxs[0], seg_bound_idxs[-1]][firstlast_idxs]


        assert len(seqs_frame_idxs) == len(seqs_firstlast_idxs)
        #if seqs_attn_idxs is not None:
        #    assert len(seqs_frame_idxs) == len(seqs_attn_idxs)

        # only load sequences that come from the end of the video, so that we can train with an appropriate reconstruction loss
        if only_load_end:
            keep_seq_idxs = [
                si for si, seq in enumerate(seqs_frame_idxs) if seq[-1] == vid_info['frames'].shape[0] - 1
            ]

            _print('Removed {} of {} seqs, only including last seqs'.format(
                len(seqs_frame_idxs) - len(keep_seq_idxs), len(seqs_frame_idxs)))

            seqs_frame_idxs = [seqs_frame_idxs[i] for i in keep_seq_idxs]
            seqs_firstlast_idxs = [seqs_firstlast_idxs[i] for i in keep_seq_idxs]
            if seqs_attn_idxs is not None:
                seqs_attn_idxs = [seqs_attn_idxs[i] for i in keep_seq_idxs]


        n_sequences = len(seqs_frame_idxs)
        if n_sequences == 0:
            continue
        print('Found {} valid seqs'.format(n_sequences))

        #################### now concatenate all the sequences we found ##########################
        # make each sequence into 1 x h x w x seq_len x 3
        #curr_vid_seqs_frames = [np.transpose(curr_vid_frames[p], (1, 2, 0, 3))[np.newaxis] for p in seqs_frame_idxs]

        '''
        if include_attn:
            #  volumes that we can concatenate
            curr_vid_seqs_attns = [np.transpose(
                curr_vid_attns[attn_seq],
                (1, 2, 0, 3))[np.newaxis] for attn_seq in seqs_attn_idxs]
            # TODO: if the seq starts right after a bad frame (e.g. an affinely shifted frame), we should zero out the attn map...
        else:
            del curr_vid_attns
        '''
        assert len(seqs_frame_idxs) == len(seqs_firstlast_idxs)
        #if include_attn:
        #    assert len(seqs_frame_idxs) == len(curr_vid_seqs_attns)

        seq_im_files = []

        for si, seq_frame_idxs in enumerate(seqs_frame_idxs):
            # binary sequences indicating whether the current frame is the very last frame of the video (i.e. equal to the
            # constant last frame)

            # also keep track of which video each sequence came from
            seqs_vidnames.append(vid_info['vid_name'])

            # a list of n_seqs strings, where each string is "im_file_t-2,im_file_t-1,im_file_t"
            seq_im_files.append(','.join([
                vid_info['frame_ids'][frame_idx] if frame_idx is not None \
                else 'blank' \
                for frame_idx in seq_frame_idxs]))

            all_seqs_frame_idxs.append(seq_frame_idxs)
        all_seqs_firstlast_idxs += seqs_firstlast_idxs
        if seqs_attn_idxs is not None:
            all_seqs_attn_idxs += seqs_attn_idxs

        # this entry will be n_seqs(tiled) x h x w x 2 x 3
        #curr_vid_firstlast_frames = np.concatenate([
        #    np.transpose(curr_vid_frames[np.asarray(fl_idxs)], (1, 2, 0, 3))[np.newaxis]
        #    for fl_idxs in seqs_firstlast_idxs], axis=0)

        # firstlast_frames += [curr_vid_firstlast_frames]
        # this entry will be n_pairs h x w x 2 x 3
        # frame_seqs += [np.concatenate(curr_vid_seqs_frames, axis=0)]
        # print(frame_seqs[-1].dtype)
        # print(frame_seqs[-1].nbytes / float(1e6))

        # if include_attn:
            # let's try to just compute attention at generator time for now
            # _print('Updating attn')
            # vid_infos_list[vi]['attn'] = curr_vid_attns

            # this entry will be n_seqs x h x w x seq_len x 1
            # aux_maps += [np.concatenate(curr_vid_seqs_attns, axis=0)]
            # assert frame_seqs[-1].shape[0] == aux_maps[-1].shape[0]
        #assert frame_seqs[-1].shape[0] == firstlast_frames[-1].shape[0]

        # this outer list will contain n_vids lists
        seqs_im_files += [seq_im_files]

        # delete all the frames and attns so we don't eat up memory
        #del vid_info['frames']
        #del vid_info['attn']

    # if len(frame_seqs) == 0:
    #     _print('Found 0 frame sequences for dataset!')
    #     # TODO: a little hacky, but helps to not break things down the road
    #     return np.empty((0,)), np.empty((0,)), np.empty((0,)), [], [], []

    # print([f.shape for f in frame_seqs])
    # now concatenate all seqs from all vids in the 0th dimension
    # frame_seqs = np.concatenate(frame_seqs, axis=0)
    # firstlast_frames = np.concatenate(firstlast_frames, axis=0)

    # flatten the nested lists into a list of len n_vids x n_seqs
    seqs_im_files_stack = []
    for vid_seq_imfiles in seqs_im_files:
        seqs_im_files_stack += [seq_imfiles for seq_imfiles in vid_seq_imfiles]

    return seqs_vidnames, seqs_im_files_stack, all_seqs_frame_idxs, all_seqs_firstlast_idxs


def extract_sequences_by_index_from_datasets(datasets, combined_dataset=None,
                                             seq_len=None, n_prev_frames=None,
                                             _print=print,
                                             do_filter_by_prev_attn=True,  # check attention area for frame t-p? the first attention in a sequence
                                             include_starter_seq=True,
                                             include_nonadj_seqs=True,
                                             only_load_end=False,
                                             do_prune_unused=False,
                                             ):
    '''
    Computes usable training sequences from each dataset, combines them and then re-computes train/validation split since
    different datasets could put a video in train and validation. Overwrites sequences with matching frame names --
    *PROBABLY* only useful if are combining multiple datasets and not replacing entire videos

    :param datasets: a list of WatercolorsDataset instances.
    :param combined_dataset: (optional) a single dataset that contains the train/valid video split that we will use
    :param load_n: number of videos to load from each dataset. Defaults to None to load all.
    :return:
    '''
    seq_infos_train = [] # entries will be tuples: (vid_idx, seq_frame_idxs)
    seq_infos_valid = []  # entries will be tuples: (vid_idx, seq_frame_idxs)

    # if we have a combined dataset, use it for the train/test split
    if combined_dataset is not None:
        train_vid_names = [(v['vid_name']) for v in combined_dataset.vids_train]
        valid_vid_names = [(v['vid_name']) for v in combined_dataset.vids_valid]
    else:
        train_vid_names = []
        valid_vid_names = []

    if datasets is None:
        datasets = [combined_dataset]

    existing_seqs_imnames = []
    for dsi, ds in enumerate(datasets):
        if combined_dataset is None:
            train_vid_names += [(v['vid_name']) for v in ds.vids_train]
            valid_vid_names += [(v['vid_name']) for v in ds.vids_valid]

        if n_prev_frames is None:
            n_prev_frames = ds.params['n_prev_frames']

        if seq_len is None:
            use_seq_len = n_prev_frames + 1
        else:
            use_seq_len = seq_len

        _print('Loading sequences of len {} from dataset {}'.format(seq_len, ds.params['dataset']))
        _print('N vids: {}'.format(len(ds.vids_data)))

        # process only 100 videos at a time, in case we need to prune some videos to save sapce\
        batch_size = 50
        n_vid_batches = int(np.ceil(len(ds.vids_data) / float(batch_size)))

        st = time.time()

        for bi in range(n_vid_batches):
            curr_vids_data = ds.vids_data[bi * batch_size : min(len(ds.vids_data), (bi + 1) * batch_size)]
            curr_vidnames = [vd['vid_name'] for vd in curr_vids_data]
            # load sequences for current dataset. we will overwrite any previously existing sequences
            # frame_seqs, firstlasts, \
            # aux_maps, \
            seqs_vidnames, seqs_imfiles, \
            frame_seq_idxs, firstlast_seq_idxs \
                = _extract_sequences_from_vids_list(
                vid_infos_list=curr_vids_data,
                seq_len=use_seq_len,
                n_prev_frames=n_prev_frames,#ds.params['n_prev_frames'],
                frame_delta_range=ds.params['frame_delta_range'],
                do_use_seg_end=ds.params['do_use_segment_end'],
                _print=_print,
                include_starter_seq=include_starter_seq,
                true_starts_file=ds.params['true_starts_file'],
                do_filter_by_prev_attn=do_filter_by_prev_attn,
                exclude_file_pairs_file=ds.params['affine_breaks_file'],
                only_load_end=only_load_end,
                include_nonadj_seqs=include_nonadj_seqs,
                **ds.params['sequence_params']
            )
            # if we updated the vid data, make sure it is updated in vids_train or vids_valid as well
            # for vd in ds.vids_data:
            #     vn = vd['vid_name']
            #     if vn in train_vid_names and combined_dataset is not None:
            #         combined_dataset.vids_train[train_vid_names.index(vn)]['attn'] = vd['attn']
            #     elif vn in valid_vid_names and combined_dataset is not None:
            #         combined_dataset.vids_valid[valid_vid_names.index(vn)]['attn'] = vd['attn']
            #     # TODO: also handle the case when we arent using a combined dataset?
            # assert len(seqs_vidnames) == len(frame_seq_idxs)

            # first, create sequences that reference the current batch of vids, so we know what to prune
            curr_batch_seq_infos = []
            for si, seq in enumerate(frame_seq_idxs):
                seq_vidname = (seqs_vidnames[si])
                curr_batch_seq_infos.append(
                    (curr_vidnames.index(seq_vidname), seq, firstlast_seq_idxs[si])
                )

            n_starter_seqs = 0
            for seq in frame_seq_idxs:
                if None in seq:
                    n_starter_seqs += 1
            _print('Batch starter seqs {}: {}...'.format(n_starter_seqs, [seq for seq in frame_seq_idxs if None in seq][:5]))
            # TODO: also do this for individual datasets?
            if do_prune_unused and combined_dataset is not None:
                curr_vids_data, curr_batch_seq_infos \
                    = dataset_utils._prune_unused_frames(curr_vids_data, curr_batch_seq_infos, _print=_print)

                combined_dataset_vid_names = [vd['vid_name'] for vd in combined_dataset.vids_data]
                combined_dataset_train_vid_names = [vd['vid_name'] for vd in combined_dataset.vids_train]
                combined_dataset_valid_vid_names = [vd['vid_name'] for vd in combined_dataset.vids_valid]
                # prune frames in the combined dataset
                for vd in curr_vids_data:
                    cdi = combined_dataset_vid_names.index(vd['vid_name'])
                    combined_dataset.vids_data[cdi]['frames'] = vd['frames']
                    combined_dataset.vids_data[cdi]['attn'] = vd['attn']
                    combined_dataset.vids_data[cdi]['frame_ids'] = vd['frame_ids']
                    combined_dataset.vids_data[cdi]['was_pruned'] = True

                    '''
                    # check to see if the training dataset also changed?
                    print(combined_dataset.vids_data[cdi]['frames'].shape)
                    if vd['vid_name'] in combined_dataset_train_vid_names:
                        cdti = combined_dataset_train_vid_names.index(vd['vid_name'])
                        print(combined_dataset.vids_train[cdti]['frames'].shape)
                    else:
                        cdvi = combined_dataset_valid_vid_names.index(vd['vid_name'])
                        print(combined_dataset.vids_valid[cdvi]['frames'].shape)
                    '''
            curr_batch_seq_infos_train = []
            curr_batch_seq_infos_valid = []
            # TODO: also include attn idxs, and update attn volumes in vids_data
            for si, seq_info in enumerate(curr_batch_seq_infos):
                vid_idx, seq, seq_firstlast_idxs  = seq_info
                seq_vidname = curr_vids_data[vid_idx]['vid_name']
                
                if seq_vidname in train_vid_names:
                    curr_batch_seq_infos_train.append(
                        (train_vid_names.index(seq_vidname),) + seq_info[1:])
                elif seq_vidname in valid_vid_names:
                    curr_batch_seq_infos_valid.append(
                        (valid_vid_names.index(seq_vidname),) + seq_info[1:])
                else:
                    raise FileNotFoundError('Sequence vidname {} not found in training or validation!'.format(seqs_vidnames[si]))

            seq_infos_train += curr_batch_seq_infos_train
            seq_infos_valid += curr_batch_seq_infos_valid
        _print('Finding seqs for dataset took {}'.format(time.time() - st))

    return seq_infos_train, seq_infos_valid


def extract_sequences_from_datasets(datasets, combined_dataset=None,
                                    seq_len=None, n_prev_frames=None,
                                    _print=print,
                                    do_filter_by_prev_attn=True,  # check attention area for frame t-p? the first attention in a sequence
                                    include_starter_seq=False,
                                    only_load_start=False,
                                    only_load_end=False
                                    ):
    '''
    Computes sequences from each dataset, combines them and then re-computes train/validation split since
    different datasets could put a video in train and validation. Overwrites sequences with matching frame names --
    *PROBABLY* only useful if are combining multiple datasets and not replacing entire videos

    :param datasets: a list of WatercolorsDataset instances.
    :param combined_dataset: (optional) a single dataset that contains the train/valid video split that we will use
    :param load_n: number of videos to load from each dataset. Defaults to None to load all.
    :return:
    '''
    frame_seqs_all = None
    firstlasts_all = []
    aux_maps_all = []
    seqs_imfiles_all = []
    seqs_vidnames_all = []
    frame_seq_idxs_all = []
    seqs_isdone_all = []

    # if we have a combined dataset, use it for the train/test split
    if combined_dataset is not None:
        train_vid_names = [dataset_utils.vid_name_to_vid_base_name(v['vid_name']) for v in combined_dataset.vids_train]
    else:
        train_vid_names = []

    existing_seqs_imnames = []
    for dsi, ds in enumerate(datasets):
        vids_data = ds.vids_data

        if combined_dataset is None:
            train_vid_names += [dataset_utils.vid_name_to_vid_base_name(v['vid_name']) for v in ds.vids_train]

        if seq_len is None:
            use_seq_len = ds.params['n_prev_frames'] + 1
        else:
            use_seq_len = seq_len

        if n_prev_frames is None:
            n_prev_frames = ds.params['n_prev_frames']

        _print('Loading sequences of len {} from dataset {}'.format(seq_len, ds.params['dataset']))
        _print('N vids: {}'.format(len(ds.vids_data)))

        # load sequences for current dataset. we will overwrite any previously existing sequences
        frame_seqs, firstlasts, \
        aux_maps, \
        seqs_vidnames, seqs_imfiles, \
        frame_seq_idxs, _, _, seqs_isdone \
            = _extract_sequences_from_vids_list(
            vid_infos_list=vids_data,
            include_attn=ds.params['include_attention'],
            seq_len=use_seq_len,
            n_prev_frames=n_prev_frames,#ds.params['n_prev_frames'],
            attn_params=ds.params['attn_params'],
            frame_delta_range=ds.params['frame_delta_range'],
            do_use_seg_end=ds.params['do_use_segment_end'],
            _print=_print,
            include_starter_seq=include_starter_seq,
            true_starts_file=ds.params['true_starts_file'],
            do_filter_by_prev_attn=do_filter_by_prev_attn,
            exclude_file_pairs_file=ds.params['affine_breaks_file'],
            only_load_end=only_load_end,
            **ds.params['sequence_params']
        )

        if only_load_start:
            # for pretraining a dense (RNN-style) director/painter. A bit hacky, but only use sequences that are starter sequences
            keep_seq_idxs = [fsi for fsi in frame_seq_idxs if fsi[0] is None]
            frame_seqs = [frame_seqs[si] for si in keep_seq_idxs]
            firstlasts = [firstlasts[si] for si in keep_seq_idxs]
            if aux_maps is not None:
                aux_maps = [aux_maps[si] for si in keep_seq_idxs]
            seqs_vidnames = [seqs_vidnames[si] for si in keep_seq_idxs]
            seqs_imfiles = [seqs_imfiles[si] for si in keep_seq_idxs]
            frame_seq_idxs = [frame_seq_idxs[si] for si in keep_seq_idxs]
            seqs_isdone = [seqs_isdone[si] for si in keep_seq_idxs]

        # check if any of these sequences have already been loaded. If so, replace them
        existing_seqs_imnames = [
            ','.join([os.path.basename(f) for f in seq_imfiles.split(',')])
            for seq_imfiles in seqs_imfiles_all]

        # each entry should be a string in the format im_file_0,im_file_1,...im_file_T.
        # convert these to filenames (without the paths since those might be different)
        curr_seqs_imnames = [
            ','.join([os.path.basename(f) for f in seq_imfiles.split(',')])
            for seq_imfiles in seqs_imfiles]

        append_seq_idxs = []  # only append sequences that we do not replace
        for si, seq_imnames in enumerate(curr_seqs_imnames):
            if seq_imnames not in existing_seqs_imnames:
                append_seq_idxs.append(si)
            else:
                existing_seq_idx = existing_seqs_imnames.index(seq_imnames)

                if seqs_imfiles_all[existing_seq_idx] == seqs_imfiles[si]:
                    continue

                _print(
                    'Replacing seq {} with {}'.format(
                        seqs_imfiles_all[existing_seq_idx], seqs_imfiles[si]))

                # replace existing seqs with later ones
                frame_seqs_all[existing_seq_idx] = frame_seqs[si]
                firstlasts_all[existing_seq_idx] = firstlasts[si]
                seqs_isdone_all[existing_seq_idx] = seqs_isdone
                if aux_maps is not None:
                    aux_maps_all[existing_seq_idx] = aux_maps[si]
                else:
                    aux_maps_all = None

                seqs_imfiles_all[existing_seq_idx] = seqs_imfiles[si]
                seqs_vidnames_all[existing_seq_idx] = seqs_vidnames[si]

        # _print(F'Loaded seqs of len {seq_len} from dataset {dsi}: {ds.display_name}, '
        #        F'seq frames shape: {frame_seqs.shape}, '
        #        F'seq imfiles len: {len(seqs_imfiles)}')

        if aux_maps is not None:
            # _print(F'Aux maps {aux_maps.shape}')
            print('')
        else:
            aux_maps_all = None

        # concatenate all datasets together
        if frame_seqs_all is None: # first video, simply initialize collection
            frame_seqs_all = frame_seqs
            firstlasts_all = firstlasts

            if aux_maps is not None:
                aux_maps_all = aux_maps

            seqs_isdone_all = seqs_isdone[:]

            seqs_imfiles_all = seqs_imfiles[:]
            seqs_vidnames_all = seqs_vidnames[:]
        else:
            frame_seqs_all = np.append(frame_seqs_all, frame_seqs[append_seq_idxs], axis=0)
            firstlasts_all = np.append(firstlasts_all, firstlasts[append_seq_idxs], axis=0)

            if aux_maps is not None:
                aux_maps_all = np.append(aux_maps_all, aux_maps[append_seq_idxs], axis=0)
            seqs_isdone_all += [seqs_isdone[i] for i in append_seq_idxs]
            seqs_imfiles_all += [seqs_imfiles[i] for i in append_seq_idxs]
            seqs_vidnames_all += [seqs_vidnames[i] for i in append_seq_idxs]

    #train_vid_names = [vd['vid_name'].split('_seg')[0] for vd in combined_dataset.vids_train]
    _print('Training vid names: {}'.format(train_vid_names))
    # split sequences according to how we split the video before
    n_seqs = len(seqs_imfiles_all)
    assert len(seqs_vidnames_all) == n_seqs
    train_idxs = [i for i in range(n_seqs) if dataset_utils.vid_name_to_vid_base_name(seqs_vidnames_all[i]) in train_vid_names]
    valid_idxs = [i for i in range(n_seqs) if dataset_utils.vid_name_to_vid_base_name(seqs_vidnames_all[i]) not in train_vid_names]

    # compile training data
    frame_seqs_train = frame_seqs_all[train_idxs]
    firstlasts_train = firstlasts_all[train_idxs]
    if aux_maps_all is not None:
        aux_maps_train = aux_maps_all[train_idxs]
    else:
        aux_maps_train = None
    seqs_isdone_train = [seqs_isdone_all[i] for i in train_idxs]
    seqs_imfiles_train = [seqs_imfiles_all[i] for i in train_idxs]
    seqs_vidnames_train = [seqs_vidnames_all[i] for i in train_idxs]

    # compile validation data
    frame_seqs_valid = frame_seqs_all[valid_idxs]
    firstlasts_valid = firstlasts_all[valid_idxs]
    if aux_maps_all is not None:
        aux_maps_valid = aux_maps_all[valid_idxs]
    else:
        aux_maps_valid = None
    seqs_isdone_valid = [seqs_isdone_all[i] for i in valid_idxs]
    seqs_imfiles_valid = [seqs_imfiles_all[i] for i in valid_idxs]
    seqs_vidnames_valid = [seqs_vidnames_all[i] for i in valid_idxs]

    _print('Train/valid classes in sequences:')
    # TODO: abstract away logic to get vid_name --> class for each dataset
    unique_train_classes = list(set([seqs_vidnames_all[i] for i in train_idxs]))
    _print(unique_train_classes)
    unique_valid_classes = list(set([seqs_vidnames_all[i] for i in valid_idxs]))
    _print(unique_valid_classes)
    return (frame_seqs_train, firstlasts_train, aux_maps_train, seqs_vidnames_train, seqs_imfiles_train, seqs_isdone_train), \
           (frame_seqs_valid, firstlasts_valid, aux_maps_valid, seqs_vidnames_valid, seqs_imfiles_valid, seqs_isdone_valid)


