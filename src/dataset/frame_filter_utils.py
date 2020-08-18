import os

import numpy as np
import scipy.io as sio

from src.utils import utils

def filter_seqs_by_framenum_deltas(vid_framenums, seqs_frame_idxs, vid_attns,
                                   frame_delta_range):
    curr_vid_framenums = np.asarray(vid_framenums)
    seq_framenum_diffs = [np.diff(curr_vid_framenums[seq]) for seq in seqs_frame_idxs]

    keep_seq_idxs = [si for si, seq in enumerate(seqs_frame_idxs)
                     if np.all(seq_framenum_diffs[si] >= frame_delta_range[0])
                     and np.all(seq_framenum_diffs[si] <= frame_delta_range[1])]

    return keep_seq_idxs



def filter_by_min_attn_area(framenums, seq_frame_idxs, attns,
                            frames,
                            min_attn_area,
                            attn_params=None, do_filter_by_prev_attn=False,
                            attn_areas=None,
                            ):
    if do_filter_by_prev_attn and attn_areas is None:
        # if we want to filter by attns before this sequence, but we dont have
        # unfiltered attn data
        attns = utils.compute_attention_maps(
            frames, do_pad_start=True,
            **attn_params)
        attn_areas = np.sum(attns, tuple(list(range(1, len(attns.shape)))))

    # compute attention from frames just in case?
    # TODO: this might be very slow...consider computing attention area by just summing intermediate attns
    keep_seq_idxs = []
    for si, seq in enumerate(seq_frame_idxs):
        seq_frames = frames[seq]

        seq_attns = utils.compute_frame_delta_maps(seq_frames)

        seq_attn_areas = np.sum(seq_attns, axis=tuple(list(range(1, len(seq_attns.shape)))))
        if do_filter_by_prev_attn:
            seq_attn_areas = np.concatenate([
                attn_areas[[seq[0]]], seq_attn_areas
            ])
        if np.all(seq_attn_areas >= min_attn_area):
            keep_seq_idxs.append(si)
    return keep_seq_idxs

def _test_filter_by_min_attn_area():
    # unit test 1: just filter by everything except prev attns
    frames = np.concatenate([
        np.zeros((1, 5, 5)),
        np.ones((1, 5, 5)),
        np.ones((1, 5, 5)),
    ], axis=0)

    attn_params = {
        'blur_sigma': 0,
        'dilate_sigma': 0,
        'erode_sigma': 0,
    }

    keep_idxs = filter_by_min_attn_area(frames, [[0, 1], [1, 2], [0, 2]],
                                        attn_params=attn_params, min_attn_area=1)
    assert [0, 1] in keep_idxs
    assert [0, 2] in keep_idxs
    assert [1, 2] not in keep_idxs
    print('UNIT TEST: filter by min_attn_area without prev attn filter PASSED')

    keep_idxs = filter_by_min_attn_area(frames, [[0, 1], [1, 2], [0, 2]],
                                        attn_params=attn_params, min_attn_area=1,
                                        do_filter_by_prev_attn=True
                                        )
    assert [0, 1] in keep_idxs
    assert [0, 2] in keep_idxs
    assert [1, 2] not in keep_idxs
    print('UNIT TEST: filter by min_attn_area without prev attn filter PASSED')



def filter_seqs_by_max_attn_area(vid_framenums, seqs_frame_idxs, vid_attns,
                                 do_filter_by_prev_attn, max_attn_area):

    count_attn_thresh = 0.3

    attn_areas = np.sum(vid_attns > count_attn_thresh, tuple(list(range(1, len(vid_attns.shape)))))
    seqs_attn_areas = [attn_areas[seq] for seq in seqs_frame_idxs]

    if do_filter_by_prev_attn:
        keep_seq_idxs = [
            si for si, seq in enumerate(seqs_frame_idxs)
            if np.all(seqs_attn_areas[si] <= max_attn_area)]
    else:
        # ignore the very first attention in teh sequence
        keep_seq_idxs = [
            si for si, seq in enumerate(seqs_frame_idxs)
            if np.all(seqs_attn_areas[si][1:] <= max_attn_area)]
    return keep_seq_idxs


def filter_by_n_frames(framenums, n_frames, vid_min_framenum=None, vid_max_framenum=None, _print=print):

    keep_framenums = np.linspace(min(framenums), max(framenums), num=n_frames, endpoint=True)
    # now figure out the nearest frame idxs to each framenum
    keep_idxs = []
    dists_to_fns = []
    for fn in keep_framenums:
        dists_to_fn = np.abs(framenums - fn)
        keep_idxs.append(int(np.argmin(dists_to_fn)))
        dists_to_fns.append(min(dists_to_fn))

    _print('Keeping {} of {} frames in range {}-{}, with max delta of {} from ideal spacing'.format(
        len(keep_idxs), len(framenums),
        min(framenums), max(framenums), max(dists_to_fns)
    ))
    return keep_idxs


def filter_by_framenum_range(framenums, framenum_range, vid_min_framenum=None, vid_max_framenum=None, _print=print):
    if vid_min_framenum is None:
        vid_min_framenum = min(framenums)
    if vid_max_framenum is None:
        vid_max_framenum = max(framenums)

    min_framenum = vid_min_framenum + int(round(framenum_range[0] * (vid_max_framenum - vid_min_framenum)))
    max_framenum = vid_min_framenum + int(round(framenum_range[1] * (vid_max_framenum - vid_min_framenum)))
    _print('Filtering frames in range {}-{} according to framenum range {}: {}'.format(
        min(framenums), max(framenums),
        framenum_range, (min_framenum, max_framenum)))
    keep_im_idxs = [i for i, f in enumerate(framenums) if f >= min_framenum and f <= max_framenum]
    _print('Keeping {} of {} frames'.format(len(keep_im_idxs), len(framenums)))
    return keep_im_idxs


def filter_by_framenum_deltas(framenums, frame_delta_range, vid_max_framenum=None, _print=print):
    T = len(framenums)
    # make a T x T matrix of the diff between each id and the id on the diagonal
    img_deltas = np.abs(
        np.tile(np.reshape(framenums, (1, T)), (T, 1)) - np.tile(np.reshape(framenums, (T, 1)), (1, T)))
    _print('Filtering according to delta range {}'.format(frame_delta_range))
    _print(img_deltas)
    # read in any frames where there is some delta that is within our frame delta range (doesn't have to be adjacent)
    keep_im_idxs = np.where(
        np.any(
            (img_deltas >= frame_delta_range[0]) * (img_deltas <= frame_delta_range[1]),
            axis=1))[0].tolist()
    _print('Keeping {} of {} frames'.format(len(keep_im_idxs), T))

    # remove duplicates and sort
    keep_im_idxs = list(sorted(list(set(keep_im_idxs))))

    return keep_im_idxs

ORIG_FPS = 30
def filter_by_fps(framenums, fps, vid_max_framenum=None, _print=print):
    T = len(framenums)
    _print('Filtering according to fps {}'.format(fps))

    step_size = int(ORIG_FPS / fps)

    # get the closest possible frames in the dataset to the optimal fps
    keep_framenums = np.arange(min(framenums), max(framenums), step=step_size)

    # now figure out the nearest frame idxs to each framenum
    keep_idxs = []
    dists_to_fns = []
    for fn in keep_framenums:
        dists_to_fn = np.abs(framenums - fn)
        keep_idxs.append(int(np.argmin(dists_to_fn)))
        dists_to_fns.append(min(dists_to_fn))

    _print('Keeping {} of {} frames in range {}-{}, with max delta of {} from ideal spacing'.format(
        len(keep_idxs), len(framenums),
        min(framenums), max(framenums), max(dists_to_fns)
    ))
    return keep_idxs


def enumerate_nonadj_seqs(framenums, attns=None, frame_delta_range=None,
                          min_attn_area=None, max_attn_area=None, seq_len=1, max_seqs_per_start=5,
                          has_valid_deltas=None, greedy=False, vid_name=None):
    '''
    Enumerates all sequences of length seq_len, where all frames in the sequence satisfy certain
    adjacency criteria (e.g. framenum difference, number of pixels changed)

    :param framenums:
    :param attns:
    :param frame_delta_range:
    :param min_attn_area:
    :param max_attn_area:
    :param seq_len:
    :param total_attn_area:
    :return:
    '''
    # total number of frames in the video
    T = len(framenums)
    valid_next_frame_idxs, has_valid_deltas = _find_valid_nonadj_frames(attns, frame_delta_range, framenums,
                                                                           has_valid_deltas, max_attn_area,
                                                                           min_attn_area, vid_name=vid_name)
    all_additional_seqs = []
    # need to be careful with indexing here, since valid_frame_idxs are not necessarily continuous
    for start_frame_idx in range(T - seq_len + 1):
        all_additional_seqs += _find_seqs_starting_with(start_frame_idx, seq_len, valid_next_frame_idxs, greedy=greedy,
                                                        max_n_seqs=max_seqs_per_start)
    return all_additional_seqs, has_valid_deltas


def _find_valid_nonadj_frames(attns, frame_delta_range, framenums, has_valid_deltas=None,
                              max_attn_area=None, min_attn_area=None, min_attn_areas_to_remove=None, vid_name=None):
    framenums = framenums.astype(np.uint16)
    T = len(framenums)
    assert len(list(set(framenums))) == len(framenums)  # framenums should be unique
    if has_valid_deltas is None and frame_delta_range is not None:
        # make a T x T matrix of the diff between each id and the id on the diagonal
        framenum_deltas = np.abs(
            np.tile(np.reshape(framenums, (1, T)), (T, 1)) \
            - np.tile(np.reshape(framenums, (T, 1)), (1, T))
        )

        if frame_delta_range is not None:
            # for each frame t, keep track of which following frames are valid (e.g. t+2, t+3)
            has_valid_deltas = (framenum_deltas >= frame_delta_range[0]) * (
                    framenum_deltas <= frame_delta_range[1]).astype(bool)
            del framenum_deltas
    elif has_valid_deltas is not None:
        assert has_valid_deltas.shape[0] == T
        assert has_valid_deltas.shape[1] == T
    elif frame_delta_range is None:
        has_valid_deltas = np.ones((T, T), dtype=np.bool)
    else:
        raise NotImplemented

    if min_attn_area is not None or max_attn_area is not None:
        # Dont consider frames that have zero attention between them -- no point in including these
        # since even if they are non-adjacent, they won't add any information to our training set

        # first, create an attention matrix
        attn_deltas = _compute_attn_area_matrix(attns)
        if min_attn_area is not None:
            has_valid_deltas *= (attn_deltas >= min_attn_area).astype(bool)

        if max_attn_area is not None:
            has_valid_deltas *= (attn_deltas <= max_attn_area).astype(bool)
        # also zero out any columns where the frame has 0 attn wrt its previous frame --
        # that means it is identical to the previous frame so there is no point in including it
        for t in range(1, T):
            # check for min_attn_area here instead? so we don't get trivial differences
            if attn_deltas[t - 1, t] < (min_attn_areas_to_remove if min_attn_areas_to_remove is not None else min_attn_area):
                # zero out the entire row and col -- basically never use this index at the start or end of a sequence
                has_valid_deltas[:, t] = 0
                has_valid_deltas[t, :] = 0

    # convert TxT binary matrix to a T-length list, where each element is a list of valid next frames
    valid_next_frame_idxs = [None] * T
    for t in range(T):
        r = has_valid_deltas[t, :]
        if np.any(r):  # if there are valid framenum deltas for this starting frame,
            # only take "next" frames that are actually after the current frame
            valid_next_frame_idxs[t] = [next_idx for next_idx in np.where(r)[0].tolist() if next_idx > t]
            assert np.all(np.asarray(valid_next_frame_idxs[t]) > t)

    return valid_next_frame_idxs, has_valid_deltas


def _find_seqs_starting_with(start_frame_idx, seq_len, valid_next_frame_idxs, greedy=False, max_n_seqs=None):
    curr_additional_seqs = [[start_frame_idx]]

    seqs = []

    # keep adding on valid frames until we reach the desired seq len
    curr_seq_len = 1
    while (curr_seq_len < seq_len):
        # keep track of seqs of the next length
        next_additional_seqs = []

        # pop off each partial sequence
        for partial_seq in curr_additional_seqs:
            partial_seq_end_idx = partial_seq[-1]

            # for each starting frame t, see if there are any valid next steps to take
            if valid_next_frame_idxs[partial_seq_end_idx] is None:
                continue

            next_frame_idxs = valid_next_frame_idxs[partial_seq_end_idx]  # a list of valid next frame idxs

            for nfi in next_frame_idxs:
                next_additional_seqs.append(partial_seq + [nfi])

                if greedy:  # just take the very first valid next frame
                    break

        if len(next_additional_seqs) > 0:
            curr_additional_seqs = next_additional_seqs
            curr_seq_len += 1
        else:
            break

    curr_additional_seqs = [seq for seq in curr_additional_seqs if len(seq) == seq_len]
    if max_n_seqs is not None and len(curr_additional_seqs) > max_n_seqs:
        keep_idxs = np.random.choice(len(curr_additional_seqs), max_n_seqs, replace=False)
        curr_additional_seqs = [curr_additional_seqs[ki] for ki in keep_idxs]
    # the actual sequence should be frame indices (into the current vid frames),
    # so make sure we convert the indices in valid_frame_idxs to actual frame idxs
    seqs += curr_additional_seqs
    return seqs

def _compute_attn_area_matrix(attns):
    T = attns.shape[0]
    if len(attns.shape) == 2: 
        curr_vid_attn_areas = attns.astype(np.uint16)
    else:
        # actual maps, we havent computed attns already
        curr_vid_attn_areas = np.sum(attns.astype(np.uint16), axis=tuple(list(range(1, len(attns.shape))))).astype(np.uint16) # sum over all axes except the first (which is time)
    
    # attn[t] represents diff between frame[t-1] and frame [t]. So to make a matrix of attns...

    # NOTE: we ignore attn[0] in this matrix, it doesn't show up. we assume it's not important
    attn_deltas = np.ones((T, T), dtype=np.uint16) * np.nan  # by default, make everything have a valid attn delta

    for t in range(T):
        if t == 0: # special case for blank-to-frame0 attention
            attn_deltas[t, t] = curr_vid_attn_areas[0]
        else:
            attn_deltas[t, t] = 0  # zero attn with self

        if t < T - 1:
            attn_deltas[t, t + 1] = curr_vid_attn_areas[t + 1]

            attn_deltas[:t, t + 1] = attn_deltas[:t, t] + attn_deltas[t, t + 1]
            # we can cheat a little by computing the attention between t, t+2 as the sum of t to t+1 and t+1 to t+2
            #for prev_t in reversed(range(t)):  # go all the way back to the first row
            #    # the first term should have been filled out in previous iterations, as we move down the matrix
            #    attn_deltas[prev_t, t + 1] = attn_deltas[prev_t, t] + attn_deltas[t, t + 1]
    return attn_deltas

def _test_compute_attn_area_matrix():
    frame_shape = (5, 5, 1)

    # attn from frame blank to 0 is 4
    attn_0 = np.zeros(frame_shape)
    attn_0[2:4, 2:4] = 1

    # attn from frame 0 to 1 is 4
    attn_1 = np.zeros(frame_shape)
    attn_1[2:4, 3:] = 1

    # attn from frame 1 to 2 is 0. but from 0 to 2 it should be 4
    attn_2 = np.zeros(attn_1.shape)

    attn_frames = [attn_0, attn_1, attn_2]
    T = len(attn_frames)
    attns = np.concatenate([f[np.newaxis] for f in attn_frames])

    attn_deltas = _compute_attn_area_matrix(attns)
    print(attn_deltas)
    for t in range(1, T): # diagonal should be all 0s
        assert attn_deltas[t, t] == 0
    assert attn_deltas[0, 1] == np.sum(attn_1)
    assert attn_deltas[1, 2] == 0
    assert attn_deltas[0, 2] == attn_deltas[0, 1]
    print('UNIT TEST: _test_compute_attn_area_matrix produced expected matrix -- PASSED')

def _test_enumerate_nonadj_seqs():
    framenums = [0, 1, 2, 5]
    computed_additional_seq_idxs, _ = enumerate_nonadj_seqs(framenums, seq_len=2, frame_delta_range=(2, 4))
    computed_additional_seqs = []
    for idxs_seq in computed_additional_seq_idxs:
        computed_additional_seqs.append([framenums[idx] for idx in idxs_seq])
    print(computed_additional_seqs)
    expected_seqs = [
        [0, 2],
        [1, 5],
        [2, 5]
    ]
    assert np.all([es in computed_additional_seqs for es in expected_seqs])
    assert len(computed_additional_seqs) == len(expected_seqs)
    print('enumerate_nonadj_seqs with framenum filters produced expected seqs {}: {} -- PASSED'.format(
        expected_seqs, computed_additional_seqs
    ))

    computed_additional_seq_idxs, _ = enumerate_nonadj_seqs(framenums, seq_len=2, frame_delta_range=(0, 4))
    computed_additional_seqs = []
    for idxs_seq in computed_additional_seq_idxs:
        computed_additional_seqs.append([framenums[idx] for idx in idxs_seq])
    print(computed_additional_seqs)
    expected_seqs = [
        [0, 1],
        [1, 2],
        [0, 2],
        [1, 5],
        [2, 5]
    ]
    assert np.all([es in computed_additional_seqs for es in expected_seqs])
    assert len(computed_additional_seqs) == len(expected_seqs)
    print('enumerate_nonadj_seqs with framenum filters produced expected seqs {}: {} -- PASSED'.format(
        expected_seqs, computed_additional_seqs
    ))

def _test_enumerate_nonadj_seqs_with_attns():
    frame_shape = (5, 5, 1)

    # attn from frame blank to 0 is 4
    attn_0_1 = np.zeros(frame_shape)
    attn_0_1[2:4, 2:3] = 1 # area: 3

    # attn from frame 0 to 1 is 4
    attn_1_2 = np.zeros(frame_shape)
    attn_1_2[2:4, 3:] = 1 # area: 4

    # attn from frame 1 to 2 is 0. but from 0 to 2 it should be 4
    attn_2_3 = np.zeros(attn_1_2.shape) # area: 0

    attn_frames = [np.zeros(frame_shape), attn_0_1, attn_1_2, attn_2_3]
    attns = np.concatenate([f[np.newaxis] for f in attn_frames])
    framenums = [0, 1, 2, 5]

    computed_additional_seq_idxs, _ = enumerate_nonadj_seqs(framenums, attns=attns, seq_len=2, frame_delta_range=(0, 4),
                                                         min_attn_area=1, max_attn_area=3)
    computed_additional_seqs = []
    for idxs_seq in computed_additional_seq_idxs:
        computed_additional_seqs.append([framenums[idx] for idx in idxs_seq])
    print(computed_additional_seqs)
    expected_seqs = [
        [0, 1],
    ]
    assert np.all([es in computed_additional_seqs for es in expected_seqs])
    assert len(computed_additional_seqs) == len(expected_seqs)
    print('enumerate_nonadj_seqs with framenum AND attn filters produced expected seqs {}: {} -- PASSED'.format(
        expected_seqs, computed_additional_seqs
    ))

    computed_additional_seq_idxs, has_valid_deltas = enumerate_nonadj_seqs(framenums, attns=attns, seq_len=2, frame_delta_range=(0, 4),
                                                         min_attn_area=0, max_attn_area=3)
    computed_additional_seqs = []
    for idxs_seq in computed_additional_seq_idxs:
        computed_additional_seqs.append([framenums[idx] for idx in idxs_seq])
    print(has_valid_deltas)
    print(computed_additional_seqs)

    expected_seqs = [
        [0, 1],
        #[1, 5],
        [2, 5]
    ]
    assert np.all([es in computed_additional_seqs for es in expected_seqs])
    assert len(computed_additional_seqs) == len(expected_seqs)
    print('enumerate_nonadj_seqs with framenum AND attn filters produced expected seqs {}: {} -- PASSED'.format(
        expected_seqs, computed_additional_seqs
    ))

    computed_additional_seq_idxs, has_valid_deltas = enumerate_nonadj_seqs(framenums, attns=attns, seq_len=2, frame_delta_range=(0, 4),
                                                         min_attn_area=0, max_attn_area=4)
    computed_additional_seqs = []
    for idxs_seq in computed_additional_seq_idxs:
        computed_additional_seqs.append([framenums[idx] for idx in idxs_seq])
    print(has_valid_deltas)
    print(computed_additional_seqs)
    expected_seqs = [
        [0, 1],
        [1, 2],
        [1, 5],
        [2, 5]
    ]
    assert np.all([es in computed_additional_seqs for es in expected_seqs])
    assert len(computed_additional_seqs) == len(expected_seqs)
    print('enumerate_nonadj_seqs with framenum AND attn filters produced expected seqs {}: {} -- PASSED'.format(
        expected_seqs, computed_additional_seqs
    ))

def apply_filter(framenums, seqs_frame_idxs, firstlast_idxs, vid_attns,
                 filter_fn, filter_name, filter_info, _print):
    keep_seq_idxs = filter_fn(framenums, seqs_frame_idxs, vid_attns)
    removed_seq_idxs = [si for si, seq in enumerate(seqs_frame_idxs) if si not in keep_seq_idxs]

    _print('Removed {} of {} seqs due to {} {}: {}...'.format(
        len(seqs_frame_idxs) - len(keep_seq_idxs), len(seqs_frame_idxs), filter_name, filter_info,
        [seqs_frame_idxs[rsi] for rsi in removed_seq_idxs[:min(len(removed_seq_idxs), 3)]]
    ))
    # do all filtering based on seq indices so that we can keep a corresponding list of
    # attn sequences
    seqs_frame_idxs = [seqs_frame_idxs[i] for i in keep_seq_idxs]
    firstlast_idxs = [firstlast_idxs[i] for i in keep_seq_idxs]

    return seqs_frame_idxs, firstlast_idxs

if __name__ == '__main__':
    _test_enumerate_nonadj_seqs()
    _test_compute_attn_area_matrix()
    _test_enumerate_nonadj_seqs_with_attns()
