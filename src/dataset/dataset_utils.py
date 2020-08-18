import copy
import os

import cv2
import numpy as np
import pickle

import src.utils.vis_utils
from src.utils import utils

VID_SEGMENT_SUFFIXES = ['seg', 'piece', 'crop']
def vid_name_to_vid_base_name(vid_name):
    for suffix in VID_SEGMENT_SUFFIXES:
        vid_name = vid_name.split('_{}'.format(suffix))[0]
    return vid_name

VID_PREPROC_SUFFIXES = ['crop']
def vid_name_to_vid_piece_name(vid_name): # removes any suffixes added to the preprocessing
    for suffix in VID_PREPROC_SUFFIXES:
        vid_name = vid_name.split('_{}'.format(suffix))[0]
    return vid_name


def frames_to_video(frames, out_file, scale_factor=None, framerate=30, interp=cv2.INTER_LINEAR):
    # assumes frames is a list of h x w x 3 frames
    T = len(frames)

    if scale_factor is not None:
        frames = np.concatenate([f[np.newaxis] for f in frames])
        print('Resizing frames by factor {}...'.format(scale_factor))
        frames = utils.resize_batch(frames, scale_factor, interp=interp).astype(np.uint8)

    vw = cv2.VideoWriter(out_file,
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         framerate, (frames[0].shape[1], frames[0].shape[0]))

    for t, frame in enumerate(frames):
        if t % 100 == 0 or t == T - 1:
            print('Writing frame {} of {}'.format(t + 1, T))
        vw.write(frame)
    vw.release()

def extract_frames_from_vids(
        ref_thumbs_dir, vids_dir,
        target_shape,
        vid_names=None,
        preprocessors=[],
        framenum_filters=[]
):
    vids = []

    if vid_names is None:
        vid_names = os.listdir(ref_thumbs_dir)

    # extract the full-size frames that match the thumbs
    for vid_name in list(reversed(vid_names)):
        curr_thumbs_dir = os.path.join(ref_thumbs_dir, vid_name)
        curr_thumbs = os.listdir(curr_thumbs_dir)
        framenums_to_extract = utils.filenames_to_im_ids(curr_thumbs)

        framenums_to_extract, curr_thumbs = zip(*sorted(zip(framenums_to_extract, curr_thumbs)))

        # find the corresponding video file
        if '_piece' in vid_name:
            vid_file = os.path.join(vids_dir, '{}.mp4'.format(vid_name.split('_piece')[0]))
        else:
            vid_file = os.path.join(vids_dir, '{}.mp4'.format(vid_name))

        if not os.path.isfile(vid_file):
            print('Could not find video {}! Skipping...'.format(vid_file))
            continue
        print('Opening vid file {}, matching frames in {}'.format(vid_file, curr_thumbs_dir))

       # open up the video file and extract frames at the desired
        # now open the video file we just downloaded, and extract the frames from it
        try:
            #reader = skvio.FFmpegReader(vid_file)
            reader = cv2.VideoCapture(vid_file)
            vid_max_framenum = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
        except:
            print('Could not open vid {}! Skipping...'.format(vid_file))
            continue

        if framenum_filters is not None:
            for framenum_filter in framenum_filters:
                keep_idxs = framenum_filter(framenums_to_extract, vid_max_framenum=vid_max_framenum)
                framenums_to_extract = [framenums_to_extract[ki] for ki in keep_idxs]

        vid_framenums = []
        frames_raw = []
        print('Reading {} frames from vid {}'.format(len(framenums_to_extract), vid_name))

        i = -1
        while reader.isOpened():
        #for i, frame in enumerate(reader.nextFrame()):
            # step through video frames until we get to one we want to extract
            _, frame = reader.read()
            i += 1

            if i in framenums_to_extract:
                #frame = frame[..., [2, 1, 0]]

                frames_raw.append(frame)
                vid_framenums.append(i)

            if i > max(framenums_to_extract):
                break
        reader.release()

        frames = np.concatenate([frame[np.newaxis] for frame in frames_raw], axis=0)
        del frames_raw

        for preprocessor in preprocessors:
            frames = preprocessor(frames, vid_name=vid_name)
        print(frames.nbytes / float(10e6))

        vids.append({
            'frames': frames,
            'vid_name': vid_name,
            'framenums': vid_framenums,
        })
    return vids


def frames_to_dir(frames, out_dir, scale_factor=None, interp=cv2.INTER_LINEAR):
    # assumes frames is a list of h x w x 3 frames
    T = len(frames)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    if scale_factor is not None:
        frames = np.concatenate([f[np.newaxis] for f in frames])
        print('Resizing frames by factor {}...'.format(scale_factor))
        frames = utils.resize_batch(frames, scale_factor, interp=interp)
    print(out_dir)
    for t, frame in enumerate(frames):
        if t % 100 == 0 or t == T - 1:
            print('Writing frame {} of {}'.format(t + 1, T))
        print(frame.shape)
        cv2.imwrite(os.path.join(out_dir, '{}.png'.format(t)), frame)


def dir_to_video(in_dir, out_dir, do_display_framenum=False, sf=None, framerate=30):
    im_files = [os.path.join(in_dir, f) for f in os.listdir(in_dir) if f.endswith('.jpg') or f.endswith('.png')]
    im_ids = utils.filenames_to_im_ids(im_files)

    im_ids, im_files = zip(*sorted(zip(im_ids, im_files)))

    vid_name = os.path.basename(in_dir)


    frames = []
    for t, im_file in enumerate(im_files):

        im = cv2.imread(im_file)
        frames.append(im)
    frames = np.concatenate([f[np.newaxis] for f in frames])

    if sf is not None:
        print('Resizing frames by factor {}...'.format(sf))
        frames = utils.resize_batch(frames, sf, interp=cv2.INTER_CUBIC)
    if do_display_framenum:
        print('Labeling frame nums...')
        frames = frames / 255.

        frames = [src.utils.vis_utils.label_ims(
            frame[np.newaxis], im_ids[i],
            display_h=frames.shape[1]) for i, frame in enumerate(frames)]

    vw = cv2.VideoWriter(os.path.join(out_dir, '{}.avi'.format(vid_name)),
                         cv2.VideoWriter_fourcc(*'XVID'),
                         framerate, (frames[0].shape[1], frames[0].shape[0]))

    for t, frame in enumerate(frames):
        if t % 100 == 0:
            print('Writing frame {} of {}'.format(t, len(im_files)))
        vw.write(frame)
    vw.release()


def load_data_from_pkl(
        pkl_file,
        crop_preprocessors=[],
        framenum_filters=[]
):
    print('Attempting to load data from file {}'.format(pkl_file))

    vid_name = os.path.splitext(pkl_file)[0]
    with open(pkl_file, 'rb') as f:
        vid_data = pickle.load(f)
    frames = vid_data['frames']
    framenums = vid_data['framenums']

    # create a string to identify each frame
    frame_ids = [f'{vid_name}_frame{fn}' for fn in framenums]

    max_framenum = max(framenums)
    if framenum_filters is not None:
        for framenum_filter in framenum_filters:
            keep_idxs = framenum_filter(framenums, vid_max_framenum=max_framenum)
            if len(keep_idxs) == 0:
                return None
            frames = frames[keep_idxs]
            framenums = [framenums[ki] for ki in keep_idxs]
            frame_ids = [frame_ids[ki] for ki in keep_idxs]

    out_vids = []
    # now, take care of preprocessors that create new videos
    if crop_preprocessors is not None:
        assert len(crop_preprocessors) == 1  # we only support one type of cropping at a time for now
        frames_list, frame_ids_list = crop_preprocessors[0](
            frames, frame_ids=frame_ids, vid_name=vid_name)

        for i, frames in enumerate(frames_list):
            assert frames.shape[0] == len(frame_ids_list[i])

            split_vid_name = '{}_crop{}'.format(vid_name, i)
            out_vids.append({
                'frames': frames,
                'vid_name': split_vid_name,
                'frame_ids': frame_ids_list[i]
            })
    else:
        ims = np.concatenate([im[np.newaxis] for im in frames], axis=0)
        out_vids.append({
            'frames': ims,
            'vid_name': vid_name,
            'frame_ids': frame_ids
        })

    return out_vids



def combine_dataset_names(datasets):
    # initialize with the first dataset
    combined_dataset = copy.deepcopy(datasets[0])
    combined_dataset.display_name = '_'.join([
        shorten_dataset_name(ds.params['dataset']) + \
        shorten_dataset_name(ds.display_name.replace(ds.frames_root, ''))
        for ds in datasets])
    return combined_dataset


def combine_dataset_vids(datasets,
                         combined_dataset,
                         do_filter_by_true_starts=False, do_filter_by_affine_breaks=False,
                         do_filter_by_excluded_vids=False,
                         load_n=None, _print=print):
    '''
    Combines the videos of multiple datasets, while maintaining the train/valid split from each dataset

    :param datasets:
    :param combined_datasets

    :param load_n:
    :param _print:
    :return:
    '''
    all_train_vid_names = []

    if len(datasets) == 1: # only one dataset, no need to combine
        ds = datasets[0]
        ds._print = _print
        _print('Loading vids from only dataset: {}'.format(ds.display_name))
        ds.load_dataset_frames(load_n=load_n)
        return ds

    _print('Combining {} datasets'.format(len(datasets)))

    for dsi, ds in enumerate(datasets):
        _print('Checking dataset {}: {}...'.format(dsi, ds.display_name))
        ds._print = _print

        # load videos and split into train and validation
        ds.load_dataset_frames(load_n=load_n)

        if do_filter_by_true_starts and 'true_starts_file' in ds.params and ds.params['true_starts_file'] is not None:
            with open(ds.params['true_starts_file'], 'r') as f:
                true_start_vidnames = f.readlines()
            true_start_vidnames = [v.strip() for v in true_start_vidnames]
            _print('Filtering by {} true start vidnames'.format(len(true_start_vidnames)))
        else:
            true_start_vidnames = None

        if do_filter_by_affine_breaks and 'affine_breaks_file' in ds.params and ds.params['affine_breaks_file'] is not None:
            with open(ds.params['affine_breaks_file'], 'r') as f:
                affine_breaks_files = f.readlines()
            affine_breaks_vidnames = [l.split('_frame')[0] for l in affine_breaks_files]
            _print('Filtering out {} affinely moving vidnames'.format(len(affine_breaks_vidnames)))
        else:
            affine_breaks_vidnames = None

        if do_filter_by_excluded_vids and 'exclude_vids_file' in ds.params and ds.params['exclude_vids_file'] is not None:
            with open(ds.params['exclude_vids_file'], 'r') as f:
                exclude_vids = f.readlines()
            exclude_vidnames = [f.strip() for f in exclude_vids]
        else:
            exclude_vidnames = None
 
        # if a video is in any of the training sets, then we will include it in the final training set
        all_train_vid_names += [vd['vid_name'].split('_seg')[0] for vd in ds.vids_train]

        # combine all of the video data (frames, im_files list, attention maps) from all datasets
        if dsi == 0 and len(datasets) > 1:
            vids_data_combined = [vd for vd in ds.vids_data \
                if (true_start_vidnames is None or (vid_name_to_vid_piece_name(vd['vid_name']) in true_start_vidnames)) \
                and (affine_breaks_vidnames is None or (vid_name_to_vid_base_name(vd['vid_name']) not in affine_breaks_vidnames))\
                and (exclude_vidnames is None or (vid_name_to_vid_base_name(vd['vid_name']) not in exclude_vidnames))
            ] # new list, same elements?
        else:
            vid_names_combined = [vd['vid_name'] for vd in vids_data_combined]

            for vi, vd in enumerate(ds.vids_data):
                if true_start_vidnames is not None and vid_name_to_vid_piece_name(vd['vid_name']) not in true_start_vidnames:
                    # skip this video if we are filtering to only include videos that actually start at teh start of the painting
                    continue

                if affine_breaks_vidnames is not None and vid_name_to_vid_base_name(vd['vid_name']) in affine_breaks_vidnames:
                    # skip this video if we are filtering by dramatic affine breaks
                    continue

                if exclude_vidnames is not None and vid_name_to_vid_base_name(vd['vid_name']) in exclude_vidnames:
                    continue

                if vd['vid_name'] in vid_names_combined:
                    existing_vid_idx = vid_names_combined.index(vd['vid_name'])
                    vids_data_combined[existing_vid_idx] = vd
                else:
                    vids_data_combined.append(vd)

    vid_names = [vd['vid_name'].split('_seg')[0] for vd in vids_data_combined]

    combined_dataset.vids_data = vids_data_combined
    combined_dataset.vids_train = [vids_data_combined[i] for i in range(len(vids_data_combined))
                                   if vids_data_combined[i]['vid_name'].split('_seg')[0] in all_train_vid_names]

    combined_dataset.vids_valid = [vids_data_combined[i] for i in range(len(vids_data_combined))
                                   if vids_data_combined[i]['vid_name'].split('_seg')[0] not in all_train_vid_names]
    return combined_dataset


def _test_prune_unused_frames():
    # use ids instead of images in this test set for easy reference
    vid_data_list = [
        {
            'frames': np.asarray(list(range(6))),
            'attn': np.asarray(list(range(6))),
        },
        {
            'frames': np.asarray(list(range(6))),
            'attn': np.asarray(list(range(6))),
        }
    ]

    seq_infos = [
        (0, [1, 3], [-1, -1], [1, 3]),
        (1, [2, 4], [4, 4], None),
    ]

    new_vid_data, new_seq_infos = _prune_unused_frames(vid_data_list, seq_infos)
    print(new_vid_data)
    # should have kept frames in sequences, as well as last frame
    assert len(new_vid_data[0]['frames']) == 2 + 1
    assert len(new_vid_data[1]['frames']) == 2 + 1

    assert np.all([idx in new_vid_data[0]['frames'] for idx in [1, 3, 5]])
    assert np.all([idx in new_vid_data[1]['frames'] for idx in [2, 4, 5]])

    # should have removed all other frames
    assert np.all(np.asarray(new_seq_infos[0][1]) == np.asarray([0, 1]))


if __name__ == '__main__':
    _test_prune_unused_frames()


def shorten_dataset_name(dataset_name):
    return dataset_name.replace('_', '-')\
        .replace('-nohand-png', '')\
        .replace('watercolors', 'wc')

def load_dataset(dataset_key, override_params=None, load_n=None):
    from config.synth_dataset_configs import synth_data_configs
    from config.watercolors_configs import watercolors_data_configs
    from config.procreate_configs import digital_data_configs
 
    from dataset import datasets as timelapse_datasets

    if 'sp-' in dataset_key or 'synth' in dataset_key:
        data_params = synth_data_configs[dataset_key]
    elif 'pc-' in dataset_key:
        data_params = digital_data_configs[dataset_key]
    else:
        data_params = watercolors_data_configs[dataset_key]
        
    if override_params is not None:
        for k, v in override_params.items():
            data_params[k] = v

    print(data_params)
    ds = timelapse_datasets.create_dataset(data_params)
    ds.load_dataset_frames(load_n=load_n)
    return ds


def _extract_approximate_fps_vids(
        vids_data_list, target_fps, target_seq_len, attn_params, min_attn_area,
        max_attn_area=None, _print=print):
    # extract videos with approximate fps, and according to any attention filtering params
    from dataset import frame_filter_utils
    import utils

    for vi, vd in enumerate(vids_data_list):
        curr_framenums = file_utils.filenames_to_im_ids(vd['im_files'])

        keep_frame_idxs = frame_filter_utils.filter_by_fps(curr_framenums, fps=target_fps)
        keep_frame_idxs.append(len(curr_framenums) - 1)  # include the last frame

        curr_attns = utils.compute_attention_maps(
            vd['frames'][keep_frame_idxs],
            **attn_params
        )
        # pad out the first index with zeros so indexing is more consistent with what we have elsewhere
        # attention at t is the delta between t-1 and t
        curr_attns = np.concatenate([np.zeros(curr_attns[[0]].shape), curr_attns], axis=0)

        curr_attns_areas = np.sum(curr_attns, axis=tuple(list(range(1, len(curr_attns.shape)))))

        if max_attn_area is None:
            print('Filtering by min attn area {}'.format(min_attn_area))
            # only filter by min attn area
            # we don't need to consider both ends of the pair because the assumption is that the frames are identical,
            # so just remove one of them 
            valid_attn_frame_idxs = [idx for i, idx in enumerate(keep_frame_idxs) if curr_attns_areas[i] >= min_attn_area]
        elif max_attn_area is not None and min_attn_area is not None:
            print('Filtering by attn area [{},{}]'.format(min_attn_area, max_attn_area))
            # only filter by both attn areas
            # we don't need to consider both ends of the pair because the assumption is that the frames are identical,
            # so just remove one of them 
            valid_attn_frame_idxs = [idx for i, idx in enumerate(keep_frame_idxs) if curr_attns_areas[i] >= min_attn_area and curr_attns_areas[i] <= max_attn_area]
        else:
            raise NotImplementedError

        keep_frame_idxs = [0, len(curr_framenums) - 1] + valid_attn_frame_idxs

        # uniquify and sort
        keep_frame_idxs = list(sorted(list(set(keep_frame_idxs))))
        _print('Keeping {} of {} frames from {}, pad/cropping to {}'.format(
            len(keep_frame_idxs),
            len(curr_framenums),
            vd['vid_name'],
            target_seq_len))

        if len(keep_frame_idxs) > target_seq_len:
            keep_frame_idxs = keep_frame_idxs[:target_seq_len]
        elif len(keep_frame_idxs) < target_seq_len:
            # tile the last frame
            keep_frame_idxs = keep_frame_idxs + [keep_frame_idxs[-1]] * (
                        target_seq_len - len(keep_frame_idxs))

        keep_frames = vids_data_list[vi]['frames'][keep_frame_idxs]
        keep_attns = np.concatenate([
            np.zeros(curr_attns[[0]].shape), 
            utils.compute_attention_maps(keep_frames,
                                         **attn_params)], axis=0)
        keep_im_files = [vids_data_list[vi]['im_files'][i] for i in keep_frame_idxs]
        vids_data_list[vi]['frames'] = keep_frames
        vids_data_list[vi]['attn'] = keep_attns

        vids_data_list[vi]['im_files'] = keep_im_files


def _prune_unused_frames(vid_data_list, seq_infos, _print=print):
    # initialize used frame indices with last index always
    used_frame_idxs = [[vd['frames'].shape[0] -1] for vd in vid_data_list]

    # first go through sequences and collect which frames are used
    for seq_info in seq_infos:
        vid_idx, seq_frame_idxs, seq_firstlast_idxs = seq_info

        used_frame_idxs[vid_idx] += [fi for fi in seq_frame_idxs if fi is not None]
        used_frame_idxs[vid_idx] += [fi for fi in seq_firstlast_idxs if not fi == -1]

    # now update indices in sequences, and delete unused frames
    for vi, curr_vid_used_frame_idxs in enumerate(used_frame_idxs):
        curr_vid_used_frame_idxs = list(sorted(list(set(curr_vid_used_frame_idxs))))

        for si, seq_info in enumerate(seq_infos):
            vid_idx, seq_frame_idxs, seq_firstlast_idxs = seq_info

            if vid_idx == vi:
                # recreate the tuple
                seq_infos[si] = (
                    vid_idx,
                    [curr_vid_used_frame_idxs.index(fi) if fi is not None else None for fi in seq_frame_idxs],
                    [curr_vid_used_frame_idxs.index(fi) if not fi == -1 else -1 for fi in seq_firstlast_idxs],
                )
        orig_n_frames = vid_data_list[vi]['frames'].shape[0]
        orig_size = vid_data_list[vi]['frames'].nbytes
        # remove unused frames
        vid_data_list[vi]['frames'] = vid_data_list[vi]['frames'][curr_vid_used_frame_idxs]
        vid_data_list[vi]['frame_ids'] = [vid_data_list[vi]['frame_ids'][idx] for idx in curr_vid_used_frame_idxs]
        vid_data_list[vi]['was_pruned'] = True # in case we want to make a frame deltas matrix
        _print('Kept {} of {} frames in vid {}, saving {:.1f}MB!'.format(
            len(curr_vid_used_frame_idxs), orig_n_frames, vid_data_list[vi]['vid_name'],
            (orig_size - vid_data_list[vi]['frames'].nbytes) / float(1e6)))
        if vid_data_list[vi]['attn'] is not None:
            # just keep track of the attentions that we need for our frames
            vid_data_list[vi]['attn'] = vid_data_list[vi]['attn'][curr_vid_used_frame_idxs]

    return vid_data_list, seq_infos
