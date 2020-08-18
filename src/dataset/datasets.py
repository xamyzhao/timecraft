import os
import functools
import pickle
import sys
import time

import numpy as np
from src.dataset import dataset_utils, frame_filter_utils, preprocessors

watercolors_root = 'C:\\Datasets'
procreate_root = 'C:\\Datasets\\procreate'
croprect_root = watercolors_root
cache_root = watercolors_root


class WatercolorsDataset(object):
    def __init__(self, params):
        # default parameter settings that might not be in keys
        if 'n_pred_frames' not in params.keys():
            params['n_pred_frames'] = None

        if 'normalize_frames' not in params.keys():
            params['normalize_frames'] = False

        if 'frame_delta_range' not in params.keys():
            params['frame_delta_range'] = None

        if 'n_prev_frames' not in params.keys():
            params['n_prev_frames'] = 1

        if 'do_stretch' not in params.keys():
            params['do_stretch'] = False

        if 'endpoints' not in params.keys():
            params['endpoints'] = None

        if 'do_use_segment_end' not in params.keys():
            params['do_use_segment_end'] = False

        if 'frame_shifts_dir' not in params.keys():
            params['frame_shifts_dir'] = None

        if 'target_fps' not in params:
            params['target_fps'] = None

        if 'anno_file_frame_shifts' in params.keys() \
                and params['anno_file_frame_shifts'] is not None:
            # TODO: has_affine.txt is perfectly aligned to the pair of frames where the break occurs,
            # while affine_breaks.txt is manually annotated and both frames should be considered to be bad
            if not type(params['anno_file_frame_shifts']) == list:
                abfs = [params['anno_file_frame_shifts']]
            else:
                abfs = params['anno_file_frame_shifts']

            self.exclude_frame_files = []
            for abf in abfs:
                with open(abf, 'r') as f:
                    exclude_file_pairs = f.readlines()
                # get rid of vid_name subfolder and .png ext
                self.exclude_frame_files += [os.path.splitext(os.path.basename(f.strip()))[0] for fp in exclude_file_pairs for f in fp.split(',')]
            self.exclude_frame_files = sorted(list(set(self.exclude_frame_files)))
            self.exclude_frame_vidnames = [f.split('_frame')[0] for f in self.exclude_frame_files]
        else:
            params['affine_breaks_file'] = None
            self.exclude_frame_files = None
            self.exclude_frame_vidnames = None

        if 'do_filter_by_shifted' not in params.keys():
            params['do_filter_by_shifted'] = False

        # sequence filtering params
        if 'min_attn_area' not in params['sequence_params'].keys():
            params['sequence_params']['min_attn_area'] = None

        if 'max_attn_area' not in params['sequence_params'].keys():
            params['sequence_params']['max_attn_area'] = None

        if 'min_good_seg_len' not in params['sequence_params'].keys():
            params['sequence_params']['min_good_seg_len'] = None

        if 'exclude_vids_file' in params.keys() and params['exclude_vids_file'] is not None:
            with open(params['exclude_vids_file'], 'r') as f:
                exclude_vids = f.readlines()
            self.exclude_vidnames = [f.split(',')[0] for f in exclude_vids]
        else:
            self.exclude_vidnames = None

        if 'load_manual_attn_from_dir' not in params.keys():
            params['load_manual_attn_from_dir'] = None

        self.params = params

        self.logger = None

        self.create_display_name()

    def _print(self, msg):
        if self.logger is not None:
            self.logger.debug(msg)
        else:
            print(msg)

    def create_display_name(self, display_name_base=None):
        display_name = dataset_utils.shorten_dataset_name(self.params['dataset'])

        if self.params['do_stretch']:
            display_name += '-ims{}-{}'.format(
                self.params['im_shape'][0],
                self.params['im_shape'][1]
            )

        if 'scale_factor' in self.params and not self.params['scale_factor'] == 1:
            display_name += '-sf{}'.format(self.params['scale_factor'])

        if 'min_im_shape' in self.params: # scale down only to a minimum size
            display_name += '-min{}-{}'.format(
                self.params['min_im_shape'][0],
                self.params['min_im_shape'][1],
            )

        if self.params['endpoints'] is not None:
            display_name += '-fr{}-{}'.format(
                self.params['endpoints'][0],
                round(self.params['endpoints'][1], 2)
            )
        if self.params['n_pred_frames'] is not None:
            display_name += '-npred{}'.format(self.params['n_pred_frames'])

        if self.params['frame_shifts_dir'] is not None and self.params['do_filter_by_shifted']:
            # check how many files there are and count them
            shift_files = [f for f in os.listdir(self.params['frame_shifts_dir']) if f.endswith('.pkl')]

            # if we are shifting frames by manual annotations
            display_name += '-{}shifts'.format(len(shift_files))

        self.frames_root = display_name

        display_name += '-crop-{}-{}-{}'.format(
            self.params['crop_type'], self.params['pad_to_shape'][0],
            self.params['pad_to_shape'][1]
        )

        # these are params that dont affect what is stored in the pkl
        if self.params['frame_delta_range'] is not None:
            display_name += '-fdr{}-{}'.format(self.params['frame_delta_range'][0], self.params['frame_delta_range'][1])

        if self.params['target_fps'] is not None:
            display_name += '-fps{}'.format(self.params['target_fps'])

        self.display_name = display_name
        return self.display_name


    def load_dataset_frames(self, load_n=None):
        '''
        Load the videos from the dataset, with no processing of the frames

        :param debug:
        :return:
        '''
        self.vids_data = []

        ############# Preprocessing functions ##############
        # take either center crops or tiled crops
        if 'crop_type' in self.params and self.params['crop_type'] == 'center':
            self.crop_preprocessors = [
                functools.partial(
                    preprocessors.crop_center,
                    target_shape=self.params['im_shape'],
                    _print=self._print)]
        elif 'crop_type' in self.params and self.params['crop_type'] == 'olap': # special case since this creates new videos
            self.crop_preprocessors = [
                functools.partial(
                    preprocessors.crop_overlapping,
                    target_shape=self.params['im_shape'],
                    _print=self._print)]

        print('Preprocessing with function {}'.format([p.func for p in self.crop_preprocessors]))

        self.framenum_filters = []

        if self.params['n_pred_frames']:
            self.framenum_filters.append(functools.partial(
                frame_filter_utils.filter_by_n_frames, n_frames=self.params['n_pred_frames'], _print=self._print))

        ############# LOAD VIDEO FRAMES #################
        self.vids_data = self._collect_vids(load_n=load_n)

        self._print('Got {} videos, {}...'.format(
            len(self.vids_data), 
            [vd['vid_name'] for vd in self.vids_data[:min(len(self.vids_data), 5)]]))

        self._print('Frames: range {}, type {}, shape {}...'.format(
            [(np.min(vd['frames']), np.max(vd['frames'])) for vd in self.vids_data[:min(len(self.vids_data), 5)]],
            [vd['frames'].dtype for vd in self.vids_data[:min(len(self.vids_data), 5)]],
            [vd['frames'].shape for vd in self.vids_data[:min(len(self.vids_data), 5)]])),

        # first sort all the vids data by vid name, then shuffle, so we can get a consistent ordering every time
        all_vid_names = list(sorted([dataset_utils.vid_name_to_vid_base_name(vd['vid_name']) for vd in self.vids_data]))

        assert len(all_vid_names) > 0, f'Did not load any videos for dataset {self.display_name}!'
        all_vid_names, self.vids_data = zip(*sorted(zip(all_vid_names, self.vids_data), key=lambda x:x[0]))

        all_vid_names = list(sorted(list(set([
            dataset_utils.vid_name_to_vid_base_name(vd['vid_name']) for vd in self.vids_data]))))
        self._print('All video base names: {}'.format(all_vid_names))

        # need at least to have the previous frame and one more frame for the video to be useful
        self.vids_data = [vd for vd in self.vids_data if vd['frames'].shape[0] > self.params['n_prev_frames']]

        np.random.seed(17)
        np.random.shuffle(self.vids_data)
        # TODO: split consistently by all vid names every time?
        np.random.seed(17)
        np.random.shuffle(all_vid_names)

        n_train = int(round(self.params['percent_train'] * len(all_vid_names)))
        train_vid_names = [all_vid_names[i] for i in range(n_train)]
        valid_vid_names = [all_vid_names[i] for i in range(n_train, len(all_vid_names))]

        if load_n is not None:
            train_vid_names = train_vid_names[:load_n]
            valid_vid_names = valid_vid_names[:load_n]
            self.vids_data = [vd for vd in self.vids_data
                              if dataset_utils.vid_name_to_vid_base_name(vd['vid_name']) in train_vid_names
                              or dataset_utils.vid_name_to_vid_base_name(vd['vid_name']) in valid_vid_names]

        total_n_frames = np.sum([vd['frames'].shape[0] for vd in self.vids_data])

        n_vids = len(self.vids_data)

        self._print('Splitting {} classes into {} train, {} validation'.format(
            len(all_vid_names), len(train_vid_names), len(valid_vid_names)))

        if load_n is None:
            train_idxs = [i for i in range(n_vids)
                          if dataset_utils.vid_name_to_vid_base_name(self.vids_data[i]['vid_name']) in train_vid_names]
            valid_idxs = [i for i in range(n_vids) if dataset_utils.vid_name_to_vid_base_name(self.vids_data[i]['vid_name']) in valid_vid_names]

            # take train idxs from the front, valid idxs from the end since this list is shuffled
            self.vids_train = [self.vids_data[i] for i in train_idxs]
            self.vids_valid = [self.vids_data[i] for i in valid_idxs]
        elif load_n > 1:
            self.vids_train = [self.vids_data[i] for i in range(len(self.vids_data))
                if dataset_utils.vid_name_to_vid_base_name(self.vids_data[i]['vid_name']) in train_vid_names]
            self.vids_valid = [self.vids_data[i] for i in range(len(self.vids_data))
                if dataset_utils.vid_name_to_vid_base_name(self.vids_data[i]['vid_name']) in valid_vid_names]
            self._print('Splitting {} vids into {} train and {} valid'.format(
                len(self.vids_data),
                len(self.vids_train),
                len(self.vids_valid)))
        else:
            # just set train and validation to the same, since we are debugging
            self.vids_train = self.vids_data
            self.vids_valid = self.vids_data

        self._print('Loaded {} total frames from {} train vids and {} validation vids!'.format(
            total_n_frames, len(self.vids_train), len(self.vids_valid)))

    def _collect_vids(self, load_n=None):
        vids_data = []
        vid_pkls_dir = self.params['vid_caches_dir']

        vid_pkls = list(sorted(
            [f for f in os.listdir(vid_pkls_dir) if f.endswith('.pkl')]))

        np.random.seed(17)
        np.random.shuffle(vid_pkls)

        if load_n is not None and load_n > 1:
            vid_pkls = vid_pkls[:load_n] + vid_pkls[-load_n:]
        elif load_n == 1:
            vid_pkls = [vid_pkls[0]]

        for vi, vp in enumerate(vid_pkls):
            curr_vid_data = dataset_utils.load_data_from_pkl(
                os.path.join(vid_pkls_dir, vp),
                crop_preprocessors=self.crop_preprocessors,
                framenum_filters=self.framenum_filters
            )

            if curr_vid_data is None:
                continue

            # could be multiple vids if we extracted multiple crops
            for vid_data in curr_vid_data:
                vid = {
                    'vid_name': vid_data['vid_name'],
                    'frames': vid_data['frames'],
                    'frame_ids': vid_data['frame_ids'],
                }

                # if there are enough frames (besides the last), include this video in our dataset
                if vid_data['frames'].shape[0] > 1:
                    vids_data.append(vid)

        return vids_data


    def load_dataset_all_frames(self, load_n=None, logger=None, profiler_logger=None):
        '''
        Simply concatenates all frames in the dataset along the 0th dimension. Does not retain video information
        :param debug:
        :param logger:
        :param profiler_logger:
        :return:
        '''
        self.logger = logger
        self.profiler_logger = profiler_logger
        self._print('Loading watercolors dataset {}'.format(self.display_name))
        self._print('Params: {}'.format(self.params))

        self.load_dataset_frames(load_n=load_n)

        frames_train = []
        files_train = []
        for vi in range(len(self.vids_train)):
            frames_train += [self.vids_train[vi]['frames']]
            files_train += self.vids_train[vi]['im_files']
        frames_train = np.concatenate(frames_train, axis=0)


        frames_valid = []
        files_valid = []
        for vi in range(len(self.vids_valid)):
            frames_valid += [self.vids_valid[vi]['frames']]
            files_valid += self.vids_valid[vi]['im_files']
        frames_valid = np.concatenate(frames_valid, axis=0)
        return (frames_train, files_train), (frames_valid, files_valid)


    def load_dataset_vids(self, load_n=None, logger=None, profiler_logger=None, min_len=None):
        '''
        Loads each video and saves them as lists
        :param debug:
        :param logger:
        :param profiler_logger:
        :return:
        '''
        self.logger = logger
        self.profiler_logger = profiler_logger
        self._print('Loading watercolors dataset {}'.format(self.display_name))
        self._print('Params: {}'.format(self.params))

        self.load_dataset_frames(load_n=load_n, min_len=min_len)

        frames_train = []
        files_train = []
        aux_maps_train = []


        # TODO: time is currently in dimension 0,
        # which is inconsistent with other datasets
        for vi in range(len(self.vids_train)):
            frames_train += [self.vids_train[vi]['frames']]
            files_train += [self.vids_train[vi]['im_files']]
            aux_maps_train += [self.vids_train[vi]['attn']]

        frames_valid = []
        files_valid = []
        aux_maps_valid = []

        for vi in range(len(self.vids_valid)):
            frames_valid += [self.vids_valid[vi]['frames']]
            files_valid += [self.vids_valid[vi]['im_files']]
            aux_maps_valid += [self.vids_valid[vi]['attn']]

        return (frames_train, aux_maps_train, files_train), \
               (frames_valid, aux_maps_valid, files_valid)



def create_datasets(params_list):
    if not isinstance(params_list, list):
        params_list = [params_list]
    datasets = [WatercolorsDataset(params=dp) for dp in params_list]
    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        dataset = dataset_utils.combine_dataset_names(datasets)
    return dataset

    
