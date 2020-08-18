import sys
import time
import os

import functools
import numpy as np

from keras.optimizers import Nadam

from src import experiment_engine, metrics, sequence_extractor, TimelapseFramesPredictor
from src.dataset import dataset_utils, datasets, preprocessors
from src.experiment_base import Experiment
from src.networks import network_wrappers, WGAN
from src.utils import utils, vis_utils

import tensorflow as tf

import shutil

class TimelapseSequencePredictor(Experiment):
    def get_model_name(self):
        exp_name = 'TLS'
        exp_name += '_{}'.format(self.dataset.display_name)

        exp_name += '_cvae-seqlen{}'.format(self.n_pred_frames)
        if self.arch_params['do_alternate_sampling']:
            exp_name += '_sampling-seqlen{}'.format(self.n_sfp_pred_frames)

        exp_name += F"-{self.arch_params['disc_arch']}"
        exp_name += F"-nconst{self.arch_params['n_const_disc_frames']}"
        exp_name += F"-nprevf{self.arch_params['n_prev_disc_frames']}"
        exp_name += F"-nci{self.arch_params['n_critic_iters']}"

        self.model_name = super(TimelapseSequencePredictor, self).get_model_name(exp_name)
        return self.model_name


    def __init__(self,
                 data_params, arch_params,
                 exp_root='C:\\Research\\experiments', prompt_delete_existing=True,
                 do_logging=True,
                 loaded_from_dir=None
                 ):
        self.arch_params = arch_params
        # assume that we always have a list of dictionaries, which allows us to combine
        # multiple datasets
        if not type(data_params) == list:
            data_params = [data_params]

        self.datasets = [datasets.WatercolorsDataset(params=dp) for dp in data_params]
        if len(self.datasets) == 1:
            self.dataset = self.datasets[0]
        else:
            self.dataset = dataset_utils.combine_dataset_names(self.datasets)

        # this assumes that if we are loading multiple datasets, they share most of the relevant
        # params that we will use throughout this class
        self.combined_data_params = data_params[0]

        # the network only sees the last (completed) frame, not the first frame
        self.const_frames = [-1]
        self.n_const_frames = len(self.const_frames)

        self.n_pred_frames = self.arch_params['n_pred_frames']
        self.n_chans = 3

        if 'n_sfp_pred_frames' not in self.arch_params:
            self.n_sfp_pred_frames = self.n_pred_frames
        else:
            self.n_sfp_pred_frames = self.arch_params['n_sfp_pred_frames']

        self.painter_disc_batch_real = None

        if 'pretrain_critic' in self.arch_params:
            print('Disabling sampling training!')
            self.do_sampling_training = False
        else:
            print('Enabling sampling training!')
            self.do_sampling_training = True

        self.pred_frames = None
        self.train_cvaes_every = 2
        self.iter_count = 0  # in case we want to alternate training
        self.epoch_count = 0

        super(TimelapseSequencePredictor, self).__init__(
            data_params, arch_params, exp_root=exp_root,
            prompt_delete_existing=prompt_delete_existing,
            do_logging=do_logging, loaded_from_dir=loaded_from_dir)


    def load_data(self, load_n=None):
        self.dataset = dataset_utils.combine_dataset_vids(
            self.datasets, self.dataset,
            load_n=load_n, _print=self.logger.debug)

        # Load discriminator training dataset first. Since the disc requires the shortest sequences,
        # it will be most inclusive dataset. This will help us prune unused frames later
        self.logger.debug('Loading sequences with at least {} prev frames'.format(
                            self.arch_params['n_prev_disc_frames']),
                          'for disc training dataset')

        self.seqs_infos_disc_train, self.seqs_infos_disc_valid \
            = sequence_extractor.extract_sequences_by_index_from_datasets(
            self.datasets, self.dataset, _print=self.logger.debug,
            seq_len=self.arch_params['n_prev_disc_frames'] + 1,  #  our disc conditions on previous frames and 1 current frame
            n_prev_frames=self.arch_params['n_prev_disc_frames'],
            do_filter_by_prev_attn=False,
            # it's okay to start with some 0s in the attn map (e.g. the first seq [0, 1,...]
            include_starter_seq=True,
            do_prune_unused=True
        )
        self._print('Loaded total disc train sequences of len {}: {}, train seq infos: {}...{}'.format(
            self.arch_params['n_prev_disc_frames'] + 1,
            len(self.seqs_infos_disc_train),
            self.seqs_infos_disc_train[:min(5, len(self.seqs_infos_disc_train))],
            self.seqs_infos_disc_train[-min(5, len(self.seqs_infos_disc_train)):]
        ))

        if load_n is None:
            seqs_imfiles_disc_train = [
                ','.join([
                    (self.dataset.vids_train[seq_info[0]]['im_files'][fi] if fi is not None else 'blank')
                    for fi in seq_info[1]]) for seq_info in self.seqs_infos_disc_train
            ]
            with open(os.path.join(self.exp_dir, 'train_disc_seqs.txt'), 'w') as f:
                f.writelines([vn + '\n' for vn in seqs_imfiles_disc_train])

            seqs_imfiles_disc_valid = [
                ','.join([
                    (self.dataset.vids_valid[seq_info[0]]['im_files'][fi] if fi is not None else 'blank')
                    for fi in seq_info[1]]) for seq_info in self.seqs_infos_disc_valid
            ]
            with open(os.path.join(self.exp_dir, 'valid_disc_seqs.txt'), 'w') as f:
                f.writelines([vn + '\n' for vn in seqs_imfiles_disc_valid])

        # keep track of all used frames
        for seq_info in self.seqs_infos_disc_train:
            vid_idx, seq_frame_idxs, seq_firstlast_idxs = seq_info
            # sanity check to make sure we're indexing the right video
            if max([fi for fi in seq_frame_idxs if fi is not None]) >= \
                    self.dataset.vids_train[vid_idx]['frames'].shape[0]:
                print(self.dataset.vids_train[vid_idx]['vid_name'])
                print(self.dataset.vids_train[vid_idx]['frames'].shape)
                print(seq_frame_idxs)
                sys.exit()

        vidnames_disc_train = [self.dataset.vids_train[vi]['vid_name']
                               for (vi, _, _) in self.seqs_infos_disc_train]
        vidnames_disc_valid = [self.dataset.vids_valid[vi]['vid_name']
                               for (vi, _, _) in self.seqs_infos_disc_valid]

        with open(os.path.join(self.exp_dir, 'train_disc_vidnames.txt'), 'w') as f:
            f.writelines([vn + '\n' for vn in list(sorted(list(set(vidnames_disc_train))))])
        with open(os.path.join(self.exp_dir, 'valid_disc_vidnames.txt'), 'w') as f:
            f.writelines([vn + '\n' for vn in list(sorted(list(set(vidnames_disc_valid))))])

        # add some time steps with no change.
        # a little hacky, but just manually add in ~10% of blank entries for the discriminator
        # n_orig_seqs = self.seq_frames_disc_train.shape[0]
        n_orig_seqs = len(self.seqs_infos_disc_train)

        self._print('Discriminator train seqs: {}'.format(len(self.seqs_infos_disc_train)))

        # keep a list of np.ndarrays, where each element is a different sequence len
        self.seq_frames_train = []
        self.firstlast_train = []
        self.aux_maps_train = []
        self.seqs_imfiles_train = []
        self.seqs_isdone_train = []

        # for data loading by index
        self.seqs_infos_train = []

        n_train_seqs = 0

        # Load the dataset we will use for sequential training
        for i, pred_seq_len in enumerate(self.n_pred_frames):
            self.logger.debug('Loading seqs with prev {}, pred {} for main training dataset'.format(
                self.combined_data_params['n_prev_frames'], pred_seq_len))

            # load an extra frame at the start so that we can compute attention with the correct framenum delta
            extract_seq_len = self.combined_data_params['n_prev_frames'] + pred_seq_len

            seq_infos_train, seq_infos_valid = sequence_extractor.extract_sequences_by_index_from_datasets(
                self.datasets, self.dataset, _print=self.logger.debug,
                seq_len=extract_seq_len,
                n_prev_frames=self.combined_data_params['n_prev_frames'],
                do_filter_by_prev_attn=False, # it's okay to start with some 0s in the attn map (e.g. the first seq [0, 1,...]
                include_starter_seq=True,
                do_prune_unused=False
            )

            self.seqs_infos_train.append(seq_infos_train)

            if i == len(self.n_pred_frames) - 1:  # we only validate on the last and longest sequence
                self.seqs_infos_valid = seq_infos_valid

            self._print('Loaded total train sequences of len {}: {}, train seq infos: {}...{}'.format(
                extract_seq_len,
                len(seq_infos_train),
                seq_infos_train[:min(5, len(seq_infos_train))],
                seq_infos_train[-min(5, len(seq_infos_train)):]
            ))
            self._print('Loaded total starter seqs of len {}: {}'.format(
                extract_seq_len,
                len([i for i, seq_info in enumerate(seq_infos_train) if seq_info[1][0] is None])))
            n_train_seqs += len(seq_infos_train)

            if load_n is None:
                seqs_imfiles_train = [
                    ','.join([
                        (self.dataset.vids_train[seq_info[0]]['im_files'][fi] if fi is not None else 'blank')
                        for fi in seq_info[1]]) for seq_info in seq_infos_train
                ]
                with open(os.path.join(self.exp_dir, 'train_seqs_len{}.txt'.format(pred_seq_len)), 'w') as f:
                    f.writelines([vn + '\n' for vn in seqs_imfiles_train])

            # keep track of which frame idxs we've used
            for seq_info in seq_infos_train:
                vid_idx, seq_frame_idxs, seq_firstlast_idxs = seq_info
                if max([fi for fi in seq_frame_idxs if fi is not None]) >= self.dataset.vids_train[vid_idx]['frames'].shape[0]:
                    print(self.dataset.vids_train[vid_idx]['vid_name'])
                    print(self.dataset.vids_train[vid_idx]['frames'].shape)
                    print(seq_frame_idxs)
                    sys.exit()

            seqs_imfiles_valid = [
                ','.join([
                    (self.dataset.vids_valid[seq_info[0]]['frame_ids'][fi] if fi is not None else 'blank')
                    for fi in seq_info[1]]) for seq_info in seq_infos_valid
            ]
            self._print('Total valid sequences of len {}: {}, valid seq infos: {}...{}'.format(
                extract_seq_len,
                len(seq_infos_valid),
                seq_infos_valid[:min(5, len(seq_infos_valid))],
                seq_infos_valid[-min(5, len(seq_infos_valid)):]
            ))

        if load_n is None:
            with open(os.path.join(self.exp_dir, 'valid_seqs.txt'), 'w') as f:
                f.writelines([vn + '\n' for vn in seqs_imfiles_valid])

        vidnames_train = [self.dataset.vids_train[seqs_infos[0]]['vid_name']
                          for seqs_infos_list in self.seqs_infos_train for seqs_infos in seqs_infos_list]

        vidnames_valid= [self.dataset.vids_valid[vi]['vid_name']
                          for vi, _, _ in seq_infos_valid]

        if load_n is None:
            # also keep track of vid names
            with open(os.path.join(self.exp_dir, 'train_vidnames.txt'), 'w') as f:
                f.writelines([vn + '\n' for vn in sorted(list(set(vidnames_train)))])
            with open(os.path.join(self.exp_dir, 'valid_vidnames.txt'), 'w') as f:
                f.writelines([vn + '\n' for vn in sorted(list(set(vidnames_valid)))])

        self._print('Total training sequences: {} loaded from dataset {}!'.format(
            n_train_seqs, self.dataset.display_name
        ))

        # collect some information from the data we loaded
        self.n_chans = self.dataset.vids_data[0]['frames'].shape[-1]
        self.frame_shape = self.dataset.vids_data[0]['frames'].shape[1:]
        self.crop_shape = tuple(self.combined_data_params['pad_to_shape'])

        # display smaller images during training so they don't take up so much space
        self.display_h = max(64, min(self.crop_shape[0], 128))

        self.gt_input_frames = [-1]
        self.n_input_frames = len(self.gt_input_frames) + self.combined_data_params['n_prev_frames']

        # we will stack prev frames and firstlast in the channels dimension
        self.cond_input_frames_shapes = [
            self.crop_shape[:-1] + (c,) for c in
            [self.n_chans] * self.n_input_frames
        ]
        self.cond_input_names = ['last_frame', 'prev_frame']

        self.n_ae_inputs = len(self.cond_input_frames_shapes) + 1
        self.n_cond_inputs = len(self.cond_input_frames_shapes)

        self.perframe_input_frames_stack_shape = self.crop_shape[:-1] + \
            ((self.n_const_frames + self.combined_data_params['n_prev_frames']) * self.n_chans,)

        self.logger.debug('Input frames stack shape (final and previous frames)')
        self.logger.debug(self.perframe_input_frames_stack_shape)

        # Load a separate dataset for the sequential sampling trainer
        self.seq_frames_sfp_train = []
        self.final_sfp_train = []
        self.aux_maps_sfp_train = []
        self.seqs_imfiles_sfp_train = []
        self.seqs_isdone_sfp_train = []  # not really used, but helpful as dumlmy shapes

        super(TimelapseSequencePredictor, self).load_data()
        return 0

    def _initialize_child_experiment(self, exp_prefix, exp_dir, exp_epoch):
        new_exp_dir = os.path.join(self.exp_dir, F'{exp_prefix}_exp')
        exp_class = TimelapseFramesPredictor.TimelapseFramesExperiment

        if not os.path.isdir(new_exp_dir) \
                or not os.path.isfile(os.path.join(new_exp_dir, 'arch_params.json'))\
                or len(os.listdir(os.path.join(new_exp_dir, 'models'))) == 0: # if this is our first time loading the painter

            # TODO: not necessary to load the dataset here
            # load the painter
            child_exp, child_epoch = experiment_engine.load_experiment_from_dir(
                exp_dir,
                exp_class,
                load_n=1, load_epoch=exp_epoch,
                do_load_models=False,
                prompt_update_name=False,
            )

            old_models_dir = child_exp.models_dir

            # make the child directory
            if not os.path.isdir(new_exp_dir):
                os.mkdir(new_exp_dir)

            # copy over folders and log files
            for fd in os.listdir(child_exp.exp_dir):
                if os.path.isdir(os.path.join(child_exp.exp_dir, fd)) \
                        and not os.path.isdir(os.path.join(new_exp_dir, fd)):
                    # logs, figures, models folders -- just make empty ones at the target dir
                    os.mkdir(os.path.join(new_exp_dir, fd))
                elif os.path.isfile(os.path.join(child_exp.exp_dir, fd)) \
                        and not os.path.isfile(os.path.join(new_exp_dir, fd)):
                    # copy any log files
                    shutil.copy2(os.path.join(child_exp.exp_dir, fd), new_exp_dir)

            # now copy over the model files that we need
            for m in child_exp.models:
                model_file_prefix = F"{m.name}_epoch{exp_epoch}_"
                matching_model_filenames = [f for f in os.listdir(old_models_dir) if f.startswith(model_file_prefix)]

                assert len(matching_model_filenames) == 1

                new_model_name = exp_prefix + '_' + m.name
                old_model_file = os.path.join(old_models_dir, matching_model_filenames[0])
                new_model_file = os.path.join(new_exp_dir, 'models',
                                 new_model_name + F"_epoch{exp_epoch}.h5")
                # copy over each relevant model with a new model name
                if not os.path.isfile(new_model_file):
                    shutil.copy2(old_model_file, new_model_file)
                    self.logger.debug('Copied initial model {} to {}'.format(old_model_file, new_model_file))

        ## point the new expeirment at the child directory
        # load the exp from our child directory
        child_exp, child_epoch = experiment_engine.load_experiment_from_dir(
            new_exp_dir,
            exp_class,
            load_n=1, load_epoch=exp_epoch, # TODO: we need to load if we are not continuing the sequential trainer
            do_load_models=False, prompt_update_name=False
        )
 
        child_exp._init_logger()
        if 'painter' in exp_prefix: # only for paitner?
            if 'recon_fn' in self.arch_params:
                child_exp.n_recon_outputs = len(self.arch_params['recon_fn'].split('-')) # TODO: a bit hacky, but we dont want to create vgg here because it slows us down at test time
                child_exp.arch_params['recon_fn'] = self.arch_params['recon_fn']
                child_exp.create_models()
            child_exp.gpu_aug_params = None

        # change the name so we dont run into a duplicate issue with the director
        for m in list(set(child_exp.models + child_exp.models_to_print + [child_exp.cvae.sampling_model, child_exp.tester_model])):
            m.name = exp_prefix + '_' + m.name

        # load models after we have updated the names
        child_epoch = child_exp.load_models(exp_epoch)
        
        child_exp._print_models()
        child_exp._save_params()

        return child_exp, child_epoch


    def create_models(self, eval=False, verbose=False):
        self.painter_exp, painter_epoch = self._initialize_child_experiment(
            exp_prefix='painter',
            exp_dir=self.arch_params['painter_dir'], exp_epoch=self.arch_params['painter_epoch']
        )

        # our painter could use a cvae, or just a unet
        self.painter_latent_shape = (self.painter_exp.latent_dim,)

        # creates trainer and tester models, one for each sequence len
        self._create_wrapper_models()

        self.models_to_print = self.painter_exp.models \
                               + self.trainer_models + [self.tester_model, self.painter_exp.cvae.sampling_model]

        self.models_to_print += self.sfp_trainer_models

        # save this as a dummy model so that load_models has files that it can get the latest epoch from.
        # we save the trainer instead of the tester since the lambda-recurrent layer in the tester/sampler can't be pkled properly by model.save
        self.models = [self.trainer_models[0]]
        # if we are using a discriminator on any of the outputs...
        self.director_disc_model = None
        self.painter_disc_model = None
        scale_disc_lr = 1.
        gp_lambda = 10.

        # discriminator on output frames
        # final frame, previous frames, and current frame (or current delta)
        self.painter_disc_input_shape = self.crop_shape[:-1] \
            + (self.n_chans * (1 + self.arch_params['n_prev_disc_frames'] + 1),)

        with tf.device('/gpu:1'):
            # create a simple encoder as a discriminator
            self.painter_disc_model = WGAN.discriminator_patch(
                input_shape=self.painter_disc_input_shape,
                enc_params=self.arch_params['disc_enc_params'],
                include_activation=False,
                include_dense=not 'star' in self.arch_params['disc_arch'],
                model_name='painter_disc_model',
            )

            # improved WGAN training wrapper
            self.painter_disc_trainer = WGAN.wgan_gp_trainer(
                self.painter_disc_input_shape,
                self.painter_disc_model,
            )
            self._make_disc_trainable(trainable=True)
            self.painter_disc_trainer.compile(
                optimizer=Nadam(lr=self.arch_params['lr'] * scale_disc_lr),
                loss=[WGAN.wasserstein_loss, WGAN.wasserstein_loss, WGAN.gp_loss],
                loss_weights=[1, 1, gp_lambda],
            )

            self._make_disc_trainable(trainable=False)
            self.disc_loss_names = ['total', 'disc_fake_score', 'disc_-real_score', 'disc_grad']

            self.models += [self.painter_disc_model]
            self.models_to_print += [self.painter_disc_trainer]


        self.load_models(self.arch_params['disc_epoch'])

        super(TimelapseSequencePredictor, self).create_models()
        return self.models


    def _create_wrapper_models(self):
        n_painter_frame_outputs = self.painter_exp.n_recon_outputs
        self.trainer_models = []

        for seq_len in self.n_pred_frames:
            self.trainer_models.append(
                network_wrappers.perframe_sequence_trainer_noattn(
                    conditioning_input_shapes=self.cond_input_frames_shapes,
                    conditioning_input_names=self.cond_input_names,
                    input_gt_frames_shape=self.crop_shape + (seq_len,),
                    perframe_painter_model=self.painter_exp.trainer_model,
                    seq_len=seq_len,
                    n_painter_frame_outputs=n_painter_frame_outputs,
                    n_prev_disc_frames=self.arch_params['n_prev_disc_frames'],
                    n_prev_frames=self.combined_data_params['n_prev_frames'],
                )
            )

        self.tester_model = network_wrappers.perframe_sequence_tester(
            perframe_tester_model=self.painter_exp.tester_model,
            latent_shape=self.painter_latent_shape,
            seq_len=self.n_pred_frames[-1], # only test on hte longest sequence
            n_prev_frames=self.combined_data_params['n_prev_frames'],
        )
    
        self.sfp_tester_model = None
        self.sfp_trainer_models = []

        for seq_len in self.n_sfp_pred_frames:
            # trainer model that samples from prior for both cvaes. loss should only be on
            # reconstructing the very last frame, if at all
            self.sfp_trainer_models.append(
                network_wrappers.perframe_sampling_sequence_trainer_noattn(
                    conditioning_input_shapes=self.cond_input_frames_shapes,
                    conditioning_input_names=self.cond_input_names,
                    perframe_painter_model=self.painter_exp.cvae.sampling_model,
                    seq_len=seq_len,
                    n_prev_frames=self.combined_data_params['n_prev_frames'],
                    n_prev_disc_frames=self.arch_params['n_prev_disc_frames'],
                    n_const_disc_frames=self.arch_params['n_const_disc_frames'],
                    n_painter_frame_outputs=n_painter_frame_outputs,
                    painter_latent_shape=self.painter_latent_shape,
                    make_painter_disc_stack=True,
                )
            )

        # if our longest sfp sequence is longer than our normal training sequence, we should show resutls on it
        self.sfp_tester_model = network_wrappers.perframe_sequence_tester(
            perframe_tester_model=self.painter_exp.tester_model,
            latent_shape=self.painter_latent_shape,
            seq_len=self.n_sfp_pred_frames[-1], # only test on hte longest sequence
            n_prev_frames=self.combined_data_params['n_prev_frames'],
        )


    def create_generators( self, batch_size ):
        self.batch_size = batch_size
        self.train_gens = []

        for i, seq_len in enumerate(self.n_pred_frames):

            self.train_gens.append(self._generate_random_frame_sequences_by_index(
                vids_data_list=self.dataset.vids_train,
                seq_infos=self.seqs_infos_train[i],
                n_pred_frames=seq_len,
                batch_size=batch_size, randomize=True,
            ))

        self.valid_gen = self._generate_random_frame_sequences_by_index(
            vids_data_list=self.dataset.vids_valid,
            seq_infos=self.seqs_infos_valid,
            n_pred_frames=self.n_pred_frames[-1],
            batch_size=batch_size, randomize=False
        )

        self.sfp_train_gens = []

        # no fixed sequence length, so just train on blank-finish video sequences
        for i, seq_len in enumerate(self.n_sfp_pred_frames):
            self.sfp_train_gens.append(self.generate_sequences_with_sampling_from_prior(
                vids_data_list=self.dataset.vids_train,
                seq_len=seq_len,
                batch_size=batch_size, randomize=True,
            ))

        self.sfp_valid_gen = self.generate_sequences_with_sampling_from_prior(
            vids_data_list=self.dataset.vids_valid,
            seq_len = self.n_sfp_pred_frames[0],
            batch_size=batch_size, randomize=False,
            do_aug=False)

        self.painter_exp.create_generators(batch_size)

        self.painter_disc_gen = self._generate_random_frame_sequences_by_index(
            vids_data_list=self.dataset.vids_train, seq_infos=self.seqs_infos_disc_train,
            n_pred_frames=1,
            batch_size=batch_size, randomize=True,
            yield_director_disc_stack=False,
            yield_painter_disc_stack=True,
        )

        self.painter_disc_valid_gen = self._generate_random_frame_sequences_by_index(
            vids_data_list=self.dataset.vids_valid, seq_infos=self.seqs_infos_disc_valid,
            n_pred_frames=1,
            batch_size=batch_size, randomize=True,
            yield_director_disc_stack=False,
            yield_painter_disc_stack=True,
        )

        self.disc_score_zeros = np.zeros((self.batch_size,))

    def _generate_random_frame_sequences_by_index(self, vids_data_list,
                                                  seq_infos,
                                         n_pred_frames,  # number of target frames to output
                                         batch_size=8,
                                         randomize=False,
                                         do_aug=False, aug_scale_range=None,
                                         do_sample_from_prior=False,
                                         yield_director_disc_stack=False,
                                         yield_painter_disc_stack=False,
                                         only_recon_end=False,
                                         ):

        seqs_gen = sequence_extractor._generate_intermediate_seqs(
            vids_data_list, seq_infos,
            batch_size=batch_size, n_pred_frames=n_pred_frames,
            n_prev_frames=self.combined_data_params['n_prev_frames'], 
            n_prev_attns=self.combined_data_params['n_prev_frames'],
            crop_type=self.combined_data_params['crop_type'],
            crop_shape=self.crop_shape,
            randomize=randomize,
            do_aug=do_aug, aug_scale_range=aug_scale_range,
            do_normalize_frames=self.combined_data_params['normalize_frames']
        )
        while True:
            firstlast_batch, prev_frames, pred_frames, \
            seq_imfiles_batch = next(seqs_gen)

            # actually re-concatenate prev and curr frames since we need to sweep a window along them
            pred_frames = np.transpose(pred_frames, (0, 1, 2, 4, 3))
            prev_frames = np.transpose(prev_frames, (0, 1, 2, 4, 3)) # put time in -2 axis
            frames_seq_batch = np.concatenate([prev_frames, pred_frames], axis=-1)


            ######## compile targets of director and painter models. use child experiments to get the order right ######
            if only_recon_end:  # SFP trainer only tries to reconstruct the last frame
                pred_frames = np.tile(pred_frames[..., [-1]],
                                      tuple([1] * (len(pred_frames.shape) - 1)) + (n_pred_frames,))
            # make sure these are the correct order
            # turn each target into a sequence, since our trainer will concatenate
            # time in the last dimension
            for t in range(n_pred_frames):
                curr_painter_targets = self.painter_exp._make_model_targets(
                    Y=pred_frames[..., t],
                )
                # add a time dimension to each target
                curr_painter_targets = [
                    target[..., np.newaxis] for target in curr_painter_targets]

                if t == 0:
                    painter_gt_targets = curr_painter_targets
                else:
                    painter_gt_targets = [
                        np.concatenate([
                            painter_gt_targets[ti], curr_painter_targets[ti]],
                            axis=-1) for ti in range(len(curr_painter_targets))]


            ############# NOW COMPILE INPUTS AND TARGETS! #####################################################
            if yield_director_disc_stack or yield_painter_disc_stack:
                last_frame_batch = firstlast_batch[..., -self.n_chans:]

                # each frames_seq_batch has n_prev_frames + n_pred_frames time steps
                real_disc_inputs = []
                for pred_t in range(n_pred_frames):
                    # we want to make a batch of shape b x h x w x n_prev_disc_frames * c x n_pred_frames,
                    # so that we can have a sequence of disc inputs
                    # count backwards because we don't always have the same number of previous frames (e.g. in disc generator)
                    t = -n_pred_frames + pred_t

                    # for an input stroke at t-1 to t, the prev frames would be t-n_disc_prev_frames,...., t-2, t-1
                    prev_frames_batch = frames_seq_batch[..., t - self.arch_params['n_prev_disc_frames']:t]
                    # stack previous frames in the channels dimensions
                    prev_frames_batch = np.reshape(
                        np.transpose(prev_frames_batch, (0, 1, 2, 4, 3)),
                        (batch_size,) + self.crop_shape[:-1] + (
                            self.n_chans * self.arch_params['n_prev_disc_frames'], 1))

                    disc_inputs = []
                    if self.arch_params['n_prev_disc_frames'] > 0:
                        disc_inputs += [prev_frames_batch]

                    if yield_painter_disc_stack:
                        frames_batch = np.reshape(
                            frames_seq_batch[..., -1],
                            (batch_size,) + self.crop_shape[:-1] + (-1, 1))

                        disc_inputs += [frames_batch]

                    curr_disc_input_stack = np.concatenate(disc_inputs, axis=-2)
                    real_disc_inputs.append(curr_disc_input_stack)

                # concatenate in time since we collected a bunch of steps ni time
                real_disc_inputs = np.concatenate(real_disc_inputs, axis=-1)

                if self.arch_params['n_const_disc_frames'] > 0:
                    real_disc_inputs = [np.tile(last_frame_batch[..., np.newaxis],  # add a time dim
                                                (1, 1, 1, 1, n_pred_frames)), real_disc_inputs]

                gt_inputs = None

                # yields disc input stack, aug matrix
                yield np.concatenate(real_disc_inputs, axis=-2)

            else:
                # sequential trainer inputs and outputs

                ########## start with initial frames and attentions ###################
                # split into (n_prev_frames + 1) frames in a list
                prev_frames = [frame[..., 0] for frame in
                               np.split(prev_frames, self.combined_data_params['n_prev_frames'], axis=-1)]

                # sequence trainer starts out with final frame, prev attns, prev frames as input
                starter_inputs = [firstlast_batch] + prev_frames

                painter_disc_target = np.ones(
                    (batch_size,) + self.painter_disc_input_shape + (n_pred_frames,))

                # input either latent zeros or gt depending on whether we are sampling or reconstructing
                # in the cvae
                if do_sample_from_prior:
                    latent_dummy_inputs = [self.painter_exp.zeros_latent]
                    target_seqs = painter_gt_targets[:self.painter_exp.n_recon_outputs]
                else:
                    # pass these gt values without augmentation, since we augment the inputs in-network
                    gt_inputs = [pred_frames]

                    # annoyingly, the ordering of these inputs is frames, attn, gt_frames, gt_attn
                    if not isinstance(painter_gt_targets, list):
                        painter_gt_targets = [painter_gt_targets]
                    target_seqs = painter_gt_targets

                if do_sample_from_prior:
                    target_seqs += [painter_disc_target]

                    yield starter_inputs, latent_dummy_inputs, target_seqs
                else:
                    yield starter_inputs, gt_inputs, target_seqs

    def generate_sequences_with_sampling_from_prior(self, vids_data_list,
                                                    seq_len,
                                                    batch_size=8,
                                                    randomize=False,
                                                    do_aug=False, aug_scale_range=0.5,
                                                    ):
        # TODO: functionality to start or end in mid-sequence? affects nuber of steps though

        n_vids = len(vids_data_list)
        # we only need to sample videos, which we will reference by the last frame
        vid_idxs_batch = np.asarray(list(range(batch_size)), dtype=int) - batch_size

        while True:
            if randomize:
                # assumes that we always have more sequences than batch size...
                vid_idxs_batch = np.random.choice(n_vids, batch_size, replace=True)
            else:
                vid_idxs_batch += batch_size
                vid_idxs_batch[vid_idxs_batch >= n_vids] -= n_vids
            
            const_frames_batch = [vids_data_list[vi]['frames'][-1] for vi in vid_idxs_batch]


            const_frames_temp = [None] * batch_size
            for ei in range(batch_size):
                curr_frame_shape = const_frames_batch[ei].shape
                ims_stack = [const_frames_batch[ei][np.newaxis]] # stack isnt necessary, but do this for consistency with sequence_extractor
                stack_chans = [np.prod(im.shape[3:]) for im in ims_stack]

                const_frames_crop \
                    = sequence_extractor._apply_op_to_im_stack(
                    ims_stack,
                    do_reshape_to=(1,) + curr_frame_shape[:-1] + (-1,), # squash channels
                    output_shapes=[self.crop_shape[:2] + im.shape[3:] for im in ims_stack],
                    use_output_idx=0,
                    op=functools.partial(preprocessors.crop_rand,
                                         target_shape=self.crop_shape,
                                         n_crops=1, frame_ids=None,
                                         border_color=tuple([255] * np.sum(stack_chans)),
                                         do_scale_before_crop=True, # no borders!
                                         verbose=False),
                )
                # index into list of ims in stack? and then into list of crops
                const_frames_temp[ei] = const_frames_crop[0][0][np.newaxis]

            const_frames = np.concatenate(const_frames_temp, axis=0) / 255.
            const_frames = vis_utils.normalize(const_frames)

            # include n_prev_frames starter frames in a list
            prev_frames = [np.ones((batch_size,) + self.crop_shape)] * self.combined_data_params['n_prev_frames']
            starter_inputs = [const_frames] + prev_frames

            latent_dummy_inputs = [self.painter_exp.zeros_latent]

            ########## put together output targets ###################
            target_seqs = [np.tile(const_frames[..., np.newaxis], (1, 1, 1, 1, seq_len))] * self.painter_exp.n_recon_outputs

            # discriminator targets go last in the sfp model
            painter_disc_target = np.ones(
                (batch_size,) + self.painter_disc_input_shape + (seq_len,))
            target_seqs += [painter_disc_target]

            yield starter_inputs, latent_dummy_inputs, target_seqs


    def compile_models(self, run_options=None, run_metadata=None):
        # TODO: make sure the child experiment loads properly if we put this here
        # TODO: a little hacky, but change the painter outputs here
        if 'recon_fn' in self.arch_params: # if we overwrote the reconstruction function
            lfs, lns = utils.parse_loss_name(
                ln=self.arch_params['recon_fn'],
                normalize_input=self.combined_data_params['normalize_frames'],
                pred_shape=self.painter_exp.pred_frame_shape,
                logger=self.logger
            )
            # a little hardcoded, but disable augmentation for now FOR THE PAINTER ONLY
            # TODO: the director doesnt use aug but it expects an aug matrix as input
        ########### use our regular cvae losses from when we did independent training ########################
        self.painter_exp.compile_models()
        painter_loss_names = [dln + '_painter' for dln in self.painter_exp.loss_names[1:]]

        if 'recon_fn' in self.arch_params: # if we overwrote the reconstruction function
            lfs, lns = utils.parse_loss_name(
                ln=self.arch_params['recon_fn'],
                normalize_input=self.combined_data_params['normalize_frames'],
                pred_shape=self.painter_exp.pred_frame_shape,
                logger=self.logger
            )
            self.painter_exp.n_recon_outputs = len(lns)
            self.painter_exp.loss_functions[:self.painter_exp.n_recon_outputs] = lfs
            for i in range(self.painter_exp.n_recon_outputs):
                self.painter_exp.loss_weights[i] = self.painter_exp.loss_weights[0]

            painter_loss_names[:self.painter_exp.n_recon_outputs] = ['recon_{}_painter'.format(ln) for ln in lns]

        ########### figure out what our sampling scoring losses should be ########################
        last_recon_weight = 1.

        painter_score_fn = metrics.CriticScore(critic_model=self.painter_disc_model).compute_loss
        painter_score_name = 'painter_critic_score'
        last_recon_weight = 1.

        # first figure out what the losses should be on each frame in the predicted sequence
        trainer_perframe_loss_functions = self.painter_exp.loss_functions
        trainer_perframe_loss_weights = self.painter_exp.loss_weights
        trainer_perframe_loss_names = ['total'] + painter_loss_names


        critic_score_weight = 1.

        # a little hacky, but the director and painter recon are set to the
        # same value by default (5000). We want the attn map recon to be lower
        # weight since there is only one channel
        trainer_perframe_loss_weights[0] /= float(self.n_chans)

        if self.arch_params['do_alternate_sampling']:
            self.n_director_outputs = 0

            # sample from prior within trainer model, so no more KL losses
            sfp_perframe_loss_functions = self.painter_exp.loss_functions[:self.painter_exp.n_recon_outputs]
            # we don't need to reconstruct the exact attention map
            sfp_perframe_loss_weights = [1.] * self.painter_exp.n_recon_outputs

            # first one is "total" in both exps
            sfp_perframe_loss_names = [pln + '_painter_dummy' for pln in self.painter_exp.loss_names[1:1 + self.painter_exp.n_recon_outputs]]


            for i in range(self.n_director_outputs, self.n_director_outputs + self.painter_exp.n_recon_outputs):
                sfp_perframe_loss_weights[i] = 0.
            # TODO: do we really want no weight on the painter outputs?

            sfp_perframe_loss_functions.append(painter_score_fn)
            sfp_perframe_loss_names += [painter_score_name]
            sfp_perframe_loss_weights += [critic_score_weight]

        # make each loss function averaged over time. We don't need to do this for each sequence here, since the fn itself computes the average
        # (which we will add as a last dimension in our sequential trainer model)
        self.loss_functions= [metrics.TimeSummedLoss(
            loss_fn=lf,
            time_axis=-1,
            compute_over_frame_idxs=self.pred_frames  # only compute loss over self.pred_frames
        ).compute_loss for li, lf in enumerate(trainer_perframe_loss_functions)]

        self.loss_names = []
        self.loss_weights = []

        self.sfp_loss_functions = []
        # now average these loss functions over each prediction in each sequence

        for i, seq_len in enumerate(self.n_pred_frames):
            if run_options is not None and run_metadata is not None:
                self.trainer_models[i].compile(
                    optimizer=Nadam(lr=self.arch_params['lr']),
                    loss=self.loss_functions,
                    loss_weights=trainer_perframe_loss_weights,
                    options=run_options, run_metadata=run_metadata
                )
            else:
                self.trainer_models[i].compile(
                    optimizer=Nadam(lr=self.arch_params['lr']),
                    loss=self.loss_functions,
                    loss_weights=trainer_perframe_loss_weights
                )
            self.loss_weights.append(trainer_perframe_loss_weights)
            self.loss_names.append(trainer_perframe_loss_names)  # these are all the same, but make a list for consistency


        self.sfp_loss_names = []

        for si, seq_len in enumerate(self.n_sfp_pred_frames):
            curr_seq_loss_functions = sfp_perframe_loss_functions[:]
            curr_seq_loss_weights = sfp_perframe_loss_weights[:]
            curr_seq_loss_names = sfp_perframe_loss_names[:]

            for li in range(self.n_director_outputs, self.n_director_outputs + self.painter_exp.n_recon_outputs):
                # put a loss on the last predicted frame
                curr_seq_loss_weights[li] = last_recon_weight


            # adjust painter reconstruction losses
            for li in range(self.n_director_outputs, self.n_director_outputs + self.painter_exp.n_recon_outputs):
                # reconstruction loss only on the last time instance
                curr_seq_loss_functions[li] = metrics.TimeSummedLoss(
                    loss_fn=curr_seq_loss_functions[li],
                    time_axis=-1,
                    compute_mean=False,
                    compute_over_frame_idxs=[seq_len - 1],  # average loss across all time steps
                ).compute_loss

                # update name since this is no longer a dummy loss
                curr_seq_loss_names[li] = curr_seq_loss_names[li].replace('_dummy', '') \
                                          + '_frame{}'.format(seq_len - 1)

            # average critic score across all time steps
            # TODO: check this
            painter_critic_loss_idx = -1
            curr_seq_loss_functions[painter_critic_loss_idx] = metrics.TimeSummedLoss(
                loss_fn=curr_seq_loss_functions[painter_critic_loss_idx],
                time_axis=-1,
                compute_mean=True,
                compute_over_frame_idxs=None,
            ).compute_loss

            curr_seq_loss_names[painter_critic_loss_idx] += '_mean-{}frames'.format(seq_len)

            self.logger.debug('Compiling alternating model (sampling from prior), seqlen {}, with losses:'.format(seq_len))
            for li in range(len(curr_seq_loss_names)):
                self.logger.debug('Loss {}, fn {}, weight {}, output {}'.format(
                    curr_seq_loss_names[li], curr_seq_loss_functions[li], curr_seq_loss_weights[li], self.sfp_trainer_models[si].outputs[li]))

            self.sfp_loss_names.append(curr_seq_loss_names)
            self.sfp_trainer_models[si].compile(
                optimizer=Nadam(lr=self.arch_params['lr']),  # train this more slowly since we are less sure?
                loss=curr_seq_loss_functions,
                loss_weights=curr_seq_loss_weights
            )
        super(TimelapseSequencePredictor, self).compile_models()


    def train_on_batch( self ):
        disc_losses, disc_loss_names = self.train_discriminator()

        if not self.arch_params['do_alternate_sampling'] \
            or (self.arch_params['do_alternate_sampling'] and self.iter_count % self.train_cvaes_every == 0) \
            or not self.do_sampling_training:
            st = time.time()

            # get a random sequence len
            rand_seqlen_idx = np.random.choice(len(self.n_pred_frames), 1).astype(int)[0]

            cond_inputs, gt_inputs, targets = next(self.train_gens[rand_seqlen_idx])

            losses = self.trainer_models[rand_seqlen_idx].train_on_batch(cond_inputs + gt_inputs, targets)

            loss_names = ['train_' + ln for ln in self.loss_names[rand_seqlen_idx]]
            if np.any(np.isnan(losses)):
                self.logger.debug('Got nan in training losses!')
                self.logger.debug(loss_names)
                self.logger.debug(losses)
                sys.exit()

        elif self.arch_params['do_alternate_sampling'] and not self.iter_count % self.train_cvaes_every == 0 and self.do_sampling_training:
            # TODO: test new separated inputs
            # alternately, sample from priors and only put loss on last frame

            # get a random sequence len
            rand_seqlen_idx = np.random.choice(len(self.n_sfp_pred_frames), 1).astype(int)[0]
            self.rand_seqlen_idx_sampled_train = rand_seqlen_idx

            cond_inputs_sampled_train, latent_inputs, targets = next(self.sfp_train_gens[rand_seqlen_idx])

            losses = self.sfp_trainer_models[rand_seqlen_idx].train_on_batch(
                cond_inputs_sampled_train + latent_inputs, targets)

            loss_names = ['train_' + ln for ln in ['total'] + self.sfp_loss_names[rand_seqlen_idx]]
            if np.any(np.isnan(losses)):
                self.logger.debug('Got nan in SFP training losses!')
                self.logger.debug(loss_names)
                self.logger.debug(losses)
                sys.exit()

        assert len(losses) == len(loss_names)
        self.iter_count += 1

        return losses + disc_losses, loss_names + disc_loss_names

    def test_batches(self):
        n_valid_batches = int(np.ceil(self.get_n_test() / float(self.batch_size)))

        for bi in range(n_valid_batches):
            self.cond_inputs_valid, self.gt_inputs_valid, self.targets_valid = next(self.valid_gen)
            losses = self.trainer_models[-1].evaluate(
                self.cond_inputs_valid + self.gt_inputs_valid, self.targets_valid, verbose=False)
            if bi == 0:
                valid_losses = np.asarray(losses) / float(n_valid_batches)
            else:
                valid_losses += np.asarray(losses) / float(n_valid_batches)
        loss_names = ['valid_' + ln for ln in self.loss_names[-1]]

        # also evaluate on real training examples since we dont do this at train time
        #n_valid_disc_batches = int(np.ceil(self.seq_frames_disc_valid.shape[0]) / float(self.batch_size))
        n_valid_disc_batches = int(np.ceil(len(self.seqs_infos_disc_train) / float(self.batch_size)))

        for bi in range(n_valid_disc_batches):
            # first compute scores over the training set
            # (a little hacky, but we don't want to slow down training)
            # this generator yields disc_input_stack, aug_matrix
            disc_stack_seq = next(self.painter_disc_gen)

            disc_batch_real = disc_stack_seq[..., -1]
            disc_score = self.painter_disc_model.predict(disc_batch_real)
            if bi == 0:
                train_painter_disc_losses = np.mean(disc_score, axis=-1)
            else:
                train_painter_disc_losses += np.mean(disc_score, axis=-1)

            # this generator yields disc_input_stack, aug_matrix
            disc_stack_seq = next(self.painter_disc_valid_gen)

            disc_batch_real = disc_stack_seq[..., -1]
            disc_score = self.painter_disc_model.predict(disc_batch_real)
            if bi == 0:
                valid_painter_disc_losses = np.mean(disc_score, axis=-1)
            else:
                valid_painter_disc_losses += np.mean(disc_score, axis=-1)

        valid_losses = valid_losses.tolist()

        loss_names += [
            'valid_painter_disc_real_score',
            'train_painter_disc_real_score',
            'valid-train_painter_disc_real_score_diff'
        ]
        valid_losses = valid_losses + [
            np.sum(valid_painter_disc_losses) / n_valid_disc_batches,
            np.sum(train_painter_disc_losses) / n_valid_disc_batches,
            np.sum(valid_painter_disc_losses - train_painter_disc_losses) / n_valid_disc_batches,
        ]

        return valid_losses, loss_names

    # def save_exp_info(self, exp_dir, figures_dir, models_dir, logs_dir):
    #     super(TimelapseSequencePredictor, self).save_exp_info(
    #         exp_dir, figures_dir, models_dir, logs_dir)

    def save_models(self, epoch, iter_count=None):
        st = time.time()

        self.painter_exp.save_models(epoch=epoch + self.arch_params['painter_epoch'],
            iter_count=iter_count)
        super(TimelapseSequencePredictor, self).save_models(epoch, iter_count=iter_count)
        if self.do_profile:
            self.profiler_logger.info('Saving models took {}'.format(time.time() - st))

    def load_models(self, load_epoch=None, stop_on_missing=True, init_layers=False):
        if load_epoch is None:
            return 0

        if load_epoch == 'latest':
            load_epoch = utils.get_latest_epoch_in_dir(self.models_dir)
        else:
            load_epoch = int(load_epoch)

        self.painter_exp.load_models(
            load_epoch=self.arch_params['painter_epoch'] + load_epoch,
            stop_on_missing=stop_on_missing)

        super(TimelapseSequencePredictor, self).load_models(load_epoch=load_epoch, stop_on_missing=False)

        start_epoch = load_epoch + 1
        if 'pretrain_critic' in self.arch_params and start_epoch < self.arch_params['pretrain_critic']:
            self.do_sampling_training = False
            self._print('Disabling sampling training!')
        else:
            self.do_sampling_training = True
            self._print('Enabling sampling training!')

        return start_epoch

    def update_epoch_count(self, e):
        self.epoch_count = e
        if 'pretrain_critic' in self.arch_params and e < self.arch_params['pretrain_critic']:
            self.do_sampling_training = False
            self._print('Disabling sampling training!')
        else:
            self.do_sampling_training = True
            self._print('Enabling sampling training!')
        

    def train_discriminator(self):
        disc_losses = []
        disc_loss_names = []

        # only train the discriminator on odd iterations (when we are also training the sampling generator)
        if self.iter_count % self.train_cvaes_every == 0:
            return [], []

        self._make_disc_trainable(True)
        # get a random sequence len
        rand_seqlen_idx = np.random.choice(len(self.n_sfp_pred_frames), 1).astype(int)[0]

        for ci in range(self.arch_params['n_critic_iters']):
            # get a generator prediction to get a fake disc batch
            self.cond_inputs_sampled_train, latent_inputs, targets = next(self.sfp_train_gens[rand_seqlen_idx])
            self.sfp_trainer_models[rand_seqlen_idx].trainable = False

            # make sure we are not training the generator
            for l in self.sfp_trainer_models[rand_seqlen_idx].layers:
                l.trainable = False
            gen_preds = self.sfp_trainer_models[rand_seqlen_idx].predict(
                self.cond_inputs_sampled_train + latent_inputs)

            generated_seq = gen_preds[-1]

            # this generator yields disc_input_stack, aug_matrix
            disc_stack_seq = next(self.painter_disc_gen)

            # apply the discriminator to a random time step for each element in the batch
            disc_batch_fake = []
            disc_batch_real = []
            for i in range(generated_seq.shape[0]):
                rand_frame_idx = np.random.choice(generated_seq.shape[-1], 1)[0]
                disc_batch_fake.append(generated_seq[[i], ..., rand_frame_idx])

                rand_frame_idx = np.random.choice(disc_stack_seq.shape[-1], 1)[0]
                disc_batch_real.append(disc_stack_seq[[i], ..., rand_frame_idx])

            disc_batch_fake = np.concatenate(disc_batch_fake, axis=0)
            disc_batch_real = np.concatenate(disc_batch_real, axis=0)

            self.painter_disc_fake_batch = disc_batch_fake
            self.painter_disc_real_batch = disc_batch_real


            disc_loss = self.painter_disc_trainer.train_on_batch(
                [disc_batch_real, disc_batch_fake],
                [np.ones((self.batch_size,)), -np.ones((self.batch_size,)), self.disc_score_zeros], # positive fake, neg real scores, gp dumm
            )

            if ci == 0:
                painter_disc_losses = disc_loss
            else:
                painter_disc_losses = [painter_disc_losses[i] + disc_loss[i] for i in range(len(disc_loss))]

        self._make_disc_trainable(False)
        painter_disc_losses = [dl / self.arch_params['n_critic_iters'] for dl in painter_disc_losses]

        disc_losses += painter_disc_losses
        disc_loss_names += ['train_' + dln + '_painter' for dln in self.disc_loss_names]

        return disc_losses, disc_loss_names

    def make_train_results_im(self):
        if self.iter_count == 1:
            return self._make_train_results_im(
                data_gens=self.train_gens, models=self.trainer_models,
                do_include_gt=True)
        else:
            out_ims = [vis_utils.concatenate_with_pad([
                self._make_train_results_im(
                    data_gens=self.train_gens, models=self.trainer_models,
                    do_include_gt=True),
                self._make_sampled_results_im(
                    data_gens=self.sfp_train_gens)
            ], axis=0)]

        # if we have started training the discriminator, start making images of the training results
        if self.painter_disc_real_batch is not None:
            painter_disc_im = self._make_disc_results_im(
                self.painter_disc_real_batch,
                self.painter_disc_fake_batch,
                self.painter_disc_model,
            )
            out_ims += [painter_disc_im]
        return np.concatenate(vis_utils.pad_images_to_size(out_ims, ignore_axes=0), axis=0)


    def make_test_results_im(self):
        out_ims = [vis_utils.concatenate_with_pad([
            self._make_train_results_im(
                data_gens=[self.valid_gen], models=[self.trainer_models[-1]],
                do_include_gt=True, do_aug=False),
            self._make_sampled_results_im(
                data_gens=[self.sfp_valid_gen])
        ], axis=0)]
        return np.concatenate(vis_utils.pad_images_to_size(out_ims, ignore_axes=0), axis=0)

    def _make_train_results_im(self, data_gens, models, do_include_gt=False, do_aug=True):
        # show results for a random sequence length
        seq_idx = np.random.choice(len(data_gens), 1)[0]

        starter_cond_inputs, gt_inputs, _ = next(data_gens[seq_idx])
        # get trainer predictions
        preds = models[seq_idx].predict(starter_cond_inputs + gt_inputs)

        pred_frames = preds[0]

        starter_const_frames = starter_cond_inputs[:self.n_const_frames]
        # inputs to autoencoder branches
        starter_input_frames = starter_cond_inputs[self.n_const_frames:self.n_const_frames + self.combined_data_params['n_prev_frames']]

        target_frames = gt_inputs[0]

        batch_size = pred_frames.shape[0]
        out_ims = []
        for ei in range(batch_size):
            const_frames = [vis_utils.label_ims(
                    const_frame[[ei]], 'starter_const_[{:.2f},{:.2f}]'.format(np.min(const_frame[ei]), np.max(const_frame[ei])),
                    display_h=self.display_h
                ) for const_frame in starter_const_frames]
            starter_frames = [vis_utils.label_ims(
                input_frame[[ei]], 'starter_input_[{:.2f},{:.2f}]'.format(np.min(input_frame[ei]), np.max(input_frame[ei])), 
                display_h=self.display_h
                ) for input_frame in starter_input_frames]

            inputs_im = [
                np.concatenate(const_frames, axis=1),
                np.concatenate(starter_frames, axis=1)]
            inputs_im = np.concatenate(inputs_im, axis=0)

            # make each video into a row, then concatenate all the rows for
            # all examples
            example_output_ims_list = [inputs_im]

            if do_include_gt:
                example_output_ims_list += [vis_utils.label_ims(
                    target_frames[ei], 'gt [{:.1f},{:.1f}]'.format(np.min(target_frames[ei]), np.max(target_frames[ei])),
                    concat_axis=1, combine_from_axis=-1,
                    display_h=self.display_h)]


            example_output_ims_list += [vis_utils.label_ims(
                    pred_frames[ei],
                    'pred_[{:.1f}.{:.1f}]'.format(np.min(pred_frames[ei]), np.max(pred_frames[ei])),
                    concat_axis=1, combine_from_axis=-1,
                    display_h=self.display_h),
                ]

            example_ims = np.concatenate(
                    vis_utils.pad_images_to_size(example_output_ims_list, ignore_axes=0), axis=0)
            out_ims.append(example_ims)

        out_im = np.concatenate(out_ims, axis=0)
        return out_im

    def _make_sampled_results_im(self, data_gens):
        '''
        Creates an image summarizing the predictions made by sampling from the prior.
        :param data_gens: data generator used for training the sequential sampling model.
        :return: a h x w x 3 np.ndarray image.
        '''
        seq_idx = np.random.choice(len(data_gens), 1)[0]

        starter_cond_inputs, _, _ = next(data_gens[seq_idx])

        # parse ordering of inputs and outputs to get the images that we want
        starter_const_frames = starter_cond_inputs[:self.n_const_frames]
        starter_input_frames = starter_cond_inputs[
                              self.n_const_frames:self.n_const_frames + self.combined_data_params['n_prev_frames']]

        tester_inputs = starter_cond_inputs + [self.painter_exp.zeros_latent]
        pred_frames = self.tester_model.predict(tester_inputs)

        T = pred_frames.shape[-1]

        # if our sequence is too long, only display some of it to save space
        max_display_T = 20
        if T > max_display_T:
            pred_frames = np.concatenate([pred_frames[..., :int(max_display_T / 2)], pred_frames[..., -int(max_display_T / 2):]], axis=-1)
            pred_frame_labels = ['pred_t={}'.format(i) for i in range(10)] + ['pred_t={}'.format(i) for i in range(T-10, T)]
        else:
            pred_frame_labels = ['pred'] * pred_frames.shape[0]


        batch_size = pred_frames.shape[0]
        out_ims = []
        for ei in range(batch_size):
            # take the current example from each type of iamge in the stack (firstframe, lastframe, etc)
            const_frames = [vis_utils.label_ims(
                const_frame[[ei]],
                'starter_const [{:.1f},{:.1f}]'.format(np.min(const_frame[ei]), np.max(const_frame[ei])),
                    display_h=self.display_h,
                ) for const_frame in starter_const_frames]
            starter_frames = [vis_utils.label_ims(
                input_frame[[ei]],
                'starter_input [{:.1f},{:.1f}]'.format(np.min(input_frame[ei]), np.max(input_frame[ei])),
                display_h=self.display_h) for input_frame in starter_input_frames]

            # take the current example from each type of iamge in the stack (firstframe, lastframe, etc)
            inputs_im = [
                np.concatenate(const_frames, axis=1),
                np.concatenate(starter_frames, axis=1)]

            inputs_im = np.concatenate(inputs_im, axis=0)

            # make each video into a row, then concatenate all the rows for
            # all examples
            example_output_ims_list = [inputs_im]


            example_output_ims_list += [vis_utils.label_ims(
                pred_frames[ei],
                [pred_frame_labels[ei] + ' [{:.1f},{:.1f}]'.format(np.min(pred_frames[ei]), np.max(pred_frames[ei]))],
                display_h=self.display_h,
                concat_axis=1, combine_from_axis = -1,
            ),
            ]

            example_ims = np.concatenate(
                vis_utils.pad_images_to_size(example_output_ims_list, ignore_axes=0), axis=0)
            out_ims.append(example_ims)

        out_im = np.concatenate(out_ims, axis=0)
        return out_im

    def _make_disc_results_im(
            self, disc_batch_real, disc_batch_fake, disc_model):
        # for each real and fake batch, keep track of the discriminator score
        disc_real_scores = []
        disc_fake_scores = []

        for i in range(disc_batch_real.shape[0]):
            scores = disc_model.predict(disc_batch_real[[i]])

            disc_real_scores.append('real: {:.3f}'.format(
                np.around(scores[0, 0], 3)))

            scores = disc_model.predict(disc_batch_fake[[i]])
            disc_fake_scores.append('fake: {:.3f}'.format(
                np.around(scores[0, 0], 3)))

        disc_im_sizes = [self.n_chans * self.arch_params['n_const_disc_frames'],
             self.n_chans * self.arch_params['n_prev_disc_frames'],
             1]
        disc_im_sizes = [s for s in disc_im_sizes if not s==0]
        disc_im_splits = np.cumsum(disc_im_sizes)[:-1] # don't include the last one because of how cumsum works

        disc_real_ims = np.split(disc_batch_real, disc_im_splits, axis=-1)
        disc_fake_ims = np.split(disc_batch_fake, disc_im_splits, axis=-1)

        real_labels = ['cond_im_last', 'cond_im_prev', disc_real_scores]
        fake_labels = ['cond_im_last', 'cond_im_prev', disc_fake_scores]

        disc_out_im = np.concatenate([
            np.concatenate(
                [vis_utils.label_ims(disc_real_ims[i], real_labels[i],
                                               display_h=self.display_h,
                                               ) for i in range(len(disc_real_ims))], axis=1),
            np.concatenate(
                [vis_utils.label_ims(disc_fake_ims[i], fake_labels[i],
                                               display_h=self.display_h,
                                               ) for i in range(len(disc_fake_ims))], axis=1)]
            , axis=1)
        return disc_out_im

    def get_n_train(self):
        # not exactly right since we might have multiple seqs per vid,
        # but this is fine for an estimate
        return max([len(seqs_infos) for seqs_infos in self.seqs_infos_train])

    def get_n_test(self):
        return min(50, len(self.seqs_infos_valid))

    def _make_disc_trainable(self, trainable=False):
        if self.director_disc_model is not None:
            self.director_disc_model.trainable = trainable
            for l in self.director_disc_model.layers:
                l.trainable = trainable

        if self.painter_disc_model is not None:
            self.painter_disc_model.trainable = trainable
            for l in self.painter_disc_model.layers:
                l.trainable = trainable
