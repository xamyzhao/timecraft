import os

from keras.optimizers import Nadam
import numpy as np

from src.dataset import dataset_utils, datasets

from src import sequence_extractor
from src.experiment_base import Experiment
from src.networks import cvae_class
from src.utils import utils, vis_utils


class TimelapseFramesExperiment(Experiment):
    '''
    Code for training a per-frame model. Given a completed painting
    and (optionally) some previous frame, predict the next frame in
    the painting time lapse video.
    '''
    def get_model_name( self ):
        exp_name = 'TLF'

        exp_name += '_{}'.format(self.dataset.display_name)

        exp_name += '_{}'.format(self.arch_params['model_arch'])

        exp_name += '_nprev{}'.format(self.combined_data_params['n_prev_frames'])
        exp_name += '_recon-{}'.format(self.arch_params['recon_fn'])

        if 'vae' in self.arch_params['model_arch']:
            exp_name += '_latent{}'.format(self.arch_params['latent_dim'])
            exp_name += '_ndec{}'.format(len(self.arch_params['enc_params']['nf_dec']))

        self.model_name = super(TimelapseFramesExperiment, self).get_model_name(exp_name)

        return self.model_name


    def __init__(self, data_params, arch_params, exp_root='C:\\Research\\experiments',
                 prompt_delete_existing=True, prompt_update_name=True,
                 do_logging=True,
                 loaded_from_dir=None):
        self.arch_params = arch_params
        self.latent_dim = self.arch_params['latent_dim']

        # assume that we always have a list of dictionaries, of data params,
        # which allows us to combine multiple datasets
        if not type(data_params) == list:
            data_params = [data_params]

        for dp in data_params:
            if 'true_starts_file' not in dp:
                dp['true_starts_file'] = None

        self.datasets = [datasets.WatercolorsDataset(params=dp) for dp in data_params]
        if len(self.datasets) == 1:
            self.dataset = self.datasets[0]
        else:
            self.dataset = dataset_utils.combine_dataset_names(self.datasets)

        # this assumes that if we are loading multiple datasets, they share most of the relevant
        # params that we will use throughout this class
        self.combined_data_params = data_params[0]
        self.crop_shape = tuple(self.combined_data_params['pad_to_shape'])

        self.gt_input_frames = [-1]
        self.n_input_frames = len(self.gt_input_frames) + self.combined_data_params['n_prev_frames']

        self.n_pred_frames = self.combined_data_params['n_pred_frames']
        self.n_chans = 3

        if 'activation' not in self.arch_params:
            self.arch_params['activation'] = None

        self.epoch_count = 0
        self.iter_count = 0

        self.n_chans = 3  # assume rgb
        self.cond_input_frames_shapes = [
            self.crop_shape[:-1] + (c,) for c in
            [self.n_chans] * self.n_input_frames]
        self.cond_input_names = ['last_frame', 'prev_frame']

        # we will include the true current frame as input
        self.ae_input_frames_shapes = self.cond_input_frames_shapes + [self.crop_shape]
        self.ae_input_names = self.cond_input_names + ['curr_frame']
        self.ae_input_names = ['ae_' + input_name for input_name in self.ae_input_names]

        # important to have a distinction between conditional and ae inputs, since some of these values are repeated
        self.cond_input_names = ['cond_' + input_name for input_name in self.cond_input_names]

        self.pred_frame_shape = self.crop_shape

        super(TimelapseFramesExperiment, self).__init__(
            data_params, arch_params, exp_root=exp_root,
            prompt_delete_existing=prompt_delete_existing,
            prompt_update_name=prompt_update_name, do_logging=do_logging,
            loaded_from_dir=loaded_from_dir)

        self.logger.debug('Autoencoder input shapes')
        self.logger.debug(self.ae_input_frames_shapes)
        self.logger.debug('Conditioning input shapes')
        self.logger.debug(self.cond_input_frames_shapes)

    def load_data(self, load_n=None):
        self.dataset = dataset_utils.combine_dataset_vids(
            self.datasets, self.dataset, load_n=load_n, _print=self.logger.debug)

        self._print('Combined dataset has {} train vids: {}...'.format(
            len(self.dataset.vids_train),
            [vd['vid_name'] for vd in self.dataset.vids_train[:min(5, len(self.dataset.vids_train))]]))

        load_n_prev_frames = self.combined_data_params['n_prev_frames']

        self.seq_infos_train, self.seq_infos_valid = sequence_extractor.extract_sequences_by_index_from_datasets(
            self.datasets, self.dataset, _print=self.logger.debug,
            n_prev_frames=load_n_prev_frames, seq_len=load_n_prev_frames + 1,
            do_filter_by_prev_attn=False,
            include_nonadj_seqs=False,
            include_starter_seq=True,
            do_prune_unused=True
        )


        self.logger.debug('Loaded {} sequences: train seq infos: {}...{}'.format(
            len(self.seq_infos_train),
            self.seq_infos_train[:min(5, len(self.seq_infos_train))],
            self.seq_infos_train[-min(5, len(self.seq_infos_train)):]
        ))

        self._print('Loaded total starter seqs: {}'.format(
            len([i for i, seq_info in enumerate(self.seq_infos_train) if seq_info[1][0] is None])))

        if load_n is None:
            #print([seq_info[1] for seq_info in self.seq_infos_train])
            seqs_imfiles_train = [
                ','.join([
                    (self.dataset.vids_train[seq_info[0]]['im_files'][fi] if fi is not None else 'blank') 
                for fi in seq_info[1]]) for seq_info in self.seq_infos_train
            ]
            with open(os.path.join(self.exp_dir, 'train_seq.txt'), 'w') as f:
                f.writelines([vn + '\n' for vn in seqs_imfiles_train])

            seqs_imfiles_valid = [
                ','.join([
                    (self.dataset.vids_valid[seq_info[0]]['im_files'][fi] if fi is not None else 'blank') 
                for fi in seq_info[1]]) for seq_info in self.seq_infos_valid
            ]
            with open(os.path.join(self.exp_dir, 'valid_seq.txt'), 'w') as f:
                f.writelines([vn + '\n' for vn in seqs_imfiles_valid])

            vidnames_train = list(sorted(list(set([self.dataset.vids_train[vi]['vid_name']
                              for vi, _, _ in self.seq_infos_train]))))
            vidnames_valid = list(sorted(list(set([self.dataset.vids_valid[vi]['vid_name']
                              for vi, _, _ in self.seq_infos_valid]))))

            with open(os.path.join(self.exp_dir, 'train_vidnames.txt'), 'w') as f:
                f.writelines([vn + '\n' for vn in vidnames_train])
            with open(os.path.join(self.exp_dir, 'valid_vidnames.txt'), 'w') as f:
                f.writelines([vn + '\n' for vn in vidnames_valid])

        self.frame_shape = self.dataset.vids_data[0]['frames'].shape[1:]




        super(TimelapseFramesExperiment, self).load_data()
        return 0


    def _make_model_targets(self, Y):
        if not self.combined_data_params['normalize_frames']:
            Y = np.clip(Y, 0., 1.)
        else:
            Y = np.clip(Y, -1., 1.)

        # put a dummy Y for the transformation output
        generator_labels = [Y, Y,self.zeros_latent, self.zeros_latent]
        if self.arch_params['recon_fn'] == 'l2-vgg' or self.arch_params['recon_fn'] == 'l1-vgg':
            generator_labels = [Y] + generator_labels

        return generator_labels


    def create_models(self, eval=False, verbose=True):
        self.do_eval = eval
        self.models = []

        if 'concat' in self.arch_params['condition_by']:
            # want the max dimension to get down to at least 3
            n_concat_scales = int(np.ceil(
                np.log2(np.max(self.crop_shape[:-1]) / 3.)))
        else:
            n_concat_scales = None

        # if we are using 2 reconstruction losses, then we need 2 output images
        # to apply the losses to
        if 'l2-vgg' in self.arch_params['recon_fn'] or 'l1-vgg' in self.arch_params['recon_fn']:
            self.n_recon_outputs = 2
        else:
            self.n_recon_outputs = 1

        self.logger.debug(
            'Creating per-frame VAE with {} recon outputs!'.format(
                self.n_recon_outputs))

        self.latent_dim = self.arch_params['latent_dim']

        self.cvae = cvae_class.CVAE(
            ae_input_shapes=self.ae_input_frames_shapes,
            ae_input_names=self.ae_input_names,
            conditioning_input_shapes=self.cond_input_frames_shapes,
            conditioning_input_names=self.cond_input_names,
            source_im_idx=1, # skip the last frame, and apply the delta to the previous frame
            n_concat_scales=n_concat_scales,
            output_shape=self.pred_frame_shape,
            n_outputs=self.n_recon_outputs,
            condition_on_image=True,
            transform_latent_dim=self.latent_dim,
            transform_enc_params=self.arch_params['enc_params'],
            dec_params=self.arch_params['enc_params'],
            transform_activation=self.arch_params['activation'],
            conditioning_type=self.arch_params['condition_by']
        )

        self.cvae.create_modules()
        self.cvae.create_train_wrapper()

        self.models += self.cvae.get_models()
        self.models_to_print = self.models[:]
        self.models_to_print.append(self.cvae.trainer_model)

        self.trainer_model = self.cvae.trainer_model
        self.tester_model = self.cvae.tester_model

        super(TimelapseFramesExperiment, self).create_models()
        return self.models


    def create_generators( self, batch_size ):
        self.batch_size = batch_size
        self.train_gen = sequence_extractor.generate_random_frames_sequences(
            vids_data_list=self.dataset.vids_train, seq_infos=self.seq_infos_train,
            batch_size=batch_size, randomize=True,
            crop_shape=self.crop_shape,
            crop_type=self.combined_data_params['crop_type'],
            do_normalize_frames=True,
            do_aug=True,
            return_ids=True,
        )

        self.test_gen = sequence_extractor.generate_random_frames_sequences(
            vids_data_list=self.dataset.vids_valid, seq_infos=self.seq_infos_valid,
            batch_size=batch_size, randomize=False,
            crop_shape=self.crop_shape,
            crop_type=self.combined_data_params['crop_type'],
            do_normalize_frames=True,
            do_aug=False,
            return_ids=True,
        )

        if 'vae' in self.arch_params['model_arch']:
            self.zeros_latent = np.zeros((self.batch_size, self.latent_dim))


    def compile_models(self, run_options=None, run_metadata=None):
        # parse loss names
        recon_fn, recon_fn_name = utils.parse_loss_name(
            ln=self.arch_params['recon_fn'],
            normalize_input=self.combined_data_params['normalize_frames'],
            pred_shape=self.pred_frame_shape,
            logger=self.logger
        )

        # just compile self.model with the specified losses
        self.loss_names, self.loss_functions, self.loss_weights = self.cvae.get_losses(
            transform_reg_fn=None,
            transform_reg_lambda=None,
            transform_reg_name=None,
            recon_loss_fn=recon_fn,
            recon_loss_weight=[self.arch_params['recon_lambda']] * self.n_recon_outputs,
            recon_loss_name=recon_fn_name
        )

        self.trainer_model.compile(
            optimizer=Nadam(lr=self.arch_params['lr']),
            loss=self.loss_functions, loss_weights=self.loss_weights)

        super(TimelapseFramesExperiment, self).compile_models()


    def make_train_results_im(self):
        return self._make_results_im(
            X=self.X_train_batch,
            gt_frames=self.Y_train_batch,
            seqs_frame_files=self.seq_imfiles_train_batch)


    def make_test_results_im(self):
        return self._make_results_im(
            X=self.X_test_batch,
            gt_frames=self.Y_test_batch,
            seqs_frame_files=self.seq_imfiles_test_batch)

    def _make_results_im(self, X, gt_frames, seqs_frame_files=None):
        if seqs_frame_files is not None:
            # convert filenames to framenums
            seqs_frame_names = []
            # there should be batch_size sequences
            for seq_frame_files in seqs_frame_files:
                frame_files = seq_frame_files.split(',')
                # jsut get the video and frame name from the latter part of the filename
                frame_names = [
                    os.path.splitext(os.path.basename(f))[0].split('-')[-1] for f in frame_files]
                seqs_frame_names.append(frame_names)
        else:
            seqs_frame_names = None

        # get predictions from neural network
        preds = self.trainer_model.predict(X)
        if not isinstance(preds,list):
            preds = [preds]
        pred = preds[0]

        ae_inputs = X[:len(self.ae_input_names)]
        cond_inputs = X[len(self.ae_input_names):len(self.ae_input_names) + len(self.cond_input_names)]

        # compile autoencoder and conditioning inputs
        cond_input_im = np.concatenate([
            vis_utils.label_ims(
                cond_input, self.cond_input_names[ii])
            for ii, cond_input in enumerate(cond_inputs)
        ], axis=1)

        ae_input_im = np.concatenate([
            vis_utils.label_ims(
                ae_input, self.ae_input_names[ii])
            for ii, ae_input in enumerate(ae_inputs)
        ], axis=1)

        # concatenate everything from the autoencoder and conditional branches together
        input_ims = np.concatenate(vis_utils.pad_images_to_size([ae_input_im, cond_input_im], ignore_axes=1),
                                   axis=1)

        # if we also included the sequence im file names as a part of the generator...
        if seqs_frame_names is not None:
            batch_size = gt_frames.shape[0]
            # each element in the batch needs to be named independently
            # we want a n_prev long outer list, where each entry is a list of batch_size strings
            gt_labels = ['gt-{}'.format(seqs_frame_names[ei][-1]) for ei in range(batch_size)]
        else:
            gt_labels = 'gt'

        prev_frame = cond_inputs[1]

        pred_im = vis_utils.label_ims(pred, 'pred')
        pred_diff_im = vis_utils.label_ims(pred - prev_frame, 'pred_diff')
        gt_im = vis_utils.label_ims(gt_frames, gt_labels)
        gt_diff_im = vis_utils.label_ims(gt_frames - prev_frame, 'gt_diff')

        out_ims_list = [input_ims, pred_diff_im, gt_diff_im, pred_im, gt_im]
        out_im = np.concatenate(vis_utils.pad_images_to_size(out_ims_list, ignore_axes=1), axis=1)

        return out_im

    def get_n_train(self):
        return len(self.seq_infos_train)

    def get_n_test(self):
        return len(self.seq_infos_valid)

    def train_on_batch( self ):
        X, Y, seq_imfiles = next(self.train_gen)

        generator_labels = self._make_model_targets(Y)
        losses = self.trainer_model.train_on_batch(X, generator_labels)
        if not isinstance(losses, list):
            losses = [losses]
        loss_names = ['train_' + ln for ln in self.loss_names]

        self.X_train_batch = X
        self.seq_imfiles_train_batch = seq_imfiles
        # _make_generator_labels applies the augmentation transform matrix, so let's use that when we print an image
        if isinstance(generator_labels, list):
            self.Y_train_batch = generator_labels[0]
        else:
            self.Y_train_batch = generator_labels # single output
        assert len(losses) == len(loss_names)
        self.iter_count += 1

        return losses, loss_names

    def test_batches(self):
        n_test_batches = int(np.ceil(self.get_n_test() / float(self.batch_size)))
        self.logger.debug('Validating {} batches...'.format(n_test_batches))

        for bi in range(n_test_batches):
            X, Y, seq_imfiles = next(self.test_gen)
            generator_labels = self._make_model_targets(Y)

            losses = self.trainer_model.evaluate(X, generator_labels, verbose=False)
            if not isinstance(losses, list):
                losses = [losses]
            if bi == 0:
                test_losses = np.asarray(losses) / float(n_test_batches)
            else:
                test_losses += np.asarray(losses) / float(n_test_batches)

        loss_names = ['valid_' + ln for ln in self.loss_names]
        self.X_test_batch = X
        self.seq_imfiles_test_batch = seq_imfiles
        if isinstance(generator_labels, list):
            self.Y_test_batch = generator_labels[0]
        else:
            self.Y_test_batch = generator_labels
        return test_losses.tolist(), loss_names

    def save_exp_info(self, exp_dir, figures_dir, models_dir, logs_dir):
        super(TimelapseFramesExperiment, self).save_exp_info(exp_dir, figures_dir, models_dir, logs_dir)

    def save_models(self, epoch, iter_count=None):
        super(TimelapseFramesExperiment, self).save_models(epoch, iter_count=iter_count)

    def print_models(self, save_figs=False, figs_dir=None):
        super(TimelapseFramesExperiment, self)._print_models(save_figs=save_figs, figs_dir=figs_dir)

    def load_models(self, load_epoch=None, stop_on_missing=True, init_layers=False):
        start_epoch = super(TimelapseFramesExperiment, self).load_models(load_epoch)

        self.epoch_count = start_epoch
        return start_epoch

    def update_epoch_count(self, e):
        self.epoch_count += 1
        return 0



