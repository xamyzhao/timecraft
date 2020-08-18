import argparse
import json
import os
import sys

import numpy as np

from config.watercolors_configs import watercolors_data_configs
from config.digital_configs import digital_data_configs

from src import experiment_engine
from src.TimelapseFramesPredictor import TimelapseFramesExperiment
from src.TimelapseSequence import TimelapseSequencePredictor

if __name__ == '__main__':
    np.random.seed(17)

    ap = argparse.ArgumentParser()
    # common params
    ap.add_argument('exp_type', nargs='*', type=str, help='one of [tlf, tls] for per-frame or sequential models respectively')
    ap.add_argument('-g', '--gpu', nargs='*', type=int, help='unmapped gpu to use (i.e. use 3 for gpu 0 on ephesia)',
                    default=[1])
    ap.add_argument('-bs', '--batch_size', nargs='?', type=int, default=16)
    ap.add_argument('-ds', '--dataset', nargs='*', type=str, help='mtg or fish', default=None, dest='data')
    ap.add_argument('-sample_from', nargs='?', type=str,
                    help='dataset key to sample transforms from. loads a new dataset', default=None)
    ap.add_argument('-m', '--model', type=str, help='Model architecture', default=None)
    ap.add_argument('-e', '--epoch', nargs='?', help='epoch number or "latest"', default=None)
    ap.add_argument('-print_every', nargs='?', type=int,
                    help='Number of seconds between printing training batches as images', default=120)
    ap.add_argument('--lr', nargs='?', type=float, help='Learning rate', default=1e-4)

    # debugging options
    ap.add_argument('--debug', action='store_true', help='Load fewer and save more often', default=False)
    ap.add_argument('--loadn', nargs='?', type=int,
                    help='Load a specified number of training and validation examples (each)')


    ap.add_argument('--from_dir', nargs='?', default=None, help='Load experiment from dir instead of by params')
    ap.add_argument('-exp_dir', nargs='?', type=str, help='experiments directory to put each experiment in',
                    default='experiments')

    # arch params
    ap.add_argument('--arch.latent', type=int, help='Latent dimensionality e.g. 100', default=None, dest='latent')

    ap.add_argument('-pd', nargs='?', type=str, help='Dir to load painter model from', default=None)
    ap.add_argument('-pe', nargs='?', type=int, help='Painter epoch number to load', default=None)

    args = ap.parse_args()
    experiment_engine.configure_gpus(args.gpu)

    if not args.debug:
        end_epoch = 100000
    else:
        save_every_n_epochs = 4
        test_every_n_epochs = 2
        if args.epoch is not None:
            end_epoch = int(args.epoch) + 10
        else:
            end_epoch = 10

    # set dataset params
    if args.from_dir:
        with open(os.path.join(args.from_dir, 'arch_params.json'), 'r') as f:
            fromdir_arch_params = json.load(f)
        with open(os.path.join(args.from_dir, 'data_params.json'), 'r') as f:
            data_params = json.load(f)
    elif args.data:
        data_params = []
        for d_cfg in args.data:
            if 'dig' in d_cfg or 'digital' in d_cfg:
                data_params += [digital_data_configs[d_cfg]]
            else:
                data_params += [watercolors_data_configs[d_cfg]]
    else:
        print('Dataset not specified, using default dataset')
        data_params = watercolors_data_configs['watercolor-example']

    for ei, exp_type in enumerate(args.exp_type):
        if exp_type.lower() == 'tlf':
            save_every_n_epochs = 5
            test_every_n_epochs = 10
            end_epoch = 500

            if args.from_dir:
                default_arch_params = fromdir_arch_params
            else:
                named_arch_params = {
                    'cvae': {
                        'model_arch': 'cvae',
                        'latent_dim': 5,
                        'condition_by': 'concat',
                        'transform_type': 'delta',
                        'recon_fn': 'l1',
                        'recon_lambda': 0.5 / (0.1) ** 2 * 50 * 50,
                        'enc_params': {
                            'nf_enc': [16, 16, 32, 32, 64, 64],
                            'nf_dec': [64] * 8,
                            'ks': [3] * 7 + [7] * 2,
                            'n_convs_per_stage': 1,
                            'fully_conv': False,
                            'use_maxpool': False,
                            'use_upsample': False,
                            'use_residuals': False,
                            'use_skips': False,
                        },
                        'color_aug': None,  # 'sat',
                        'spatial_aug': None,  # 'less'
                    },
                }
                default_arch_params = named_arch_params['cvae']

                if not args.model is None:
                    model_arch_params = named_arch_params[args.model]
                    for k, v in model_arch_params.items():
                        default_arch_params[k] = v
                print(default_arch_params)

            if args.lr:
                default_arch_params['lr'] = args.lr

            if args.latent:
                default_arch_params['latent_dim'] = args.latent

            exp = TimelapseFramesExperiment(data_params, default_arch_params)
        elif exp_type.lower() == 'tls':
            save_every_n_epochs = 2
            test_every_n_epochs = 5
            end_epoch = 1500

            named_arch_params = {
                'cvae-pd': {   # sequential CVAE with a painter discriminator
                    'spatial_aug': None,  # 'less',
                    'n_pred_frames': [3, 5],
                    'do_alternate_sampling': True,
                    'painter_dir': 'C:/experiments/TLF_wc-batch123-pruned_procreate_cvae_nprev1_recon-l1_latent5_condby-concat_ndec8_1',
                    'painter_epoch': 200,
                    'pretrain_critic': 5,
                    'n_prev_disc_frames': 1,
                    'n_prev_disc_attns': 0,
                    'use_is_done_classifier': False,
                    'allow_empty_steps': True,
                    'n_const_disc_frames': 1,
                    'do_jitter_attn_amp': False,
                    'disc_epoch': None,
                    'do_train_sfp_on_endpoints': True,
                    'n_sfp_pred_frames': [40],
                    'use_sfp_last_recon_loss': True,
                    'use_sfp_true_last': True,
                    'n_critic_iters': 5,
                    'disc_arch': 'star-wgan',
                    'disc_enc_params': {
                        'nf_enc': [128, 128, 128, 128],  # , 256],#[32, 64, 64, 128, 128, 128],
                        'n_convs_per_stage': 0,
                        'use_maxpool': False,
                        'ks': [4] * 2 + [4] * 3,
                    }
                },
            }

            if args.from_dir:
                arch_params = fromdir_arch_params
                arch_params['exp_dir'] = args.from_dir
            else:
                arch_params = named_arch_params['cvae-pd'] # no default

            if args.pd is not None:
                arch_params['painter_dir'] = args.pd
                arch_params['painter_epoch'] = args.pe

            if not args.model is None:
                model_arch_params = named_arch_params[args.model]
                for k, v in model_arch_params.items():
                    arch_params[k] = v

            if args.lr:
                arch_params['lr'] = args.lr

            exp = TimelapseSequencePredictor(data_params, arch_params, loaded_from_dir=args.from_dir)

        prev_exp_dir = experiment_engine.run_experiment(exp=exp, run_args=args,
                                                        end_epoch=end_epoch,
                                                        save_every_n_epochs=save_every_n_epochs,
                                                        test_every_n_epochs=test_every_n_epochs)
        print('Done with experiment {}, models saved to {}'.format(exp_type, prev_exp_dir))
