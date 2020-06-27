import os

import argparse
import cv2
import functools
import numpy as np
from keras.models import load_model
import keras.backend as K
import tensorflow as tf

from src.utils import utils
from src.networks import network_wrappers

SCALE_TO_SHAPE = (126, 168)
MODEL_INPUT_SHAPE = (50, 50)

if __name__ == '__main__':
    np.random.seed(17)

    ap = argparse.ArgumentParser()

    ap.add_argument('input_img_file', type=str, help='path to the image file')
    ap.add_argument('-g', '--gpu', nargs='*', type=int, help='gpu ID to use',
                    default=[0])

    args = ap.parse_args()

    # set gpu id and tf settings
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(g) for g in args.gpu])
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    K.tensorflow_backend.set_session(tf.Session(config=config))

    if not os.path.isfile(args.input_img_file):
        raise Exception(f"Could not find input file {args.input_img_file}")
    im = cv2.imread(args.input_img_file)

    # resize the image roughly the way we preprocessed youtube video
    # frames for training in the paper: scale each frame to
    # roughly 126 × 168, and then take a center crop
    max_scale_factor = max(
        SCALE_TO_SHAPE[0] / float(im.shape[0]),
        SCALE_TO_SHAPE[1] / float(im.shape[1])
    )

    im = cv2.resize(im, None, fx=max_scale_factor, fy=max_scale_factor)
    im_center = (im.shape[0] / 2., im.shape[1] / 2.)
    crop_start_row = int(im_center[0] - MODEL_INPUT_SHAPE[0] / 2.)
    crop_start_col = int(im_center[1] - MODEL_INPUT_SHAPE[1] / 2.)
    im = im[
         crop_start_row : crop_start_row + MODEL_INPUT_SHAPE[0],
         crop_start_col : crop_start_col + MODEL_INPUT_SHAPE[1]
         ]
    im = utils.normalize(im / 255.)  # normalize to range [-1, 1]


    frame_predictor_model = load_model(
        '../trained_models/ours_watercolor_epoch300.h5',
        custom_objects={
            'Blur_Downsample': functools.partial(
                utils.Blur_Downsample,
                n_chans=3),
        }
    )

    print('Per-frame predictor')
    frame_predictor_model.summary()

    # create a sequential model
    video_predictor_model = network_wrappers.perframe_sequence_tester(
        frame_predictor_model
    )

    print('Video predictor')
    video_predictor_model.summary()

    n_samples = 5
    for i in range(n_samples):
        pred_vid = video_predictor_model.predict(
            [im[np.newaxis], np.ones((1,) + im.shape), np.zeros((1, 5))])
        print(f'Predicted video shape: {pred_vid.shape}')

        pred_vid_im = utils.visualize_video(
            pred_vid[0], normalized=True)

        cv2.imshow(f'Video sample {i+1}', pred_vid_im)
    cv2.waitKey()
