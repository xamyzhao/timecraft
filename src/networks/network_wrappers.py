import tensorflow as tf
import keras

import keras.backend as K
from keras.models import Model
from keras.layers import Concatenate, Input, Lambda, Reshape


def sampling_sigma1(z_mean):
    epsilon_std = 1.0
    epsilon = K.random_normal(shape=K.shape(z_mean), mean=0.,
        stddev=epsilon_std)
    return z_mean + epsilon


def perframe_sequence_tester(
    perframe_tester_model,
    img_shape=(50, 50, 3),
    seq_len=40,
    latent_shape=(5,),
    n_const_frames=1,
    n_prev_frames = 1,
):
    inputs = [
        Input(img_shape, name='input_last_frame'),
        Input(img_shape, name='input_prev_frame')
    ]

    # first two frames of the input stack will always be first and last frame
    const_frames = inputs[:n_const_frames]
    curr_prev_frames = inputs[n_const_frames:n_const_frames + n_prev_frames]

    # a dummy input that enables us to do sampling of the latent input in the network
    z_dummy = Input(latent_shape, name='input_z_dummy')
    inputs += [z_dummy]

    painter_preds_seqs = []

    for t in range(seq_len):
        # assumes frame prediction is always the first pred, others might be KL
        painter_cond_inputs = const_frames + curr_prev_frames
        z_samp = Lambda(sampling_sigma1,
                        name=f'lambda_z_sampling_frame{t}'
                        )(z_dummy)
        painter_preds = perframe_tester_model(painter_cond_inputs + [z_samp])

        if not isinstance(painter_preds, list):
            painter_preds = [painter_preds]
        else:
            painter_preds = [painter_preds[0]] # cvae painter might return transformed, delta

        clipped_painter_frames = []
        for ppi in range(len(painter_preds)):
            pred_frame = painter_preds[ppi]
            pred_frame = Lambda(lambda x: tf.clip_by_value(x, -1., 1.), name=F'lambda_clip_frame_{ppi}_t{t}')(pred_frame)
            clipped_painter_frames.append(pred_frame)
        painter_preds = clipped_painter_frames


        # shift previous frames to make room for our new painter prediction
        curr_prev_frames = curr_prev_frames[1:] + [painter_preds[0]]

        for ppi, pp in enumerate(painter_preds):
            # give every prediction a time dimension, and concatenate it

            pp = Reshape(pp.get_shape().as_list()[1:] + [1],
                         name='reshape_t{}_pp{}'.format(t, ppi))(pp)
            if t == 0:
                painter_preds_seqs.append(pp)
            else:
                painter_preds_seqs[ppi] = Concatenate(
                    name='concat_t{}_pp{}'.format(t, ppi))(
                    [painter_preds_seqs[ppi], pp])

    return Model(inputs=inputs,
                 outputs=painter_preds_seqs,
                 name='seqlen{}_model'.format(seq_len))