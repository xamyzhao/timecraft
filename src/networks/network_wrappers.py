import os

import tensorflow as tf

from keras import Input, backend as K, Model
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


def perframe_sequence_trainer_noattn(
        conditioning_input_shapes,
        conditioning_input_names,
        input_gt_frames_shape,
        perframe_painter_model,
        seq_len,
        is_done_model=None,
        n_const_frames=1,
        do_output_disc_stack=False,
        n_prev_frames=None,
        n_prev_disc_frames=1,
        n_painter_frame_outputs=2,
):
    if n_prev_frames is None:
        n_prev_frames = seq_len - 1

    # collect conditioning inputs, which should include last frame, prev frames, prev attns
    if not isinstance(conditioning_input_shapes, list):
        conditioning_input_shapes = [conditioning_input_shapes]
    if conditioning_input_names is None:
        conditioning_input_names = ['cond_input_{}'.format(ii) for ii in range(len(conditioning_input_shapes))]

    conditioning_inputs = []
    for ii, input_shape in enumerate(conditioning_input_shapes):
        conditioning_inputs.append(Input(input_shape, name=conditioning_input_names[ii]))

    input_gt_frames = Input(input_gt_frames_shape, name='input_gt_frames')

    inputs = conditioning_inputs + [input_gt_frames]

    prev_frames = conditioning_inputs[1:]
    gt_frames = Reshape(input_gt_frames_shape, name='reshape_gt')(input_gt_frames)


    # first two frames of the input stack will always be first and last frame
    const_frames = conditioning_inputs[:n_const_frames]
    curr_prev_frames = conditioning_inputs[n_const_frames:n_const_frames + n_prev_frames]
    curr_prev_attn_maps = conditioning_inputs[n_const_frames + n_prev_frames:]
    # first two frames of the input stack will always be first and last frame
    last_frame_seq = Lambda(
        lambda x: tf.tile(K.expand_dims(x, axis=-1), [1, 1, 1, 1, seq_len]),
        name='lambda_tile_slice_last_frame_seq')(const_frames[0])

    director_preds_seqs = []
    painter_preds_seqs = []

    for t in range(seq_len):

        # cvae
        painter_cond_inputs = const_frames + curr_prev_frames

        # provide the true frame as input to the autoencoding branch of the painter
        gt_frame = Lambda(lambda x: tf.gather(x, t, axis=-1),
                          name='lambda_slice_gt_frames_t{}'.format(t))(gt_frames)
        painter_ae_inputs = painter_cond_inputs + [gt_frame]
        # TODO: painter network currently expects cond input first, then ae input
        painter_preds = perframe_painter_model(painter_ae_inputs + painter_cond_inputs)

        if not isinstance(painter_preds, list):
            painter_preds = [painter_preds]

        clipped_painter_frames = []
        for ppi in range(n_painter_frame_outputs):
            pred_frame = painter_preds[ppi]
            # TODO: get rid of hardcoding of range (e.g. if we are not normalizing)
            pred_frame = Lambda(lambda x: tf.clip_by_value(x, -1., 1.), name=F'lambda_clip_frame_{ppi}_t{t}')(pred_frame)
            clipped_painter_frames.append(pred_frame)
        painter_preds = clipped_painter_frames + painter_preds[n_painter_frame_outputs:]

        if n_prev_disc_frames > 0:
            prev_frames = curr_prev_frames[-n_prev_disc_frames:]

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

        ####### compile information needed for conditional discriminators #######
        if (do_output_disc_stack):
            if n_prev_disc_frames > 0:
                prev_frames = Reshape(prev_frames.get_shape().as_list()[1:] + [1],
                                name='exptdim_t{}_prevframes'.format(t))(prev_frames)
                if t == 0:
                    prev_frames_seq = prev_frames
                else:
                    prev_frames_seq = Concatenate(
                        name='concat_t{}_prevframe'.format(t))([prev_frames_seq, prev_frames])

        ####### compile information needed for our "is done" classifier
        if is_done_model is not None:
            # completed painting, and current prediction
            is_done_inputs = Concatenate(axis=-1)([const_frames[0], pred_frame])

            # run the classifier
            is_done_pred = is_done_model(is_done_inputs)

            # add a time dimension
            is_done_pred = Reshape(is_done_pred.get_shape().as_list()[1:] + [1],
                                 name='exptdim_t{}_isdone'.format(t))(is_done_pred)
            if t == 0:
                is_done_preds_seq = is_done_pred
            else:
                is_done_preds_seq = Concatenate(
                    name='concat_t{}_isdone'.format(t), axis=-1)([is_done_preds_seq, is_done_pred])

    outputs = director_preds_seqs + painter_preds_seqs

    # if we are using a discriminator, output the discriminator input stacks at the end so we can evaluate the scores
    if do_output_disc_stack:
        disc_inputs = [last_frame_seq]
        if n_prev_disc_frames > 0:
            disc_inputs.append(prev_frames_seq)
        # discriminator on attention map
        director_disc_stack = Concatenate(axis=-2, name='concat_director_disc_stack')(disc_inputs + [director_preds_seqs[0]])
        outputs += [director_disc_stack]

    return Model(inputs=inputs,
                 outputs=outputs,
                 name='seqlen{}_perframe_trainer_model'.format(seq_len))


def perframe_sampling_sequence_trainer_noattn(
        conditioning_input_shapes,
        conditioning_input_names,
        perframe_painter_model,
        seq_len,
        n_prev_frames=1,
        n_const_frames=1,
        n_prev_disc_frames=1,
        n_const_disc_frames=1,
        n_painter_frame_outputs=2,
        painter_latent_shape=None,
        make_painter_disc_stack=False,
):
    if n_prev_frames is None:
        n_prev_frames = seq_len - 1

    if not isinstance(conditioning_input_shapes, list):
        conditioning_input_shapes = [conditioning_input_shapes]
    if conditioning_input_names is None:
        conditioning_input_names = ['cond_input_{}'.format(ii) for ii in range(len(conditioning_input_shapes))]

    conditioning_inputs = []
    for ii, input_shape in enumerate(conditioning_input_shapes):
        conditioning_inputs.append(Input(input_shape, name=conditioning_input_names[ii]))
    inputs = [ci for ci in conditioning_inputs]

    # these inputs are required so we can use keras sampling..still havent
    # figured out how to do it without an initial input first

    if painter_latent_shape is not None:
        # if the painter uses a CVAE
        dummy_z_p_input = Input(painter_latent_shape, name='input_z_p_dummy')
        inputs += [dummy_z_p_input]

    # first two frames of the input stack will always be first and last frame
    const_frames = conditioning_inputs[:n_const_frames]
    curr_prev_frames = conditioning_inputs[n_const_frames:n_const_frames + n_prev_frames]
    curr_prev_attn_maps = conditioning_inputs[n_const_frames + n_prev_frames:]


    # first two frames of the input stack will always be first and last frame
    last_frame_seq = Lambda(
        lambda x: tf.tile(K.expand_dims(x, axis=-1), [1, 1, 1, 1, seq_len]),
        name='lambda_tile_slice_last_frame_seq')(const_frames[0])

    painter_preds_seqs = []
    painter_deltas_seq = []
    for t in range(seq_len):
        ####### compile information needed for conditional discriminators #######
        if (make_painter_disc_stack) and n_prev_disc_frames > 0:
            # both discriminators will probably require prev frames
            # TODO: remove hardcoding that assumes 1 const frame
            prev_frames = Reshape(const_frames[0].get_shape().as_list()[1:] + [1],
                            name='exptdim_t{}_prevframes'.format(t))(curr_prev_frames[-1])
            if t == 0:
                prev_frames_seq = prev_frames
            else:
                prev_frames_seq = Concatenate(
                    name='concat_t{}_prevframe'.format(t))([prev_frames_seq, prev_frames])

        # cvae
        painter_cond_inputs = const_frames + curr_prev_frames

        # sample from the painter's prior instead
        painter_preds = perframe_painter_model(painter_cond_inputs + [dummy_z_p_input])

        if not isinstance(painter_preds, list):
            painter_preds = [painter_preds]

        # assumes frame prediction is always the first pred, others might be KL
        clipped_painter_frames = []
        for ppi in range(n_painter_frame_outputs):
            curr_pred_frame = painter_preds[ppi]
            curr_pred_name = os.path.basename(os.path.dirname(curr_pred_frame.name))
            # TODO: get rid of hardcoding of range (e.g. if we are not normalizing)
            curr_pred_frame = Lambda(lambda x: tf.clip_by_value(x, -1., 1.),
                name=F'clip_pp{ppi}_{curr_pred_name}_t{t}')(curr_pred_frame)
            clipped_painter_frames.append(curr_pred_frame)
        painter_preds = clipped_painter_frames + painter_preds[n_painter_frame_outputs:]
        curr_pred_frame = painter_preds[0]

        curr_prev_frames = curr_prev_frames[1:] + [curr_pred_frame]

        ########### compile predictions into sequences in time ######################
        # hacky, but if the painter predicts a delta, we only want the first few recon outputs (the transformed frame)
        # and we can ignore the following output (the transform/delta)
        for ppi, pp in enumerate(painter_preds[:n_painter_frame_outputs]):
            # give every prediction a time dimension, and concatenate it
            ppn = os.path.basename(os.path.dirname(pp.name))

            pp = Reshape(pp.get_shape().as_list()[1:] + [1],
                         name='reshape_t{}_pp{}'.format(t, ppi))(pp)
            if t == 0:
                painter_preds_seqs.append(pp)
            else:
                painter_preds_seqs[ppi] = Concatenate(
                    name='concat_t{}_pp-{}'.format(t, ppn))(
                    [painter_preds_seqs[ppi], pp])

    outputs = painter_preds_seqs

    # if we are using a discriminator, output the discriminator input stacks at the end so we can evaluate the scores
    if make_painter_disc_stack:
        disc_inputs = []
        if n_const_disc_frames > 0:
            disc_inputs.append(last_frame_seq)

        if n_prev_disc_frames > 0:
            disc_inputs.append(prev_frames_seq)

        painter_disc_stack = Concatenate(axis=-2, name='concat_painter_disc_stack')(disc_inputs + [painter_preds_seqs[0]])
        outputs += [painter_disc_stack]

    return Model(inputs=inputs,
                 outputs=outputs,
                 name='seqlen{}_perframe_sampling_trainer_model'.format(seq_len))