import sys

import keras.initializers as keras_init
from keras.layers import Input, Flatten, Dense, Reshape, Lambda, Activation, Dropout
from keras.layers.convolutional import Conv2D, UpSampling2D, Conv2DTranspose, Cropping2D, ZeroPadding2D
from keras.layers.convolutional import Conv3D, UpSampling3D, Conv3DTranspose, MaxPooling3D
from keras.layers.merge import Add, Concatenate, Multiply
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import keras.backend as K

import numpy as np
from src.utils import network_utils
from keras.layers.advanced_activations import LeakyReLU
import tensorflow as tf

def transform_encoder_model(input_shapes, input_names=None,
                            latent_shape=(50,),
                            model_name='encoder',
                            enc_params=None,
                            ):
    '''
    Generic encoder for a stack of inputs

    :param input_shape:
    :param latent_shape:
    :param model_name:
    :param enc_params:
    :return:
    '''
    if not isinstance(input_shapes, list):
        input_shapes = [input_shapes]

    if input_names is None:
        input_names = ['input_{}'.format(ii) for ii in range(len(input_shapes))]

    inputs = []
    for ii, input_shape in enumerate(input_shapes):
        inputs.append(Input(input_shape, name='input_{}'.format(input_names[ii])))
    if len(inputs) > 1:
        inputs_stacked = Concatenate(name='concat_inputs', axis=-1)(inputs)
    else:
        inputs_stacked = inputs[0]
    input_stack_shape = inputs_stacked.get_shape().as_list()[1:]
    n_dims = len(input_stack_shape) - 1

    x_transform_enc = encoder(
        x=inputs_stacked,
        img_shape=input_stack_shape,
        conv_chans=enc_params['nf_enc'],
        n_convs_per_stage=enc_params['n_convs_per_stage'] if 'n_convs_per_stage' in enc_params else 1,
        use_residuals=enc_params['use_residuals'] if 'use_residuals' in enc_params else False,
        use_maxpool=enc_params['use_maxpool'] if 'use_maxpool' in enc_params else False,
        kernel_initializer=enc_params['kernel_initializer'] if 'kernel_initializer' in enc_params else None,
        bias_initializer=enc_params['bias_initializer'] if 'bias_initializer' in enc_params else None,
        prefix="cvae"
    )

    latent_size = np.prod(latent_shape)

    # the last layer in the basic encoder will be a convolution, so we should activate after it
    x_transform_enc = LeakyReLU(0.2)(x_transform_enc)
    x_transform_enc = Flatten()(x_transform_enc)

    z_mean = Dense(latent_size, name='latent_mean',
        kernel_initializer=keras_init.RandomNormal(mean=0., stddev=0.00001))(x_transform_enc)
    z_logvar = Dense(latent_size, name='latent_logvar',
                    bias_initializer=keras_init.RandomNormal(mean=-2., stddev=1e-10),
                    kernel_initializer=keras_init.RandomNormal(mean=0., stddev=1e-10),
                )(x_transform_enc)

    return Model(inputs=inputs, outputs=[z_mean, z_logvar], name=model_name)


def encoder(x, img_shape,
            conv_chans=None,
            n_convs_per_stage=1,
            min_h=5, min_c=None,
            prefix='',
            ks=3,
            return_skips=False, use_residuals=False, use_maxpool=False,
            kernel_initializer=None, bias_initializer=None):
    skip_layers = []
    concat_skip_sizes = []
    n_dims = len(img_shape) - 1  # assume img_shape includes spatial dims, followed by channels

    if conv_chans is None:
        n_convs = int(np.floor(np.log2(img_shape[0] / min_h)))
        conv_chans = [min_c * 2] * (n_convs - 1) + [min_c]
    elif not type(conv_chans) == list:
        n_convs = int(np.floor(np.log2(img_shape[0] / min_h)))
        conv_chans = [conv_chans] * (n_convs - 1) + [min_c]
    else:
        n_convs = len(conv_chans)

    if isinstance(ks, list):
        assert len(ks) == (n_convs + 1)  # specify for each conv, as well as the last one
    else:
        ks = [ks] * (n_convs + 1)


    for i in range(len(conv_chans)):
        #if n_convs_per_stage is not None and n_convs_per_stage > 1 or use_maxpool and n_convs_per_stage is not None:
        for ci in range(n_convs_per_stage):
            x = myConv(nf=conv_chans[i], ks=ks[i], strides=1, n_dims=n_dims,
                       kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                       prefix='{}_enc'.format(prefix),
                       suffix='{}_{}'.format(i, ci + 1))(x)

            if ci == 0 and use_residuals:
                residual_input = x
            elif ci == n_convs_per_stage - 1 and use_residuals:
                x = Add(name='{}_enc_{}_add_residual'.format(prefix, i))([residual_input, x])

            x = LeakyReLU(0.2, name='{}_enc_leakyrelu_{}_{}'.format(prefix, i, ci + 1))(x)

        if return_skips:
            skip_layers.append(x)
            concat_skip_sizes.append(np.asarray(x.get_shape().as_list()[1:-1]))

        if use_maxpool and i < len(conv_chans) - 1:
            # changed 5/30/19, don't pool after our last conv
            x = myPool(n_dims=n_dims, prefix=prefix, suffix=i)(x)
        else:
            x = myConv(conv_chans[i], ks=ks[i], strides=2, n_dims=n_dims,
                       kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                       prefix='{}_enc'.format(prefix), suffix=i)(x)

            # don't activate right after a maxpool, it makes no sense
            if i < len(conv_chans) - 1:  # no activation on last convolution
                x = LeakyReLU(0.2, name='{}_enc_leakyrelu_{}'.format(prefix, i))(x)

    if min_c is not None and min_c > 0:
        # if the last number of channels is specified, convolve to that
        if n_convs_per_stage is not None and n_convs_per_stage > 1:
            for ci in range(n_convs_per_stage):
                # TODO: we might not have enough ks for this
                x = myConv(min_c, ks=ks[-1], n_dims=n_dims, strides=1,
                           kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                           prefix='{}_enc'.format(prefix), suffix='last_{}'.format(ci + 1))(x)

                if ci == 0 and use_residuals:
                    residual_input = x
                elif ci == n_convs_per_stage - 1 and use_residuals:
                    x = Add(name='{}_enc_{}_add_residual'.format(prefix, 'last'))([residual_input, x])
                x = LeakyReLU(0.2, name='{}_enc_leakyrelu_last'.format(prefix))(x)

        x = myConv(min_c, ks=ks[-1], strides=1, n_dims=n_dims,
                   kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                   prefix='{}_enc'.format(prefix),
                   suffix='_last')(x)

        if return_skips:
            skip_layers.append(x)
            concat_skip_sizes.append(np.asarray(x.get_shape().as_list()[1:-1]))

    if return_skips:
        return x, skip_layers, concat_skip_sizes
    else:
        return x


def transformer_concat_model(conditioning_input_shapes, conditioning_input_names=None,
                             output_shape=None,
                             model_name='CVAE_transformer',
                             transform_latent_shape=(100,),
                             enc_params=None,
                             condition_on_image=True,
                             n_concat_scales=3,
                             transform_activation=None, clip_output_range=None,
                             source_input_idx=None,
                             ):
    # collect conditioning inputs, and concatentate them into a stack
    if not isinstance(conditioning_input_shapes, list):
        conditioning_input_shapes = [conditioning_input_shapes]
    if conditioning_input_names is None:
        conditioning_input_names = ['cond_input_{}'.format(ii) for ii in range(len(conditioning_input_shapes))]

    conditioning_inputs = []
    for ii, input_shape in enumerate(conditioning_input_shapes):
        conditioning_inputs.append(Input(input_shape, name=conditioning_input_names[ii]))

    if len(conditioning_inputs) > 1:
        conditioning_input_stack = Concatenate(name='concat_cond_inputs', axis=-1)(conditioning_inputs)
    else:
        conditioning_input_stack = conditioning_inputs[0]

    conditioning_input_shape = tuple(conditioning_input_stack.get_shape().as_list()[1:])
    n_dims = len(conditioning_input_shape) - 1

    # we will always give z as a flattened vector
    z_input = Input((np.prod(transform_latent_shape),), name='z_input')

    # determine what we should apply the transformation to
    if source_input_idx is None:
        # the image we want to transform is exactly the input of the conditioning branch
        x_source = conditioning_input_stack
        source_input_shape = conditioning_input_shape
    else:
        # slice conditioning input to get the single source im that we will apply the transform to
        source_input_shape = conditioning_input_shapes[source_input_idx]
        x_source = conditioning_inputs[source_input_idx]

    # assume the output is going to be the transformed source input, so it should be the same shape
    if output_shape is None:
        output_shape = source_input_shape

    layer_prefix = 'color'
    decoder_output_shape = output_shape

    if condition_on_image: # assume we always condition by concat since it's better than other forms
        # simply concatenate the conditioning stack (at various scales) with the decoder volumes
        include_fullres = True

        concat_decoder_outputs_with = [None] * len(enc_params['nf_dec'])
        concat_skip_sizes = [None] * len(enc_params['nf_dec'])

        # make sure x_I is the same shape as the output, including in the channels dimension
        if not np.all(output_shape <= conditioning_input_shape):
            tile_factor = [int(round(output_shape[i] / conditioning_input_shape[i])) for i in
                           range(len(output_shape))]
            print('Tile factor: {}'.format(tile_factor))
            conditioning_input_stack = Lambda(lambda x: tf.tile(x, [1] + tile_factor), name='lambda_tile_cond_input')(conditioning_input_stack)

        # downscale the conditioning inputs by the specified number of times
        xs_downscaled = [conditioning_input_stack]
        for si in range(n_concat_scales):
            curr_x_scaled = network_utils.Blur_Downsample(
                n_chans=conditioning_input_shape[-1], n_dims=n_dims,
                do_blur=True,
                name='downsample_scale-1/{}'.format(2**(si + 1))
            )(xs_downscaled[-1])
            xs_downscaled.append(curr_x_scaled)

        if not include_fullres:
            xs_downscaled = xs_downscaled[1:]  # exclude the full-res volume

        print('Including downsampled input sizes {}'.format([x.get_shape().as_list() for x in xs_downscaled]))

        # the smallest decoder volume will be the same as the smallest encoder volume, so we need to make sure we match volume sizes appropriately
        n_enc_scales = len(enc_params['nf_enc'])
        n_ds = len(xs_downscaled)
        concat_decoder_outputs_with[n_enc_scales - n_ds + 1:n_enc_scales] = list(reversed(xs_downscaled))
        concat_skip_sizes[n_enc_scales - n_ds + 1:n_enc_scales] = list(reversed(
            [np.asarray(x.get_shape().as_list()[1:-1]) for x in xs_downscaled if
             x is not None]))

    else:
        # just ignore the conditioning input
        concat_decoder_outputs_with = None
        concat_skip_sizes = None


    if 'ks' not in enc_params:
        enc_params['ks'] = 3


    # determine what size to reshape the latent vector to
    reshape_encoding_to = get_encoded_shape(
        img_shape=conditioning_input_shape,
        conv_chans=enc_params['nf_enc'],
    )

    x_enc = Dense(np.prod(reshape_encoding_to), name='dense_encoding_to_vol')(z_input)
    x_enc = LeakyReLU(0.2)(x_enc)

    x_enc = Reshape(reshape_encoding_to)(x_enc)
    print('Decoder starting shape: {}'.format(reshape_encoding_to))

    x_transformation = decoder(
        x_enc, decoder_output_shape,
        encoded_shape=reshape_encoding_to,
        prefix='{}_dec'.format(layer_prefix),
        conv_chans=enc_params['nf_dec'],
        ks=enc_params['ks'] if 'ks' in enc_params else 3,
        n_convs_per_stage=enc_params['n_convs_per_stage'] if 'n_convs_per_stage' in enc_params else 1,
        use_upsample=enc_params['use_upsample'] if 'use_upsample' in enc_params else False,
        kernel_initializer=enc_params['kernel_initializer'] if 'kernel_initializer' in enc_params else None,
        bias_initializer=enc_params['bias_initializer'] if 'bias_initializer' in enc_params else None,
        include_skips=concat_decoder_outputs_with,
        target_vol_sizes=concat_skip_sizes
    )

    if transform_activation is not None:
        x_transformation = Activation(
            transform_activation,
            name='activation_transform_{}'.format(transform_activation))(x_transformation)

        if transform_activation=='tanh':
            # TODO: maybe move this logic
            # if we are learning a colro delta with a tanh, make sure to multiply it by 2
            x_transformation = Lambda(lambda x: x * 2, name='lambda_scale_tanh')(x_transformation)

    im_out = Add()([x_source, x_transformation])

    if clip_output_range is not None:
        im_out = Lambda(lambda x: tf.clip_by_value(x, clip_output_range[0], clip_output_range[1]),
            name='lambda_clip_output_{}-{}'.format(clip_output_range[0], clip_output_range[1]))(im_out)


    return Model(inputs=conditioning_inputs + [z_input], outputs=[im_out, x_transformation], name=model_name)


def decoder(x,
            output_shape,
            encoded_shape,
            conv_chans=None,
            min_h=5, min_c=4,
            prefix='vte_dec',
            n_convs_per_stage=1,
            include_dropout=False,
            ks=3,
            include_skips=None,
            use_residuals=False,
            use_upsample=False,
            use_batchnorm=False,
            target_vol_sizes=None,
            n_samples=1,
            kernel_initializer=None, bias_initializer=None,
            ):
    n_dims = len(output_shape) - 1
    if conv_chans is None:
        n_convs = int(np.floor(np.log2(output_shape[0] / min_h)))
        conv_chans = [min_c * 2] * n_convs
    elif not type(conv_chans) == list:
        n_convs = int(np.floor(np.log2(output_shape[0] / min_h)))
        conv_chans = [conv_chans] * n_convs
    elif type(conv_chans) == list:
        n_convs = len(conv_chans)

    if isinstance(ks, list):
        assert len(ks) == (n_convs + 1)  # specify for each conv, as well as the last one
    else:
        ks = [ks] * (n_convs + 1)

    print('Decoding with conv filters {}'.format(conv_chans))
    # compute default sizes that we want on the way up, mainly in case we have more convs than stages
    # and we upsample past the output size
    if n_dims == 2:
        # just upsample by a factor of 2 and then crop the final volume to the desired volume
        default_target_vol_sizes = np.asarray(
            [(int(encoded_shape[0] * 2. ** (i + 1)), int(encoded_shape[1] * 2. ** (i + 1)))
             for i in range(n_convs - 1)] + [output_shape[:2]])
    else:
        print(output_shape)
        print(encoded_shape)
        # just upsample by a factor of 2 and then crop the final volume to the desired volume
        default_target_vol_sizes = np.asarray(
            [(
                min(output_shape[0], int(encoded_shape[0] * 2. ** (i + 1))),
                min(output_shape[1], int(encoded_shape[1] * 2. ** (i + 1))),
                min(output_shape[2], int(encoded_shape[2] * 2. ** (i + 1))))
            for i in range(n_convs - 1)] + [output_shape[:3]])

    # automatically stop when we reach the desired image shape
    for vi, vs in enumerate(default_target_vol_sizes):
        if np.all(vs >= output_shape[:-1]):
            default_target_vol_sizes[vi] = output_shape[:-1]
    print('Automatically computed target output sizes: {}'.format(default_target_vol_sizes))

    if target_vol_sizes is None:
        target_vol_sizes = default_target_vol_sizes
    else:
        print('Target concat vols to match shapes to: {}'.format(target_vol_sizes))

        # TODO: check that this logic makes sense for more convs
        # fill in any Nones that we might have in our target_vol_sizes
        filled_target_vol_sizes = [None] * len(target_vol_sizes)
        for i in range(n_convs):
            if i < len(target_vol_sizes) and target_vol_sizes[i] is not None:
                filled_target_vol_sizes[i] = target_vol_sizes[i]
        target_vol_sizes = filled_target_vol_sizes

    if include_skips is not None:
        print('Concatentating padded/cropped shapes {} with skips {}'.format(target_vol_sizes, include_skips))

    curr_shape = np.asarray(encoded_shape[:n_dims])
    for i in range(n_convs):
        print(target_vol_sizes[i])
        if i < len(target_vol_sizes) and target_vol_sizes[i] is not None:
            x = _pad_or_crop_to_shape(x, curr_shape, target_vol_sizes[i])
            curr_shape = np.asarray(target_vol_sizes[i])  # we will upsample first thing next stage

        # if we want to concatenate with another volume (e.g. from encoder, or a downsampled input)...
        if include_skips is not None and i < len(include_skips) and include_skips[i] is not None:
            x_shape = x.get_shape().as_list()
            skip_shape = include_skips[i].get_shape().as_list()

            print('Attempting to concatenate current layer {} with previous skip connection {}'.format(x_shape, skip_shape))
            # input size might not match in time dimension, so just tile it
            if n_samples > 1:
                tile_factor = [1] + [n_samples] + [1] * (len(x_shape)-1)
                print('Tiling by {}'.format(tile_factor))
                print(target_vol_sizes[i])
                skip = Lambda(lambda y: K.expand_dims(y, axis=1))(include_skips[i])
                skip = Lambda(lambda y:tf.tile(y, tile_factor), name='{}_lambdatilesamples_{}'.format(prefix,i),
                    output_shape=[n_samples] + skip_shape[1:]
                    )(skip)
                skip = Lambda(lambda y:tf.reshape(y, [-1] + skip_shape[1:]), output_shape=skip_shape[1:])(skip)
            else:
                skip = include_skips[i]

            x = Concatenate(axis=-1, name='{}_concatskip_{}'.format(prefix, i))([x, skip])

        for ci in range(n_convs_per_stage):
            x = myConv(conv_chans[i],
                       ks=ks[i],
                       strides=1,
                       n_dims=n_dims,
                       prefix=prefix,
                       suffix='{}_{}'.format(i, ci + 1),
                       kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                       )(x)
            if use_batchnorm: # TODO: check to see if this should go before the residual
                x = BatchNormalization()(x)
            # if we want residuals, store them here
            if ci == 0 and use_residuals:
                residual_input = x
            elif ci == n_convs_per_stage - 1 and use_residuals:
                x = Add(name='{}_{}_add_residual'.format(prefix, i))([residual_input, x])
            x = LeakyReLU(0.2,
                          name='{}_leakyrelu_{}_{}'.format(prefix, i, ci + 1))(x)

        if include_dropout and i < 2:
            x = Dropout(0.3)(x)

        # if we are not at the output resolution yet, upsample or do a transposed convolution
        if not np.all(curr_shape == output_shape[:len(curr_shape)]):
            if not use_upsample:
                # if we have convolutional filters left at the end, just apply them at full resolution
                x = myConvTranspose(conv_chans[i], n_dims=n_dims,
                                    ks=ks[i], strides=2,
                                    prefix=prefix, suffix=i,
                                    kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                    )(x)
                if use_batchnorm:
                    x = BatchNormalization()(x)
                x = LeakyReLU(0.2, name='{}_leakyrelu_{}'.format(prefix, i))(x)  # changed 5/15/2018, will break old models
            else:
                x = myUpsample(size=2, n_dims=n_dims, prefix=prefix, suffix=i)(x)
            curr_shape *= 2

    # last stage of convolutions, no more upsampling
    x = myConv(output_shape[-1], ks=ks[-1], n_dims=n_dims,
               strides=1,
               prefix=prefix,
               suffix='final',
               kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
               )(x)

    return x

def _pad_or_crop_to_shape(x, in_shape, tgt_shape):
    if len(in_shape) == 2:
        '''
        in_shape, tgt_shape are both 2x1 numpy arrays
        '''
        in_shape = np.asarray(in_shape)
        tgt_shape = np.asarray(tgt_shape)
        print('Padding input from {} to {}'.format(in_shape, tgt_shape))
        im_diff = in_shape - tgt_shape
        if im_diff[0] < 0:
            pad_amt = (int(np.ceil(abs(im_diff[0])/2.0)), int(np.floor(abs(im_diff[0])/2.0)))
            x = ZeroPadding2D( (pad_amt, (0,0)) )(x)
        if im_diff[1] < 0:
            pad_amt = (int(np.ceil(abs(im_diff[1])/2.0)), int(np.floor(abs(im_diff[1])/2.0)))
            x = ZeroPadding2D( ((0,0), pad_amt) )(x)

        if im_diff[0] > 0:
            crop_amt = (int(np.ceil(im_diff[0]/2.0)), int(np.floor(im_diff[0]/2.0)))
            x = Cropping2D( (crop_amt, (0,0)) )(x)
        if im_diff[1] > 0:
            crop_amt = (int(np.ceil(im_diff[1]/2.0)), int(np.floor(im_diff[1]/2.0)))
            x = Cropping2D( ((0,0),crop_amt) )(x)
        return x
    else:
        return _pad_or_crop_to_shape_3D(x, in_shape, tgt_shape)


def myConvTranspose(nf, n_dims, prefix=None, suffix=None, ks=3, strides=1,
                    kernel_initializer=None, bias_initializer=None):
    if kernel_initializer is None:
        kernel_initializer = 'glorot_uniform' # keras default for conv kernels
    if bias_initializer is None:
        bias_initializer = 'zeros' # default for keras conv
    # wrapper for 2D and 3D conv
    if n_dims == 2:
        if not isinstance(strides, tuple):
            strides = (strides, strides)
        return Conv2DTranspose(nf, kernel_size=ks, padding='same', strides=strides,
                               kernel_initializer=kernel_initializer,
                               bias_initializer=bias_initializer,
                      name='_'.join([
                          str(part) for part in [prefix, 'conv2Dtrans', suffix]  # include prefix and suffix if they exist
                          if part is not None and len(str(part)) > 0]))
    elif n_dims == 3:
        if not isinstance(strides, tuple):
            strides = (strides, strides, strides)
        return Conv3DTranspose(nf, kernel_size=ks, padding='same', strides=strides,
                               kernel_initializer=kernel_initializer,
                               bias_initializer=bias_initializer,
                      name='_'.join([
                          str(part) for part in [prefix, 'conv3Dtrans', suffix]  # include prefix and suffix if they exist
                          if part is not None and len(str(part)) > 0]))

def myUpsample(n_dims, size=2, prefix=None, suffix=None):
    if n_dims == 2:
        if not isinstance(size, tuple):
            size = (size, size)

        return UpSampling2D(size=size,
                            name='_'.join([
                                str(part) for part in [prefix, 'upsamp2D', suffix]  # include prefix and suffix if they exist
                                if part is not None and len(str(part)) > 0]))
    elif n_dims == 3:
        if not isinstance(size, tuple):
            size = (size, size, size)

        return UpSampling3D(size=size,
                            name='_'.join([
                                str(part) for part in [prefix, 'upsamp3D', suffix]  # include prefix and suffix if they exist
                                if part is not None and len(str(part)) > 0]))



def get_encoded_shape( img_shape, min_c = None, conv_chans = None, n_convs = None):
    if n_convs is None:
        n_convs = len(conv_chans)
        min_c = conv_chans[-1]
    encoded_shape = tuple([int(np.ceil(s/2. ** n_convs)) for s in img_shape[:-1]] + [min_c])
    #encoded_shape = (int(np.ceil(img_shape[0]/2.**n_convs)), int(np.ceil(img_shape[1]/2.**n_convs)), min_c)
    print('Encoded shape for img {} with {} convs is {}'.format(img_shape, n_convs, encoded_shape))
    return encoded_shape


# applies a decoder to x_enc and then applies the transform to I
def apply_transformation(x_source, x_transformation,
                         output_shape,
                         conditioning_input_shape,
                         transform_name,
                         flow_indexing='xy',
                         color_transform_type='WB',
                         ):
    n_dims = len(conditioning_input_shape) - 1

    transformation_shape = x_transformation.get_shape().as_list()[1:]
    x_transformation = Reshape(transformation_shape, name='{}_dec_out'.format(transform_name))(x_transformation)

    # apply color transform
    print('Applying color transform {}'.format(color_transform_type))
    if color_transform_type == 'delta':
        x_color_out = Add()([x_source, x_transformation])
    elif color_transform_type == 'mult':
        x_color_out = Multiply()([x_source, x_transformation])
    else:
        raise NotImplementedError('Only color transform types delta and mult are supported!')
    im_out = Reshape(output_shape, name='color_transformer')(x_color_out)

    return im_out, x_transformation


def cvae_trainer_wrapper(
        ae_input_shapes, ae_input_names,
        conditioning_input_shapes, conditioning_input_names,
        output_shape=None,
        model_name='transformer_trainer',
        transform_encoder_model=None, transformer_model=None,
        transform_type='flow',
        transform_latent_shape=(50,),
        n_outputs=1,
):
    '''''''''''''''''''''
    CVAE trainer model
        - takes I, I+J as input
        - encodes I+J to z
        - condition_on_image = True means that the transform is decoded from the transform+image embedding,
                otherwise it is decoded from only the transform embedding
        - decodes latent embedding into transform and applies it
    '''''''''''''''''''''
    ae_inputs, ae_stack, conditioning_inputs, cond_stack = _collect_inputs(
        ae_input_shapes, ae_input_names,
        conditioning_input_shapes, conditioning_input_names)
    conditioning_input_shape = cond_stack.get_shape().as_list()[1:]

    inputs = ae_inputs + conditioning_inputs

    # encode x_stacked into z
    z_mean, z_logvar = transform_encoder_model(ae_inputs)

    z_mean = Reshape(transform_latent_shape, name='latent_mean')(z_mean)
    z_logvar = Reshape(transform_latent_shape, name='latent_logvar')(z_logvar)

    z_sampled = Lambda(sampling, output_shape=transform_latent_shape, name='lambda_sampling')(
        [z_mean, z_logvar])

    decoder_out = transformer_model(conditioning_inputs + [z_sampled])

    if transform_type == 'flow':
        im_out, transform_out = decoder_out
        transform_shape = transform_out.get_shape().as_list()[1:]

        transform_out = Reshape(transform_shape, name='decoder_flow_out')(transform_out)
        im_out = Reshape(output_shape, name='spatial_transformer')(im_out)
    elif transform_type == 'color':
        im_out, transform_out = decoder_out

        transform_out = Reshape(output_shape, name='decoder_color_out')(transform_out)
        im_out = Reshape(output_shape, name='color_transformer')(im_out)
    else:
        im_out = decoder_out

    if transform_type is not None:
        return Model(inputs=inputs, outputs=[im_out] * n_outputs + [transform_out, z_mean, z_logvar], name=model_name)
    else:
        return Model(inputs=inputs, outputs=[im_out] * n_outputs + [z_mean, z_logvar], name=model_name)


def cvae_tester_wrapper(
        conditioning_input_shapes, conditioning_input_names,
        latent_shape,
        dec_model,
        n_outputs=1,
        model_name='cvae_tester_model'
):
    # collect conditioning inputs, and concatentate them into a stack
    if not isinstance(conditioning_input_shapes, list):
        conditioning_input_shapes = [conditioning_input_shapes]
    if conditioning_input_names is None:
        conditioning_input_names = ['cond_input_{}'.format(ii) for ii in range(len(conditioning_input_shapes))]

    conditioning_inputs = []
    for ii, input_shape in enumerate(conditioning_input_shapes):
        conditioning_inputs.append(Input(input_shape, name=conditioning_input_names[ii]))


    z_dummy_input = Input(latent_shape, name='z_input')

    z_samp = Lambda(sampling_sigma1,
                        name='lambda_z_sampling_stdnormal'
                        )(z_dummy_input)
    y = dec_model(conditioning_inputs + [z_samp])
    if isinstance(y, list): # multiple outputs, assume the first is the actual transformed frame
        y = [y[0]] * n_outputs + y[1:]
    else:
        y = [y] * n_outputs

    return Model(inputs=conditioning_inputs + [z_dummy_input], outputs=y, name=model_name)


def _collect_inputs(ae_input_shapes, ae_input_names,
        conditioning_input_shapes, conditioning_input_names,):

    ae_inputs = []
    ae_stack = None
    if ae_input_shapes is not None:
        if not isinstance(ae_input_shapes, list):
            ae_input_shapes = [ae_input_shapes]

        if ae_input_names is None:
            ae_input_names = ['input_{}'.format(ii) for ii in range(len(ae_input_names))]

        for ii, input_shape in enumerate(ae_input_shapes):
            ae_inputs.append(Input(input_shape, name='input_{}'.format(ae_input_names[ii])))

        ae_stack = Concatenate(name='concat_inputs', axis=-1)(ae_inputs)
        ae_stack_shape = ae_stack.get_shape().as_list()[1:]

    # collect conditioning inputs, and concatentate them into a stack
    if not isinstance(conditioning_input_shapes, list):
        conditioning_input_shapes = [conditioning_input_shapes]
    if conditioning_input_names is None:
        conditioning_input_names = ['cond_input_{}'.format(ii) for ii in range(len(conditioning_input_shapes))]

    conditioning_inputs = []
    for ii, input_shape in enumerate(conditioning_input_shapes):
        conditioning_inputs.append(Input(input_shape, name=conditioning_input_names[ii]))

    if len(conditioning_inputs) > 1:
        cond_stack = Concatenate(name='concat_cond_inputs', axis=-1)(conditioning_inputs)
    else:
        cond_stack = conditioning_inputs[0]
    return ae_inputs, ae_stack, conditioning_inputs, cond_stack


##### basic layers ########
def myConv(nf, n_dims, prefix=None, suffix=None, ks=3, strides=1,
           kernel_initializer=None, bias_initializer=None):
    if kernel_initializer is None:
        kernel_initializer = 'glorot_uniform' # keras default for conv kernels
    if bias_initializer is None:
        bias_initializer = 'zeros' # default for keras conv

    # wrapper for 2D and 3D conv
    if n_dims == 2:
        if not isinstance(strides, tuple):
            strides = (strides, strides)
        return Conv2D(nf, kernel_size=ks, padding='same', strides=strides,
                      kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                      name='_'.join([
                          str(part) for part in [prefix, 'conv2D', suffix]  # include prefix and suffix if they exist
                          if part is not None and len(str(part)) > 0]))
    elif n_dims == 3:
        if not isinstance(strides, tuple):
            strides = (strides, strides, strides)
        return Conv3D(nf, kernel_size=ks, padding='same', strides=strides,
                      kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                      name='_'.join([
                          str(part) for part in [prefix, 'conv3D', suffix]  # include prefix and suffix if they exist
                          if part is not None and len(str(part)) > 0]))
    else:
        print('N dims {} is not supported!'.format(n_dims))
        sys.exit()

def myPool(n_dims, prefix=None, suffix=None):
    if n_dims == 2:
        return MaxPooling2D(padding='same',
                            name='_'.join([
                                str(part) for part in [prefix, 'maxpool2D', suffix]  # include prefix and suffix if they exist
                                if part is not None and len(str(part)) > 0]))
    elif n_dims == 3:
        return MaxPooling3D(padding='same',
                            name='_'.join([
                                str(part) for part in [prefix, 'maxpool3D', suffix]  # include prefix and suffix if they exist
                                if part is not None and len(str(part)) > 0]))


def sampling(args):
	epsilon_std = 1.0
	z_mean, z_logvar = args
	epsilon = K.random_normal(shape=K.shape(z_mean), mean=0.,
														stddev=epsilon_std)
	sample = z_mean + K.exp(z_logvar / 2.) * epsilon
	return sample

# a little hacky, but put anything as input here just to get the batch size
def sampling_stdnormal(args):
	epsilon_std = 1.0

	epsilon = K.random_normal(
		shape=K.shape(args),
		mean=0., stddev=epsilon_std)
	return epsilon

def sampling_sigma1(z_mean):
	epsilon_std = 1.0
	epsilon = K.random_normal(shape=K.shape(z_mean), mean=0.,
		stddev=epsilon_std)
	return z_mean + epsilon


