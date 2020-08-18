import keras.backend as K
import numpy as np
from keras.layers import Input, Concatenate, Dense, Flatten, MaxPooling2D, LeakyReLU, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model

import tensorflow as tf

from src.networks import network_modules

def gp_loss(y_true, y_pred):
    # y_pred should be the gradient of disc_model(interp_input) wrt interp_input
    slopes = tf.sqrt(tf.reduce_sum(tf.square(y_pred), axis=1))
    return tf.reduce_mean((slopes - 1)**2)

def wgan_gp_trainer(img_size, disc_model):
    x_input_real = Input(img_size, name='disc_wgan_input_real')
    x_input_fake = Input(img_size, name='disc_wgan_input_fake')

    disc_pred_fake = disc_model(x_input_fake)
    disc_pred_real = disc_model(x_input_real)

    averaged_samples = RandomWeightedAverage(name='averaged_samples')([x_input_real, x_input_fake])
    disc_pred_interp = disc_model(averaged_samples)

    return Model(inputs=[x_input_real, x_input_fake], outputs=[disc_pred_fake, disc_pred_real, disc_pred_interp],
                  name='disc_wgan_trainer')

def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)

from keras.layers.merge import _Merge
class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((K.shape(inputs[0])[0], 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


def gradient_penalty_loss(y_true, y_pred, averaged_samples,
                          gradient_penalty_weight):
    """Calculates the gradient penalty loss for a batch of "averaged" samples.
    In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the
    loss function that penalizes the network if the gradient norm moves away from 1.
    However, it is impossible to evaluate this function at all points in the input
    space. The compromise used in the paper is to choose random points on the lines
    between real and generated samples, and check the gradients at these points. Note
    that it is the gradient w.r.t. the input averaged samples, not the weights of the
    discriminator, that we're penalizing!
    In order to evaluate the gradients, we must first run samples through the generator
    and evaluate the loss. Then we get the gradients of the discriminator w.r.t. the
    input averaged samples. The l2 norm and penalty can then be calculated for this
    gradient.
    Note that this loss function requires the original averaged samples as input, but
    Keras only supports passing y_true and y_pred to loss functions. To get around this,
    we make a partial() of the function with the averaged_samples argument, and use that
    for model training."""
    # first get the gradients:
    #   assuming: - that y_pred has dimensions (batch_size, 1)
    #             - averaged_samples has dimensions (batch_size, nbr_features)
    # gradients afterwards has dimension (batch_size, nbr_features), basically
    # a list of nbr_features-dimensional gradient vectors
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)


def discriminator_patch(input_shape,
                        enc_params=None,
                        include_activation=False,
                        model_name='discriminator_patch',
                        use_lsgan=False,
                        include_dense=True,
                        include_1channel_conv=True,
                        ):
    from keras.layers.convolutional import Conv2D
    x_input = Input(shape=input_shape, name='input_patch')

    if use_lsgan:
        include_activation = True
        x = network_modules.encoder(x_input, ks=5,
                                   img_shape=input_shape,
                                   prefix='disc_dsgan',
                                   n_convs_per_stage=enc_params['n_convs_per_stage'],
                                   conv_chans=enc_params['nf_enc'],
                                   use_maxpool=False,
                                   use_batchnorm=True,
                                   )
    else:
        x = network_modules.encoder(x_input, ks=enc_params['ks'], # stargan uses ks 4
                                   img_shape=input_shape,
                                   prefix='disc',
                                   n_convs_per_stage=enc_params['n_convs_per_stage'],
                                   conv_chans=enc_params['nf_enc'],
                                   use_maxpool=enc_params['use_maxpool'],
                                kernel_initializer='he_uniform', bias_initializer='lecun_uniform',
                            )
    if include_1channel_conv: # TODO: this is specifically for stargan arch
        x = Conv2D(1, kernel_size=3, padding='same', name='conv_last')(x)

    x = Flatten()(x)

    if include_dense:
        x = Dense(256)(x)
        x = LeakyReLU(0.2)(x)
        x = Dense(1, kernel_initializer='he_normal')(x)

    if include_activation:
        x = Activation('sigmoid')(x)
    return Model(inputs=x_input, outputs=x, name=model_name)