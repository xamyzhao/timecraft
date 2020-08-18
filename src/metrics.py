import os

import keras.backend as K
from keras.applications.vgg16 import VGG16
from keras.layers import Conv2D, Input, MaxPooling2D, Lambda
from keras.models import Model, load_model

import tensorflow as tf

class TimeSummedLoss(object):
    def __init__(self, loss_fn, weights_output=None,
                     time_axis=-2, compute_mean=True, pad_amt=None,
                    do_reshape_to=None,
                     compute_over_frame_idxs=None,
                     ):
        self.time_axis = time_axis
        self.loss_fn = loss_fn
        self.weights_output = weights_output # in case we want to predict the weight for each time step
        self.compute_mean = compute_mean
        self.pad_amt = pad_amt
        self.include_frames = compute_over_frame_idxs
        self.do_reshape_to = do_reshape_to

    def compute_loss(self, y_true, y_pred):
        if self.do_reshape_to is not None: # reshape a flattened video
            y_true = tf.reshape(y_true, [-1] + list(self.do_reshape_to))
            y_pred = tf.reshape(y_pred, [-1] + list(self.do_reshape_to))

        if self.pad_amt is not None:
            y_true = tf.pad(y_true, paddings=self.pad_amt, constant_values=1.)
            y_pred = tf.pad(y_pred, paddings=self.pad_amt, constant_values=1.)

        n_frames = y_pred.get_shape().as_list()[self.time_axis]
        if self.include_frames is None:
            include_frames = list(range(n_frames))
        else:
            include_frames = self.include_frames

        # TODO: switch to tf.map_fn?
        true_frames = tf.unstack(y_true, num=n_frames, axis=self.time_axis)
        pred_frames = tf.unstack(y_pred, num=n_frames, axis=self.time_axis)

        if self.weights_output is not None:
            weights_list = tf.unstack(self.weights_output, num=n_frames, axis=-1)
        else:
            weights_list = None

        total_loss = 0
        for t in include_frames:
            loss = self.loss_fn(y_true=true_frames[t], y_pred=pred_frames[t])
            if weights_list is not None:
                curr_weights = tf.expand_dims(weights_list[t], axis=-1)
                loss = tf.multiply(loss, curr_weights)

            total_loss += loss

        if self.compute_mean:
            total_loss /= float(n_frames)

        return total_loss


class CriticScore(object):
    # WGAN loss
    def __init__(self, critic_model, loss_fn=None, target_score=None):
        self.critic_model = critic_model
        self.critic_model.trainable = False
        for l in self.critic_model.layers:
            l.trainable = False

        self.loss_fn = loss_fn
        self.target_score = target_score

    def compute_loss(self, y_true, y_pred):
        critic_score = self.critic_model(y_pred)

        if self.loss_fn is None:
            return -tf.reduce_mean(critic_score)
        elif self.loss_fn is not None and self.target_score is not None:
            return self.loss_fn(self.target_score, critic_score)
        elif self.loss_fn is not None and self.target_score is None:
            return self.loss_fn(y_true, critic_score)


def norm_vgg(x):
    import tensorflow as tf
    eps = 1e-10
    x_norm = tf.sqrt(tf.reduce_sum(x * x, axis=-1, keep_dims=True))
    x_norm = tf.divide(x, x_norm + eps)
    return x_norm

class VggFeatLoss(object):
    def __init__(self, feat_net, dist='l2'):
        self.feat_net = feat_net
        self.dist = dist

    def compute_loss(self, y_true, y_pred):
        import tensorflow as tf
        # just preprocess as a part of the model
        n_feature_layers = len(self.feat_net.outputs)
        x1 = self.feat_net(y_true)
        x2 = self.feat_net(y_pred)

        loss = []

        for li in range(n_feature_layers):
            x1_l = x1[li]
            x2_l = x2[li]

            # unit normalize in channels dimension
            x1_l_norm = norm_vgg(x1_l)
            x2_l_norm = norm_vgg(x2_l)

            hw = tf.shape(x1_l)[1] * tf.shape(x1_l)[2]

            if self.dist == 'l1':
                d = tf.reduce_sum(tf.abs(x1_l_norm - x2_l_norm), [1, 2, 3])  # bx1
            else:
                d = tf.reduce_sum(tf.square(x1_l_norm - x2_l_norm), [1, 2, 3])  # bx1
            d_mean = tf.divide(d, tf.cast(hw, tf.float32))

            if li == 0:
                loss = d_mean
            else:
                loss = loss + d_mean
        return loss


class VAE_metrics(object):
    """
    Losses for variational auto-encoders
    """

    def __init__(self,
                 var_target=None,
                 logvar_target=None,
                 mu_target=None,
                 axis=1):
        #        self.log_var_pred = log_var_pred
        #        self.mu_pred = mu_pred

        self.var_target = var_target
        self.logvar_target = logvar_target

        self.mu_target = mu_target

        self.axis = axis

    def kl_log_sigma(self, y_true, y_pred):
        """
        kl_log_sigma terms of the KL divergence
        """
        eps = 1e-8

        logvar_pred = y_pred
        var_pred = K.exp(y_pred)

        if self.var_target is None and self.logvar_target is not None:
            var_target = K.exp(self.logvar_target)
        elif self.var_target is not None:
            var_target = self.var_target
        elif self.var_target is None and self.logvar_target is None:
            var_target = y_true

        kl_sigma_out = 0.5 * K.sum(
            (var_pred / (var_target + eps)) \
            + K.log(var_target)
            - logvar_pred \
            - 1, axis=self.axis)
        return kl_sigma_out

    def kl_mu(self, y_true, y_pred):
        """
        kl_mu terms of the KL divergence
        y_pred should be mu_out
        """
        eps = 1e-8

        if self.var_target is None and self.logvar_target is not None:
            var_target = K.exp(self.logvar_target)
        elif self.var_target is not None:
            var_target = self.var_target
        elif self.var_target is None and self.logvar_target is None:
            var_target = y_true

        # TODO: we cant have both self.mu_target is None and slef.var_target is None
        if self.mu_target is None:
            mu_target = y_true
        else:
            mu_target = self.mu_target

        kl_mu_out = 0.5 * K.sum(
            K.square(y_pred - mu_target) / (var_target + eps),
            axis=self.axis)
        return kl_mu_out


def vgg_preprocess(arg):
    import tensorflow as tf
    z = 255.0 * tf.clip_by_value(arg, 0., 1.)
    b = z[:, :, :, 0] - 103.939
    g = z[:, :, :, 1] - 116.779
    r = z[:, :, :, 2] - 123.68
    return tf.stack([b, g, r], axis=3)

def vgg_preprocess_norm(arg):
    import tensorflow as tf
    #z = tf.clip_by_value(arg, -1., 1.)
    z = 255.0 * (arg * 0.5 + 0.5)
    b = z[:, :, :, 0] - 103.939
    g = z[:, :, :, 1] - 116.779
    r = z[:, :, :, 2] - 123.68
    return tf.stack([b, g, r], axis=3)

def vgg_isola_norm(shape=(64, 64, 3), normalized_inputs=False, do_preprocess=True):
    img_input = Input(shape=shape)

    if do_preprocess:
        if normalized_inputs:
            vgg_model_file = 'vgg16_normtanh_ims{}-{}.h5'.format(shape[0], shape[1])
            img = Lambda(vgg_preprocess_norm,
                         name='lambda_preproc_norm-11')(img_input)
        else:
            vgg_model_file = 'vgg16_01_ims{}-{}.h5'.format(shape[0], shape[1])
            img = Lambda(vgg_preprocess,
                         name='lambda_preproc_clip01')(img_input)
    else:
        vgg_model_file = 'vgg16_ims{}-{}.h5'.format(shape[0], shape[1])
        img = img_input

    if os.path.isfile(vgg_model_file):
        print('Loading vgg model from {}'.format(vgg_model_file))
        return load_model(vgg_model_file,
                          custom_objects={
                              'vgg_preprocess_norm': vgg_preprocess_norm,
                              'vgg_preprocess': vgg_preprocess})

    # Block 1
    x1 = Conv2D(64, (3, 3),
                activation='relu',
                padding='same',
                name='block1_conv1')(img)
    x2 = Conv2D(64, (3, 3),
                activation='relu',
                padding='same',
                name='block1_conv2')(x1)  # relu is layer 4 in torch implementation
    x3 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x2)

    # Block 2
    x4 = Conv2D(128, (3, 3),
                activation='relu',
                padding='same',
                name='block2_conv1')(x3)
    x5 = Conv2D(128, (3, 3),
                activation='relu',
                padding='same',
                name='block2_conv2')(x4)  # relu is layer 9
    x6 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x5)

    # Block 3
    x7 = Conv2D(256, (3, 3),
                activation='relu',
                padding='same',
                name='block3_conv1')(x6)
    x8 = Conv2D(256, (3, 3),
                activation='relu',
                padding='same',
                name='block3_conv2')(x7)
    x9 = Conv2D(256, (3, 3),
                activation='relu',
                padding='same',
                name='block3_conv3')(x8)  # relu is layer 16
    x10 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x9)

    # Block 4
    x11 = Conv2D(512, (3, 3),
                 activation='relu',
                 padding='same',
                 name='block4_conv1')(x10)
    x12 = Conv2D(512, (3, 3),
                 activation='relu',
                 padding='same',
                 name='block4_conv2')(x11)
    x13 = Conv2D(512, (3, 3),
                 activation='relu',
                 padding='same',
                 name='block4_conv3')(x12)  # relu is layer 23 in torch
    x14 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x13)

    # Block 5
    x15 = Conv2D(512, (3, 3),
                 activation='relu',
                 padding='same',
                 name='block5_conv1')(x14)
    x16 = Conv2D(512, (3, 3),
                 activation='relu',
                 padding='same',
                 name='block5_conv2')(x15)
    x17 = Conv2D(512, (3, 3),
                 activation='relu',
                 padding='same',
                 name='block5_conv3')(x16)  # relu is layer 30

    # isola implementation uses layer outputs 4, 9, 16, 23, 30, but need to count activations
    model = Model(inputs=[img_input], outputs=[x2, x5, x9, x13, x17])
    model_orig = VGG16(weights='imagenet', input_shape=shape, include_top=False)

    # ignore the lambda we put in for preprocessing
    vgg_layers = [l for l in model.layers if not isinstance(l, Lambda)]
    for li, l in enumerate(vgg_layers):
        weights = model_orig.layers[li].get_weights()
        l.set_weights(weights)
        print('Copying imagenet weights for layer {}: {}'.format(li, l.name))
        l.trainable = False

    if not os.path.isfile(vgg_model_file):
        model.save(vgg_model_file)

    return model

