from keras.layers import Layer
import numpy as np
import tensorflow as tf

def create_gaussian_kernel(sigma, n_sigmas_per_side=8, n_dims=2):
    t = np.linspace(-sigma * n_sigmas_per_side / 2, sigma * n_sigmas_per_side / 2, int(sigma * n_sigmas_per_side + 1))
    gauss_kernel_1d = np.exp(-0.5 * (t / sigma) ** 2)

    if n_dims == 2:
        gauss_kernel_2d = gauss_kernel_1d[:, np.newaxis] * gauss_kernel_1d[np.newaxis, :]
    else:
        gauss_kernel_2d = gauss_kernel_1d[:, np.newaxis, np.newaxis] * gauss_kernel_1d[np.newaxis, np.newaxis,
                                                                       :] * gauss_kernel_1d[np.newaxis, :, np.newaxis]
    gauss_kernel_2d = gauss_kernel_2d / np.sum(gauss_kernel_2d)

    return gauss_kernel_2d

class Blur_Downsample(Layer):
    def __init__(self, n_chans=3, n_dims=2, do_blur=True, **kwargs):
        super(Blur_Downsample, self).__init__(**kwargs)
        scale_factor = 0.5  # we only support halving right now

        if do_blur:
            # according to scikit-image.transform.rescale documentation
            blur_sigma = (1. - scale_factor) / 2

            blur_kernel = create_gaussian_kernel(blur_sigma, n_dims=n_dims, n_sigmas_per_side=4)
            blur_kernel = np.reshape(blur_kernel, blur_kernel.shape + (1, 1))
            self.blur_kernel = tf.constant(blur_kernel, dtype=tf.float32)
        else:
            self.blur_kernel = tf.constant(np.ones([1] * n_dims + [1, 1]), dtype=tf.float32)
        self.n_dims = n_dims
        self.n_chans = n_chans

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        self.n_chans = inputs[0].get_shape().as_list()[-1]

        if self.n_dims == 2:
            strides = [1, 2, 2, 1]
            conv_fn = tf.nn.conv2d
        elif self.n_dims == 3:
            strides = [1, 2, 2, 2, 1]
            conv_fn = tf.nn.conv3d

        chans_list = tf.unstack(inputs, num=self.n_chans, axis=-1)
        blurred_chans = []
        for c in range(self.n_chans):
            blurred_chan = conv_fn(tf.expand_dims(chans_list[c], axis=-1), self.blur_kernel,
                                   strides=strides, padding='SAME')
            # get rid of last dimension (size 1)
            blurred_chans.append(tf.gather(blurred_chan, 0, axis=-1))
        blurred = tf.stack(blurred_chans, axis=-1)
        return blurred

    def compute_output_shape(self, input_shape):
        return tuple([input_shape[0]] + [int(np.ceil(s / 2)) for s in input_shape[1:-1]] + [self.n_chans])
