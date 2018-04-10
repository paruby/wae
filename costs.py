# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license,
# (See accompanying file ./LICENSE.txt or copy at
# https://opensource.org/licenses/BSD-3-Clause)

# Different additional losses for the WAE framework
import tensorflow as tf
import numpy as np
import ops
from models import encoder, decoder

valid_smart_costs = ['patch_variances', 'l2sq', '_sylvain_recon_loss_using_disc_conv', '_sylvain_recon_loss_using_moments']

def flatten(tensor):
    return tf.reshape(tensor, [-1, prod_dim(tensor)])

def prod_dim(tensor):
    return np.prod([int(d) for d in tensor.get_shape()[1:]])

def check_valid_smart_cost(costs_list):
    """Checks if smart cost options are valid.
    Valid costs are lists of 2-tuples giving component and weight.
    e.g. [('patch_variances', 1.0)] is valid.
    """
    if not isinstance(costs_list, list):
        return False
    for x in costs_list:
        assert isinstance(x, tuple) and len(x) == 2 and x[0] in valid_smart_costs
        cost = x[0]
        weight = x[1]
        if cost == 'patch_variances':
            assert isinstance(weight, float)
        if cost == 'l2sq':
            assert isinstance(weight, float)
        if cost == '_sylvain_recon_loss_using_disc_conv':
            assert isinstance(weight, list)
            assert len(weight) == 2
            assert isinstance(weight[0], float) and isinstance(weight[1], float)
        if cost == '_sylvain_recon_loss_using_moments':
            assert isinstance(weight, float)
    return True

def construct_cost(wae, opts, real, reconstr):
    loss = 0
    for cost, weight in opts['cost']:
        if cost == 'patch_variances':
            loss += weight * _patch_variances(opts, real, reconstr)
        if cost == 'l2sq':
            loss += 0.05 * weight * _l2sq(opts, real, reconstr) # magic number copied from l2sq loss in wae.py
        if cost == '_sylvain_recon_loss_using_disc_conv':
            adv_c_loss_w, emb_c_loss_w = weight
            adv_c_loss, emb_c_loss = _sylvain_recon_loss_using_disc_conv(opts, reconstr, real)
            loss += adv_c_loss * adv_c_loss_w + emb_c_loss * emb_c_loss_w
        if cost == '_sylvain_recon_loss_using_moments':
            loss += weight * _sylvain_recon_loss_using_moments(opts, reconstr, real)
    return loss

def _patch_variances(opts, real, reconstr):
    """Cost is l2_sq difference between variances of patches in real
    and reconstructed images.
    """
    real_sq = real**2
    reconstr_sq = reconstr**2
    height, width, channels = [int(real.get_shape()[i]) for i in range(1,4)]
    if 'cost_kernel_sizes' in opts:
        kernel_sizes = opts['cost_kernel_sizes']
    else:
        kernel_sizes = [3,4,5] #defaults
    if isinstance(kernel_sizes, int):
        kernel_sizes = [kernel_sizes]
    assert isinstance(kernel_sizes, list)

    loss = 0
    for kernel_size in kernel_sizes: #patch sizes in which we calculate variance
        w_sum = tf.eye(num_rows=channels, num_columns=channels, batch_shape=[kernel_size * kernel_size])
        w_sum = tf.reshape(w_sum, [kernel_size, kernel_size, channels, channels])
        w_sum = w_sum / (kernel_size*kernel_size)

        real_mean = tf.nn.conv2d(real, w_sum, strides=[1,1,1,1], padding='VALID')
        reconstr_mean = tf.nn.conv2d(reconstr, w_sum, strides=[1,1,1,1], padding='VALID')

        real_var = tf.nn.conv2d(real_sq, w_sum, strides=[1,1,1,1], padding='VALID') - real_mean**2
        reconstr_var = tf.nn.conv2d(reconstr_sq, w_sum, strides=[1,1,1,1], padding='VALID') - reconstr_mean**2

        sq_var_diff = tf.reduce_sum((real_var - reconstr_var)**2, axis=[1,2,3])
        sq_var_diff = tf.reduce_mean(sq_var_diff)
        loss += sq_var_diff

    return loss

def _l2sq(opts, real, reconstr):
    """(Pixel wise) L2sq distance between real and reconstructed images
    """
    loss = tf.reduce_sum(tf.square(real - reconstr), axis=[1, 2, 3])
    loss = tf.reduce_mean(loss)
    return loss


def _sylvain_recon_loss_using_disc_conv(opts, reconstructed_training, real_points):
    """Build an additional loss using a discriminator in X space."""
    def _conv_flatten(x, kernel_size):
        height = int(x.get_shape()[1])
        width = int(x.get_shape()[2])
        channels = int(x.get_shape()[3])
        w_sum = tf.eye(num_rows=channels, num_columns=channels, batch_shape=[kernel_size * kernel_size])
        w_sum = tf.reshape(w_sum, [kernel_size, kernel_size, channels, channels])
        w_sum = w_sum / (kernel_size * kernel_size)
        sum_ = tf.nn.conv2d(x, w_sum, strides=[1, 1, 1, 1], padding='SAME')
        size = prod_dim(sum_)
        assert size == height * width * channels, size
        return tf.reshape(sum_, [-1, size])

    def _gram_scores(tensor, kernel_size):
        assert len(tensor.get_shape()) == 4, tensor
        ttensor = tf.transpose(tensor, [3, 1, 2, 0])
        rand_indices = tf.random_shuffle(tf.range(ttensor.get_shape()[0]))
        shuffled = tf.gather(ttensor, rand_indices)

        shuffled = tf.transpose(shuffled, [3, 1, 2, 0])
        cross_p = _conv_flatten(tensor * shuffled, kernel_size)  # shape [batch_size, height * width * channels]
        diag_p = _conv_flatten(tf.square(tensor), kernel_size)  # shape [batch_size, height * width * channels]
        return cross_p, diag_p

    def _architecture(inputs, reuse=None):
        with tf.variable_scope('DISC_X_LOSS', reuse=reuse):
            num_units = opts['adv_c_num_units']
            num_layers = 1
            filter_sizes = opts['adv_c_patches_size']
            if isinstance(filter_sizes, int):
                filter_sizes = [filter_sizes]
            else:
                filter_sizes = [int(n) for n in filter_sizes.split(',')]
            embedded_outputs = []
            linear_outputs = []
            for filter_size in filter_sizes:
                layer_x = inputs
                for i in xrange(num_layers):
                    layer_x = ops.conv2d(opts, layer_x, num_units, d_h=1, d_w=1, scope='h%d_conv%d' % (i, filter_size),
                                         conv_filters_dim=filter_size, padding='SAME')
                    layer_x = ops.lrelu(layer_x, 0.1)
                last = ops.conv2d(
                    opts, layer_x, 1, d_h=1, d_w=1, scope="last_lin%d" % filter_size, conv_filters_dim=1, l2_norm=True)
                if opts['cross_p_w'] > 0.0 or opts['diag_p_w'] > 0.0:
                    cross_p, diag_p = _gram_scores(layer_x, filter_size)
                    embedded_outputs.append(cross_p * opts['cross_p_w'])
                    embedded_outputs.append(diag_p * opts['diag_p_w'])
                fl = flatten(layer_x)
                embedded_outputs.append(fl)
                size = int(last.get_shape()[1])
                linear_outputs.append(tf.reshape(last, [-1, size * size]))
            if len(embedded_outputs) > 1:
                embedded_outputs = tf.concat(embedded_outputs, 1)
            else:
                embedded_outputs = embedded_outputs[0]
            if len(linear_outputs) > 1:
                linear_outputs = tf.concat(linear_outputs, 1)
            else:
                linear_outputs = linear_outputs[0]

            return embedded_outputs, linear_outputs

    if 'adv_use_sq' in opts:
        if opts['adv_use_sq'] is True:
            reconstructed_training_sq = reconstructed_training ** 2
            real_points_sq = real_points ** 2

            reconstructed_training = tf.concat([reconstructed_training, reconstructed_training_sq], axis=-1)
            real_points = tf.concat([real_points, real_points_sq], axis=-1)

    reconstructed_embed_sg, adv_fake_layer = _architecture(tf.stop_gradient(reconstructed_training), reuse=None)
    reconstructed_embed, _ = _architecture(reconstructed_training, reuse=True)
    # Below line enforces the forward to be reconstructed_embed and backwards to NOT change the discriminator....
    crazy_hack = reconstructed_embed-reconstructed_embed_sg+tf.stop_gradient(reconstructed_embed_sg)
    real_p_embed_sg, adv_true_layer = _architecture(tf.stop_gradient(real_points), reuse=True)
    real_p_embed, _ = _architecture(real_points, reuse=True)

    adv_fake = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=adv_fake_layer, labels=tf.zeros_like(adv_fake_layer))
    adv_true = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=adv_true_layer, labels=tf.ones_like(adv_true_layer))
    adv_fake = tf.reduce_mean(adv_fake)
    adv_true = tf.reduce_mean(adv_true)

    adv_c_loss = adv_fake + adv_true
    emb_c = tf.reduce_mean(tf.square(crazy_hack - tf.stop_gradient(real_p_embed)), 1)

    real_points_shuffle = tf.stop_gradient(tf.random_shuffle(real_p_embed))
    emb_c_shuffle = tf.reduce_mean(tf.square(real_points_shuffle - tf.stop_gradient(reconstructed_embed)), 1)

    raw_emb_c_loss = tf.reduce_mean(emb_c)
    shuffled_emb_c_loss = tf.reduce_mean(emb_c_shuffle)
    emb_c_loss = raw_emb_c_loss / shuffled_emb_c_loss
    emb_c_loss = emb_c_loss * 40

    return adv_c_loss, emb_c_loss


def _sylvain_recon_loss_using_moments(opts, reconstructed_training, real_points):
    """Build an additional loss using moments."""

    def _architecture(_inputs):
        return _sylvain_compute_moments(_inputs, moments=[2])  # TODO

    reconstructed_embed = _architecture(reconstructed_training)
    real_p_embed = _architecture(real_points)

    emb_c = tf.reduce_mean(tf.square(reconstructed_embed - tf.stop_gradient(real_p_embed)), 1)

    emb_c_loss = tf.reduce_mean(emb_c)
    return emb_c_loss * 100.0 * 100.0 # TODO: constant.


def _sylvain_compute_moments(_inputs, moments=[2, 3]):
    """From an image input, compute moments"""
    _inputs_sq = tf.square(_inputs)
    _inputs_cube = tf.pow(_inputs, 3)
    height = int(_inputs.get_shape()[1])
    width = int(_inputs.get_shape()[2])
    channels = int(_inputs.get_shape()[3])
    def ConvFlatten(x, kernel_size):
        w_sum = tf.eye(num_rows=channels, num_columns=channels, batch_shape=[kernel_size * kernel_size])
        w_sum = tf.reshape(w_sum, [kernel_size, kernel_size, channels, channels])
        w_sum = w_sum / (kernel_size * kernel_size)
        sum_ = tf.nn.conv2d(x, w_sum, strides=[1, 1, 1, 1], padding='VALID')
        size = prod_dim(sum_)
        assert size == (height - kernel_size + 1) * (width - kernel_size + 1) * channels, size
        return tf.reshape(sum_, [-1, size])
    outputs = []
    for size in [3, 4, 5]:
        mean = ConvFlatten(_inputs, size)
        square = ConvFlatten(_inputs_sq, size)
        var = square - tf.square(mean)
        if 2 in moments:
            outputs.append(var)
        if 3 in moments:
            cube = ConvFlatten(_inputs_cube, size)
            skewness = cube - 3.0 * mean * var - tf.pow(mean, 3)  # Unnormalized
            outputs.append(skewness)
    return tf.concat(outputs, 1)

def add_aefixedpoint_cost(opts, wae_model):

    w_aefixedpoint = tf.placeholder(tf.float32, name='w_aefixedpoint')
    wae_model.w_aefixedpoint = w_aefixedpoint

    gen_images = wae_model.decoded
    gen_images.set_shape([opts['batch_size']] + wae_model.data_shape)
    tmp = encoder(opts, reuse=True, inputs=gen_images,
                  is_training=wae_model.is_training)
    tmp_sg = encoder(opts, reuse=True,
                     inputs=tf.stop_gradient(gen_images),
                     is_training=wae_model.is_training)
    encoded_gen_images = tmp[0]
    encoded_gen_images_sg = tmp_sg[0]
    if opts['e_noise'] == 'gaussian':
        # Encoder outputs means and variances of Gaussian
        # Encoding into means
        encoded_gen_images = encoded_gen_images[0]
        encoded_gen_images_sg = encoded_gen_images_sg[0]
    autoencoded_gen_images, _ = decoder(
        opts, reuse=True, noise=encoded_gen_images,
        is_training=wae_model.is_training)
    autoencoded_gen_images_sg, _ = decoder(
        opts, reuse=True, noise=encoded_gen_images_sg,
        is_training=wae_model.is_training)
    a = wae_model.reconstruction_loss(gen_images, autoencoded_gen_images)
    b = tf.stop_gradient(a)
    c = wae_model.reconstruction_loss(
            tf.stop_gradient(gen_images),
            autoencoded_gen_images_sg)
    extra_cost = b + a - c
    # Check gradients
    # encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
    # wae_model.grad_extra = tf.gradients(ys=extra_cost, xs=encoder_vars)
    # for idx, el in enumerate(wae_model.grad_extra):
    #    print encoder_vars[idx].name, el

    wae_model.wae_objective += wae_model.w_aefixedpoint * extra_cost
