# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license,
# (See accompanying file ./LICENSE.txt or copy at
# https://opensource.org/licenses/BSD-3-Clause)

# Different additional losses for the WAE framework
import tensorflow as tf
import numpy as np

valid_smart_costs = ['patch_variances']

def check_valid_smart_cost(costs_list):
    """Checks if smart cost options are valid.
    Valid costs are lists of 2-tuples giving component and weight.
    e.g. [('patch_variances', 1.0)] is valid.
    """
    if not isinstance(costs_list, list):
        return False
    for x in costs_list:
        if not (isinstance(x, tuple) and len(x) == 2 and x[0] in valid_smart_costs and isinstance(x[1], float)):
            return False
    return True

def construct_cost(opts, real, reconstr):
    loss = 0
    for cost, weight in opts['cost']:
        if cost == 'patch_variances':
            loss += weight * _patch_variances(opts, real, reconst)
        if cost == 'pixel_wise_l2_sq':
            loss += weight * _pixel_wise_l2_sq(opts, real, reconstr)

    return loss

def _patch_variances(opts, real, reconstr):
    """Cost is l2_sq difference between variances of patches in real
    and reconstructed images.
    """
    real_sq = real**2
    reconstr_sq = reconstr**2
    height, width, channels = [int(real.get_shape()[i]) for i in range(1,4)]

    kernel_sizes = opts['cost_kernel_sizes']
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

def _pixel_wise_l2_sq(opts, real, reconstr):
    loss = tf.reduce_sum(tf.square(real - reconstr), axis=[1, 2, 3])
    loss = tf.reduce_mean(loss)
