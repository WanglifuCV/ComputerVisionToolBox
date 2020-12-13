# -*- coding:utf-8 -*-

import tf_slim as slim
import tensorflow.compat.v1 as tf


def subsample(inputs, stride, scope=None):
    """
    Subsamples the input along the spatial dimensions.

    Args:
        inputs: A tensor of size [batch, height_in, width_in, channels]
        factor: The subsampling factor.
        scope: Optional variable_scope.

    Returns:
        output: A tensor of size [batch, height_out, width_out, channels]
    """

    if stride == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [stride, stride], stride=stride, scope=scope)


def conv2d_same(inputs, depth_output, kernel_size, stride, rate=1, scope=None):
    """
    Args: 
        inputs: A 4-D tensor of size [batch, height_in, width_in, channels]
        depth_output: An integer, number of output filters.
        stride: An integer, the output stride.
        rate: An integer, ???

    Returns:
        output: A 4-D tensor of size [batch, height_out, width_out, channels] with
            the convolution output.

    Notes: 为空洞卷积特意定制的，如果不考虑空洞卷积，那么这个可以没有。
    """

    if stride == 1:
        return slim.conv2d(inputs, depth_output, kernel_size=kernel_size, 
            rate=rate, padding='same', scope=scope
        )
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        inputs = tf.pad(
            tensor=inputs, 
            paddings = [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]]
        )

        return slim.conv2d(inputs, depth_output, kernel_size=kernel_size, stride=stride,
            rate=rate, padding='VALID', scope=scope
        )


