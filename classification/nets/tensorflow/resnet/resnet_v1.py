# -*- coding:utf-8 -*-

import tensorflow.compat.v1 as tf 

BATCH_NORM_DECAY = 0.997
BATCH_NORM_EPSILON = 1e-5
BATCH_SCALE = True


def fixed_padding(inputs, kernel_size):
    """
    Args:
        inputs: A tensor of size [batch, height_in, width_in, channels]
        kernel_size: The size of kernel which to be used in the conv2d or mak_pool2d.
            Should be a positive integer.

    Returns:
        output: A tensor with the same format.
    """

    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    output = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    return output


def batch_norm(inputs, training):

    return tf.layers.batch_normalization(
        inputs=inputs, axis=0, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON, center=True, 
        scale=BATCH_SCALE, training=training, fused=True
    )


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, padding=None):
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size)
    
    if padding is None:
        padding = ('SAME' if strides == 1 and isinstance(kernel_size, int) else "VALID")

    return tf.layers.conv2d(
        inputs=inputs, 
        filters=filters, 
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
    )


def building_block_v1(
    inputs,
    filters,
    training,
    projection_shortcut,
    strides
):
    """
    A single block for ResNetV1, without a bottleneck.
    Args:
        inputs: A tensor of size [batch, height_in, width_in, channels]
        filters: 
        training: A Boolean for whether the model is in training or inference
            mode. Needed for batch normalization.
        projection_shortcut:
        strides: The block's stride. 
    """

    shortcut = inputs

    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)
        shortcut = batch_norm(inputs=shortcut, training=training)

    residual = conv2d_fixed_padding(
        inputs=inputs,
        filters=filters,
        kernel_size=3, 
        strides=strides
    )
    residual = batch_norm(
        inputs=residual,
        training=training
    )
    residual = tf.nn.relu(residual)

    residual = conv2d_fixed_padding(
        inputs=residual,
        filters=filters,
        kernel_size=3, 
        strides=strides
    )
    residual = batch_norm(
        inputs=residual,
        training=training
    )
    
    residual += shortcut

    residual = tf.nn.relu(residual)

    return residual


def bottleneck_block_v1(
    inputs,
    filters,
    bottleneck_channel,
    training,
    projection_shortcut,
    strides
):

    """
    Bottleneck
    """

    shortcut = inputs

    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)
        shortcut = batch_norm(inputs=shortcut, training=training)

    residual = conv2d_fixed_padding(
        inputs=inputs,
        filters=bottleneck_channel,
        kernel_size=1, 
        strides=strides
    )
    residual = batch_norm(
        inputs=residual,
        training=training
    )
    residual = tf.nn.relu(residual)

    residual = conv2d_fixed_padding(
        inputs=inputs,
        filters=bottleneck_channel,
        kernel_size=3, 
        strides=strides
    )
    residual = batch_norm(
        inputs=residual,
        training=training
    )
    residual = tf.nn.relu(residual)

    residual = conv2d_fixed_padding(
        inputs=inputs,
        filters=filters,
        kernel_size=1, 
        strides=strides
    )
    residual = batch_norm(
        inputs=residual,
        training=training
    )
    

    residual += shortcut

    residual = tf.nn.relu(residual)

    return residual
