# -*- coding:utf-8 -*-

import tensorflow as tf
from tflearn.layers.conv import global_avg_pool

BATCH_NORM_DECAY=0.997
BATCH_NORM_EPLISON=1e-5
BATCH_SCALE=True


def conv_layer(input, filter, kernel, stride, layer_name, padding='SAME'):
    with tf.name_scope(layer_name):
        output = tf.layers.conv2d(input, filter, kernel, stride, padding)

        return output


def Global_Average_Pooling(inputs):
    return global_avg_pool(inputs)


def Fully_Connect(inputs, units):
    return tf.layers.dense(inputs, units=units)


def Squeeze_Excitation_layer(inputs, out_dim, ratio, layer_scope):
    with tf.name_scope(layer_scope):
        squeeze = Global_Average_Pooling(inputs=inputs)

        excitation = Fully_Connect(inputs=squeeze, units= out_dim // ratio)

        excitation = tf.nn.relu(excitation)

        excitation = Fully_Connect(excitation, units=out_dim)

        excitation = tf.nn.sigmoid(excitation)

        excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])

        scale = inputs * excitation

        return scale


def BatchNormalization(input, is_training):
    return tf.layers.batch_normaliztion(
        input, axis=0, momentum=BATCH_NORM_DECAY,
        eplison=BATCH_NORM_EPLISON, 
        center=True,
        scale=BATCH_SCALE, 
        training=is_training,
        fused=True
    )


def inverted_block(input, input_filters, output_filters, expand_ratio, stride, scope):

    # TODO: 可以再优化

    with tf.name_scope(scope):
        res_block = conv_layer(input, filter=input_filters * expand_ratio, kernel=[1, 1], stride=1)

        res_block = BatchNormalization(res_block, is_training=True)

        res_block = tf.layers.separable_conv2d(
            res_block, filters=input_filters * expand_ratio,
            kernel_size=3, 
            strides=stride,
            padding='SAME'
        )

        res_block = BatchNormalization(res_block, is_training=True)

        res_block = tf.nn.relu(res_block)

        res_block = conv_layer(input=res_block, filter=output_filters, kernel=1)

        res_block = BatchNormalization(input=res_block, is_training=True)

        if input_filters != output_filters:
            input = conv_layer(input=input, filter=output_filters, kernel=1, stride=stride)

        return tf.add(res_block, input)


def inverted_eff_block(input, input_filters, output_filters, expend_ratio, stride, scope):
    pass