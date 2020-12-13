# -*- coding:utf-8 -*-

import tf_slim as slim
import tensorflow.compat.v1 as tf
import sys
sys.path.append('/home/wanglifu/learning/Deep-Learning-with-Python/ComputerVisionToolBox')
from classification.nets.tensorflow_slim.resnet.resnet_utils import conv2d_same, subsample


@slim.add_arg_scope
def bottleneck(
    inputs,
    depth,
    depth_bottleneck,
    stride,
    batch_norm_params,
    rate=1,
    output_collections=None,
    scope=None,
    use_bounded_activations=False
    ):

    """
    Bottleneck residual unit variant with BN after convolutions.

    Args:
        inputs: A tensor of size [batch, height, width, channels]
        depth: The depth of the ResNet unit output.
        depth_bottleneck: The depth of the bottleneck layers.
        stride: The ResNet unit's stride. Determines the amount of
            downsampling of the units output compared to its input.
        rate: An integer, for atrous convolution(空洞卷积)
        outputs_collections: Collection to add the ResNet unit output.
        use_bounded_activations: Whether or not to use bounded activations.

    Returns: 
        The ResNet unit's output

    """

    with slim.arg_scope([slim.conv2d], 
                        normalizer_fn=slim.batch_norm,
                        weights_regularizer=None,
                        weights_initializer= tf.contrib.layers.xavier_initializer(),
                        biases_initializer = tf.zeros_initializer()):

        with slim.arg_scope([slim.batch_norm], **batch_norm_params):

            with tf.variable_scope(scope, 'bottlenect_v1', [inputs]) as sc:
                depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
                if depth == depth_in:
                    shortcut = subsample(inputs=inputs, stride=stride, scope='shortcut')
                else:
                    shortcut = slim.conv2d(
                        inputs,
                        depth,
                        [1, 1], 
                        stride=stride,
                        activation_fn=tf.nn.relu6 if use_bounded_activations else None,
                        scope='shortcut'
                    )

                residual = slim.conv2d(inputs, depth_bottleneck, [1, 1], stride=1, scope='conv1')
                residual = conv2d_same(residual, depth_bottleneck, 3, stride, rate=rate, scope='conv2')
                residual = slim.conv2d(residual, depth, [1, 1], stride=1, activation_fn=None, scope='conv3')

                if use_bounded_activations:
                    residual = tf.clip_by_value(residual, -6.0, 6.0)
                    output = tf.nn.relu6(shortcut + residual)
                else:
                    output = tf.nn.relu(shortcut + residual)
                
                return slim.utils.collect_named_outputs(
                    output_collections,
                    sc.name, 
                    output
                )


if __name__ == '__main__':
    intput_placeholder = tf.placeholder(shape=[4, 224, 224, 3], dtype=tf.float32)

    batch_norm_params = {
        'decay': 0.997,
        'epsilon': 1e-5,
        'scale': True,
        'is_training': True
    }

    output = bottleneck(
        inputs=intput_placeholder,
        depth= 32, 
        depth_bottleneck=16,
        batch_norm_params=batch_norm_params,
        stride=1, 
        rate=1,
        scope='backbone'
    )
    print(output)