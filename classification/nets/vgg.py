# -*- coding:utf-8 -*-

from keras.models import Sequential, Model
from keras import layers, Input
import os
from keras.applications.vgg19 import VGG19

class VGGNet(object):

    def __init__(self,
                 input_shape,
                 classes=1000,
                 batch_norm=None,
                 pooling_method='fc'):

        self.input_shape = input_shape
        self.classes = classes
        self.batch_norm = batch_norm
        self.pooling = pooling_method

    def Conv2DLayer(self,
                    input_tensor,
                    filters,
                    name_scope,
                    conv_filter_idx,
                    kernel_size=(3,3),
                    padding='same',
                    activation='relu',
                    batch_norm=None):
        if batch_norm is not None and batch_norm.lower() == 'before_activation':
            """
            表示batch_norm是加载激活函数之前的
            """
            x = layers.Conv2D(filters=filters,
                                          kernel_size=kernel_size,
                                          padding=padding,
                                          activation=None,
                                          name='{}_conv{}x{}_{}'.format(name_scope,
                                                                        kernel_size[0],
                                                                        kernel_size[1],
                                                                        conv_filter_idx))(input_tensor)
            x = layers.BatchNormalization(name='{}_bn_{}'.format(name_scope, conv_filter_idx))(x)
            output_tensor = layers.Activation(name='{}_{}_{}'.format(name_scope, activation, conv_filter_idx),
                                              activation=activation)(x)
        elif batch_norm is not None and batch_norm.lower() == 'after_activation':
            raise RuntimeError('Activation type error : {}'.format(batch_norm.lower()))
            # """
            #             表示batch_norm是加载激活函数之后的
            #             """
            # x = layers.Conv2D(filters=filters,
            #                   kernel_size=kernel_size,
            #                   padding=padding,
            #                   activation=activation,
            #                   name='{}_conv{}x{}_{}'.format(name_scope,
            #                                                 kernel_size[0],
            #                                                 kernel_size[1],
            #                                                 conv_filter_idx))(input_tensor)
            # output_tensor = layers.BatchNormalization(name='{}_bn_{}'.format(name_scope, conv_filter_idx))(x)
        else:
            output_tensor = layers.Conv2D(filters=filters,
                                          kernel_size=kernel_size,
                                          padding=padding,
                                          activation=activation,
                                          name='{}_conv{}x{}_{}'.format(name_scope,
                                                                        kernel_size[0],
                                                                        kernel_size[1],
                                                                        conv_filter_idx))(input_tensor)

        return output_tensor


class VGGNet19(VGGNet):

    def __init__(self,
                 input_shape,
                 classes=1000,
                 batch_norm=None, 
                 pooling_method='fc'):
        VGGNet.__init__(self, input_shape=input_shape, classes=classes, batch_norm=batch_norm, pooling_method=pooling_method)

    def build(self):
        input_tensor = Input(shape=self.input_shape, name='Input')

        x = input_tensor

        #  Block 1
        for i in range(2):
            x = self.Conv2DLayer(x,
                                 filters=64,
                                 kernel_size=(3,3),
                                 padding='same',
                                 activation='relu',
                                 batch_norm=self.batch_norm,
                                 name_scope='block1',
                                 conv_filter_idx=i+1)

        x = layers.MaxPooling2D(pool_size=(2,2), name='block1_pooling')(x)

        #  Block 2
        for i in range(2):
            x = self.Conv2DLayer(x,
                                 filters=128,
                                 kernel_size=(3, 3),
                                 padding='same',
                                 activation='relu',
                                 batch_norm=self.batch_norm,
                                 name_scope='block2',
                                 conv_filter_idx=i + 1)

        x = layers.MaxPooling2D(pool_size=(2, 2), name='block2_pooling')(x)

        #  Block 3
        for i in range(4):
            x = self.Conv2DLayer(x,
                                 filters=256,
                                 kernel_size=(3, 3),
                                 padding='same',
                                 activation='relu',
                                 batch_norm=self.batch_norm,
                                 name_scope='block3',
                                 conv_filter_idx=i + 1)

        x = layers.MaxPooling2D(pool_size=(2, 2), name='block3_pooling')(x)

        #  Block 4
        for i in range(4):
            x = self.Conv2DLayer(x,
                                 filters=512,
                                 kernel_size=(3, 3),
                                 padding='same',
                                 activation='relu',
                                 batch_norm=self.batch_norm,
                                 name_scope='block4',
                                 conv_filter_idx=i + 1)

        x = layers.MaxPooling2D(pool_size=(2, 2), name='block4_pooling')(x)

        #  Block 5
        for i in range(4):
            x = self.Conv2DLayer(x,
                                 filters=512,
                                 kernel_size=(3, 3),
                                 padding='same',
                                 activation='relu',
                                 batch_norm=self.batch_norm,
                                 name_scope='block5',
                                 conv_filter_idx=i + 1)

        x = layers.MaxPooling2D(pool_size=(2, 2), name='block5_pooling')(x)

        x = layers.Flatten(name='flatten')(x)
        x = layers.Dense(units=4096, activation='relu', name='fully_connect_1')(x)
        x = layers.Dense(units=4096, activation='relu', name='fully_connect_2')(x)
        output_tensor = layers.Dense(units=self.classes,
                                     activation='softmax',
                                     name='predictions')(x)

        model = Model(input_tensor, output_tensor)

        return model

    def build_cifar(self):
        input_tensor = Input(shape=self.input_shape, name='Input')

        x = input_tensor

        #  Block 1
        for i in range(2):
            x = self.Conv2DLayer(x,
                                 filters=64,
                                 kernel_size=(3,3),
                                 padding='same',
                                 activation='relu',
                                 batch_norm=self.batch_norm,
                                 name_scope='block1',
                                 conv_filter_idx=i+1)

        x = layers.MaxPooling2D(pool_size=(2,2), name='block1_pooling')(x)

        #  Block 2
        for i in range(2):
            x = self.Conv2DLayer(x,
                                 filters=128,
                                 kernel_size=(3, 3),
                                 padding='same',
                                 activation='relu',
                                 batch_norm=self.batch_norm,
                                 name_scope='block2',
                                 conv_filter_idx=i + 1)

        x = layers.MaxPooling2D(pool_size=(2, 2), name='block2_pooling')(x)

        #  Block 3
        for i in range(4):
            x = self.Conv2DLayer(x,
                                 filters=256,
                                 kernel_size=(3, 3),
                                 padding='same',
                                 activation='relu',
                                 batch_norm=self.batch_norm,
                                 name_scope='block3',
                                 conv_filter_idx=i + 1)

        x = layers.MaxPooling2D(pool_size=(2, 2), name='block3_pooling')(x)

        if self.pooling == 'global_average_pooling':
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dense(units=4096, activation='relu', name='fully_connect_1')(x)
            x = layers.Dense(units=4096, activation='relu', name='fully_connect_2')(x)
            output_tensor = layers.Dense(units=self.classes,
                                        activation='softmax',
                                        name='predictions')(x)
        else:
            x = layers.Flatten(name='flatten')(x)
            x = layers.Dense(units=4096, activation='relu', name='fully_connect_1')(x)
            x = layers.Dense(units=4096, activation='relu', name='fully_connect_2')(x)
            output_tensor = layers.Dense(units=self.classes,
                                        activation='softmax',
                                        name='predictions')(x)

        model = Model(input_tensor, output_tensor)

        return model


if __name__ == '__main__':
    vggnet = VGGNet19(
        input_shape=(32, 32, 3),
        classes=1000,
        batch_norm='activation',
        pooling_method='global_average_pooling'
        )
    model = vggnet.build_cifar()
    model.summary()