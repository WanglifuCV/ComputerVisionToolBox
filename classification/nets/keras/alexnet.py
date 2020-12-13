# -*- coding:utf-8 -*-

from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from typing import Set
from keras import Input
from keras.models import Model
from keras import regularizers
from utils.argments_parser import Args, arg_parser


class AlexNet(object):

    def __init__(self, input_shape, class_num: int=1000):
        self.input_shape = input_shape
        self.class_num = class_num

    def build_model(self, dataset: str='normal', architecture: str='ordinary'):

        if Args.regularizer is not None:
            regularizer = regularizers.l2(Args.regularizer)
        else:
            regularizer = None

        input_tensor = Input(shape=self.input_shape, name='input')

        if dataset.lower() in ('cifar10', 'cifar100'):
            x = Conv2D(96, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=regularizer)(input_tensor)
        else:
            x = Conv2D(96, (11, 11), strides=(4, 4), padding='same', activation='relu', kernel_regularizer=regularizer)(input_tensor)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

        if dataset.lower() in ('cifar10', 'cifar100'):
            x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=regularizer)(x)
        else:
            x = Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=regularizer)(x)

        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

        x = Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=regularizer)(x)

        x = Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=regularizer)(x)

        x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=regularizer)(x)

        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

        x = Flatten()(x)

        x = Dense(4096, activation='relu')(x)

        x = Dropout(0.5)(x)

        x = Dense(4096, activation='relu')(x)

        x = Dropout(0.5)(x)

        output_tensor = Dense(self.class_num, activation='softmax')(x)

        model = Model(input_tensor, output_tensor)

        model.summary()

        return model

    def build_model_cifar(self, dataset: str='normal', architecture: str='ordinary'):

        if Args.regularizer is not None:
            regularizer = regularizers.l2(Args.regularizer)
        else:
            regularizer = None

        input_tensor = Input(shape=self.input_shape, name='input')

        x = Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=regularizer)(input_tensor)

        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

        x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=regularizer)(x)

        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

        x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=regularizer)(x)

        x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=regularizer)(x)

        x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=regularizer)(x)

        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

        x = Flatten()(x)

        x = Dense(4096, activation='relu')(x)

        x = Dropout(0.5)(x)

        x = Dense(4096, activation='relu')(x)

        x = Dropout(0.5)(x)

        output_tensor = Dense(self.class_num, activation='softmax')(x)

        model = Model(input_tensor, output_tensor)

        model.summary()

        return model


if __name__ == '__main__':
    alexnet = AlexNet(input_shape=(32, 32, 3), class_num=100)
    model = alexnet.build_model(dataset='cifar100')
