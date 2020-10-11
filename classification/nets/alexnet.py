# -*- coding:utf-8 -*-

from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from typing import Set
from keras import Input
from keras.models import Model


class AlexNet(object):

    def __init__(self, input_shape, class_num: int=1000):
        self.input_shape = input_shape
        self.class_num = class_num

    def build_model(self):
        input_tensor = Input(shape=self.input_shape, name='input')

        x = Conv2D(96, (11, 11), strides=(4, 4), padding='same', activation='relu')(input_tensor)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

        x = Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

        x = Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)

        x = Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)

        x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)

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
    alexnet = AlexNet(input_shape=(227, 227, 3), class_num=10)
    model = alexnet.build_model()
