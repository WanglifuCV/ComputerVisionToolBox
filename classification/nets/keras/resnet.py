# -*- coding:utf-8 -*-

from keras import Input
from keras.layers import Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, concatenate, Activation, ZeroPadding2D, add, Flatten, GlobalAveragePooling2D
from keras.models import Model
import numpy as np

def Conv2d_BatchNorm(
    input_tensor,
    filter_num,
    kernel_size,
    strides=(1, 1),
    padding='same', 
    name=None
):
    if 