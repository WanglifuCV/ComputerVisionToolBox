# -*- coding:utf-8 -*-
import sys
import os.path as osp
sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), '../'))
from keras.preprocessing.image import ImageDataGenerator
from utils.argments_parser import Args
from keras.datasets import cifar10, cifar100, mnist
from keras.utils import to_categorical


def generate_data_flow_from_folder(train_data_folder, val_data_folder):
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=10,
                                       horizontal_flip=True,
                                       zoom_range=0.2,
                                       shear_range=0.2,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2)
    val_datagen = ImageDataGenerator(rescale=1./ 255)

    train_generator = train_datagen.flow_from_directory(
        directory=train_data_folder,
        target_size=(Args.input_size, Args.input_size),
        batch_size=Args.batch_size,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        directory=val_data_folder,
        target_size=(Args.input_size, Args.input_size),
        batch_size=Args.batch_size,
        class_mode='categorical'
    )

    return train_generator, val_generator


def generate_data_flow_from_dataset(dataset_name: str, valid_size: float):
    if dataset_name.lower() == 'cifar10':
        (image_train, label_train), (image_test, label_test) = cifar10.load_data()
        value_scale = 255
        num_classes = 10
    elif dataset_name.lower() == 'cifar100':
        (image_train, label_train), (image_test, label_test) = cifar100.load_data(label_mode='fine')
        value_scale = 255
        num_classes = 100
    elif dataset_name.lower() == 'mnist':
        (image_train, label_train), (image_test, label_test) = mnist.load_data()
        value_scale = 1
        num_classes = 10
    else:
        raise RuntimeError('dataset_name error : {}'.format(dataset_name))

    val_num = int(len(image_train) * valid_size)
    image_val = image_train[:val_num]
    label_val = label_train[:val_num]
    image_train = image_train[val_num:]
    label_train = label_train[val_num:]

    train_datagen = ImageDataGenerator(rescale=1./value_scale,
                                       rotation_range=10,
                                       horizontal_flip=True,
                                       zoom_range=0.2,
                                       shear_range=0.2,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2)
    val_datagen = ImageDataGenerator(rescale=1./ value_scale)
    test_datagen = ImageDataGenerator(rescale=1./ value_scale)

    train_datagen.fit(image_train)
    val_datagen.fit(image_val)
    test_datagen.fit(image_test)

    label_train = to_categorical(label_train, num_classes=num_classes)
    label_val = to_categorical(label_val, num_classes=num_classes)
    label_test = to_categorical(label_test, num_classes=num_classes)

    data_flow_train = train_datagen.flow(image_train, label_train)
    data_flow_val = val_datagen.flow(image_val, label_val)
    data_flow_test = val_datagen.flow(image_test, label_test)

    return data_flow_train, data_flow_val, data_flow_test


if __name__ == '__main__':
    generate_data_flow_from_dataset(dataset_name='mnist', valid_size=0.1)