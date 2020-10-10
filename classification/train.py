# -*- coding:utf-8 -*-

from keras import optimizers
import os, shutil
import os.path as osp
import tensorflow as tf 
import keras.backend.tensorflow_backend as KTF
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint
import matplotlib.pyplot as plt
import tensorflow as tf
from models.vgg import VGGNet19
from models.alexnet import AlexNet
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
import pickle
import argparse


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('optimizer', 'sgd', 'Optimizer')
tf.app.flags.DEFINE_string('backbone', 'vgg19', 'Backbone')
tf.app.flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate')
tf.app.flags.DEFINE_string('gpu_list', '0', 'GPU')
tf.app.flags.DEFINE_float('momentum', 0.9, 'Momentum of SGD')
tf.app.flags.DEFINE_string('batch_norm', 'activation', 'Batch norm')
tf.app.flags.DEFINE_integer('steps_per_epoch', 320, 'Steps per epoch')
tf.app.flags.DEFINE_integer('epochs', 50, 'Epochs')
tf.app.flags.DEFINE_integer('batch_size', 16, 'Batch size')
tf.app.flags.DEFINE_string('log_folder_dir', './logs/cats-vs-dogs/vgg19_keras_after_activation/', 'Tensorboard log dir')
tf.app.flags.DEFINE_integer('category_num', 2, 'Category number')
tf.app.flags.DEFINE_string('history_save_dir', 'alexnet', 'History save dir.')
tf.app.flags.DEFINE_integer('input_size', 224, 'input shape')
tf.app.flags.DEFINE_string('model_save_dir', 'vgg19.h5', 'model_save_dir')


def arg_parse():
    parser = argparse.ArgumentParser(description='Parse argments.')
    parser.add_argument('--optimizer', type=str, default='sgd', metavar='optimizer', help='Optimizer')
    parser.add_argument('--backbone', type=str, default='vgg19', metavar='backbone', help='Backbone')
    parser.add_argument('--learning_rate', type=float, default=1e-3, metavar='learning_rate', help='Learning rate')
    parser.add_argument('--gpu_list', type=str, default='0', metavar='gpu_list', help='GPU list')
    parser.add_argument('--gpu_list', type=str, default='0', metavar='gpu_list', help='GPU list')

def generate_data_flow(train_data_folder, val_data_folder):
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
        target_size=(FLAGS.input_size, FLAGS.input_size),
        batch_size=FLAGS.batch_size,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        directory=val_data_folder,
        target_size=(FLAGS.input_size, FLAGS.input_size),
        batch_size=FLAGS.batch_size,
        class_mode='categorical'
    )

    return train_generator, val_generator


def inference():
    if FLAGS.backbone.lower() == 'vgg19':
        model = VGGNet19(input_shape=(FLAGS.input_size, FLAGS.input_size, 3),
                         batch_norm=FLAGS.batch_norm,
                         classes=FLAGS.category_num).model
    elif FLAGS.backbone.lower() == 'vgg19_app':
        model = VGG19(weights=None,
                      input_shape=(FLAGS.input_size, FLAGS.input_size, 3),
                      classes=FLAGS.category_num)
    elif FLAGS.backbone.lower() == 'resnet50':
        model = ResNet50(weights=None,
                         input_shape=(FLAGS.input_size, FLAGS.input_size, 3),
                         classes=FLAGS.category_num)
    elif FLAGS.backbone.lower() == 'alexnet':
        alexnet = AlexNet(input_shape=(FLAGS.input_size, FLAGS.input_size, 3), class_num=FLAGS.category_num)
        model = alexnet.build_model()
    else:
        model = None

    if model is not None:
        model.summary()

    return model


def make_gpu_config():
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7, allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)

    sess = tf.Session(config=config)
    KTF.set_session(sess)


def train(train_generator, test_generator):
    make_gpu_config()

    model = inference()
    if model is None:
        print('Model error : {}'.format(FLAGS.backbone))
    else:
        model.summary()

    if FLAGS.optimizer.lower() == 'sgd':
        opt = optimizers.SGD(lr=FLAGS.learning_rate,
                             momentum=FLAGS.momentum)
    elif FLAGS.optimizer.lower() == 'rmsprop':
        opt = optimizers.RMSprop(lr=FLAGS.learning_rate)
    else:
        opt = None
        print('Optimizer error : {}'.format(FLAGS.optimizer))

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['acc'])

    if not osp.isdir(FLAGS.log_folder_dir):
        os.mkdir(FLAGS.log_folder_dir)

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=FLAGS.steps_per_epoch,
        epochs=FLAGS.epochs,
        validation_data=test_generator,
        validation_steps=50,
        callbacks=callbacks_config()
    )

    with open(osp.join('history', FLAGS.history_save_dir), 'wb') as file_writer:
        pickle.dump(history, file_writer)
    return history

def callbacks_config():
    callbacks = [
        TensorBoard(
            log_dir=FLAGS.log_folder_dir,
            histogram_freq=0,
            write_graph=True,
            batch_size=32
        ), 
        ModelCheckpoint(
            osp.join('models', FLAGS.model_save_dir),
            monitor='val_loss',
            mode='min',
            save_weights_only=True,
            save_best_only=True,
            verbose=1, 
            period=1
        )
    ]
    return callbacks

def plot_history(history, history_save_name):
    history_dict = history.history

    train_loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    train_acc = history_dict['acc']
    val_acc = history_dict['val_acc']

    epochs = range(1, len(train_loss) + 1)

    plt.plot(epochs, train_loss, color = 'blue', label = 'Training loss', linestyle = '-')
    plt.plot(epochs, train_acc, color='blue', label='Training accuracy', linestyle='--')
    plt.plot(epochs, val_loss, color='red', label='validation loss', linestyle='-')
    plt.plot(epochs, val_acc, color='red', label='validation accuracy', linestyle='--')

    plt.title('Training and Validation loss and accuracy.')

    plt.xlabel('Epochs')
    plt.ylabel('Loss and accuracy')
    plt.legend()

    plt.savefig(history_save_name)

    plt.show()


if __name__ == '__main__':
    train_folder = '/home/lifu/data/datasets/classification/dogs-vs-cats/train'
    val_folder = '/home/lifu/data/datasets/classification/dogs-vs-cats/validation'
    train_gen, test_gen = generate_data_flow(train_data_folder=train_folder,
                                             val_data_folder=val_folder)
    history = train(train_generator=train_gen, test_generator=test_gen)
