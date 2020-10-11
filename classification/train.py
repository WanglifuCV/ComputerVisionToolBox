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
from nets.vgg import VGGNet19
from nets.alexnet import AlexNet
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
import pickle
import argparse
import json
from contextlib import redirect_stdout

root_dir = None


def arg_parse():
    parser = argparse.ArgumentParser(description='Parse argments.')
    parser.add_argument('--optimizer', type=str, default='sgd', metavar='optimizer', help='Optimizer')
    parser.add_argument('--backbone', type=str, default='vgg19', metavar='backbone', help='Backbone')
    parser.add_argument('--learning_rate', type=float, default=1e-3, metavar='learning_rate', help='Learning rate')
    parser.add_argument('--gpu_list', type=str, default='0', metavar='gpu_list', help='GPU list')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='momentum', help='Momentum of SGD')
    parser.add_argument('--batch_norm_type', type=str, default='before_activation', metavar='batch_norm_type', help='Batch norm type')
    parser.add_argument('--steps_per_epoch', type=int, default=1000, metavar='steps_per_epoch', help='Steps per epoch')
    parser.add_argument('--epochs', type=int, default=50, metavar='epochs', help='Epochs')
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size', help='Batch size')
    parser.add_argument('--model_save_dir', type=str, default=root_dir, metavar='model_save_dir', help='Model save dir')
    parser.add_argument('--category_num', type=int, default=2, metavar='category_num', help='Category num')
    parser.add_argument('--input_size', type=int, default=224, metavar='input_size', help='Input size')
    return parser


def dir_prepare():
    if osp.exists(Args.model_save_dir):
        shutil.rmtree(Args.model_save_dir)
    
    os.mkdir(Args.model_save_dir)
    os.mkdir(osp.join(Args.model_save_dir, 'log'))
    os.mkdir(osp.join(Args.model_save_dir, 'model'))
    os.mkdir(osp.join(Args.model_save_dir, 'history'))


def save_args():
    with open(osp.join(Args.model_save_dir, 'args.json'), 'w') as file_writer:
        json.dump(vars(Args), file_writer, indent=4, ensure_ascii=False)


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


def inference():
    if Args.backbone.lower() == 'vgg19':
        model = VGGNet19(input_shape=(Args.input_size, Args.input_size, 3),
                         batch_norm=Args.batch_norm_type,
                         classes=Args.category_num).model
    elif Args.backbone.lower() == 'vgg19_app':
        model = VGG19(weights=None,
                      input_shape=(Args.input_size, Args.input_size, 3),
                      classes=Args.category_num)
    elif Args.backbone.lower() == 'resnet50':
        model = ResNet50(weights=None,
                         input_shape=(Args.input_size, Args.input_size, 3),
                         classes=Args.category_num)
    elif Args.backbone.lower() == 'alexnet':
        alexnet = AlexNet(input_shape=(Args.input_size, Args.input_size, 3), class_num=Args.category_num)
        model = alexnet.build_model()
    else:
        model = None

    if model is not None:
        model.summary()

        with open(osp.join(Args.model_save_dir, 'model.json'), 'w') as file_writer:
            json.dump(json.loads(model.to_json()), file_writer, indent=4, ensure_ascii=False)

        # model.summary(print_fn=myprint)

        with open(osp.join(Args.model_save_dir, 'modelsummary.txt'), 'w') as f:
            with redirect_stdout(f):
                model.summary()

    return model


def myprint(s):
    with open(osp.join(Args.model_save_dir, 'model.txt') ,'w+') as file_writer:
        print(s, file=file_writer)


def make_gpu_config():
    os.environ['CUDA_VISIBLE_DEVICES'] = Args.gpu_list
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7, allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)

    sess = tf.Session(config=config)
    KTF.set_session(sess)


def train(train_generator, test_generator):
    make_gpu_config()

    model = inference()
    if model is None:
        print('Model error : {}'.format(Args.backbone))
    else:
        model.summary()

    if Args.optimizer.lower() == 'sgd':
        opt = optimizers.SGD(lr=Args.learning_rate,
                             momentum=Args.momentum)
    elif Args.optimizer.lower() == 'rmsprop':
        opt = optimizers.RMSprop(lr=Args.learning_rate)
    else:
        opt = None
        print('Optimizer error : {}'.format(Args.optimizer))

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['acc'])

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=Args.steps_per_epoch,
        epochs=Args.epochs,
        validation_data=test_generator,
        validation_steps=50,
        callbacks=callbacks_config()
    )

    with open(osp.join(Args.model_save_dir, 'history', 'history'), 'wb') as file_writer:
        pickle.dump(history, file_writer)
    return history

def callbacks_config():
    callbacks = [
        TensorBoard(
            log_dir=osp.join(Args.model_save_dir, 'log'),
            histogram_freq=0,
            write_graph=True,
            batch_size=32
        ), 
        ModelCheckpoint(
            osp.join(Args.model_save_dir, 'model', '{}.h5'.format(Args.backbone)),
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

ArgParser = arg_parse()
Args = ArgParser.parse_args()



if __name__ == '__main__':
    dir_prepare()
    save_args()
    train_folder = '/home/lifu/data/datasets/classification/dogs-vs-cats/train'
    val_folder = '/home/lifu/data/datasets/classification/dogs-vs-cats/validation'
    train_gen, test_gen = generate_data_flow(train_data_folder=train_folder,
                                             val_data_folder=val_folder)
    history = train(train_generator=train_gen, test_generator=test_gen)
