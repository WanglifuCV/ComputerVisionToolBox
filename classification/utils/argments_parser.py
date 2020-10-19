# -*- coding:utf-8 -*-

import argparse
import os.path as osp
import json

class ArgmentParser(object):

    def __init__(self):
        self.args = self.arg_parse()

    def arg_parse(self):
        root_dir = '/home/wanglifu/learning/Deep-Learning-with-Python/ComputerVisionToolBox/classification/models/cats-vs-dogs/check'
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
        parser.add_argument('--train_folder', type=str, default=None, metavar='train_folder', help='Train folder')
        parser.add_argument('--val_folder', type=str, default=None, metavar='val_folder', help='Validation folder')
        parser.add_argument('--test_folder', type=str, default=None, metavar='test_folder', help='Test folder')
        parser.add_argument('--dataset_name', type=str, default=None, metavar='dataset_name', help='Dataset name')
        parser.add_argument('--pre_train_model', type=str, default=None, metavar='pre_train_model', help='Pre train model')
        parser.add_argument('--regularizer', type=float, default=None, metavar='regularizer', help='Regularizer weight')
        parser.add_argument('--architecture', type=str, default=None, metavar='architecture', help='同一个网络里面的不同架构的选择')
        parser.add_argument('--lr_decay', type=float, default=None, metavar='lr_decay', help='学习率衰减的权重')
        parser.add_argument('--final_pooling', type=str, default=None, metavar='final_pooling', help='最后一层的pooling方法')
        return parser.parse_args()

    def save_args(self):
        with open(osp.join(self.args.model_save_dir, 'args.json'), 'w') as file_writer:
            json.dump(vars(self.args), file_writer, indent=4, ensure_ascii=False)

arg_parser = ArgmentParser()
Args = arg_parser.args