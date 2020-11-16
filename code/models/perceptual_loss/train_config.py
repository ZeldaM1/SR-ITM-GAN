# -*- coding: utf-8 -*-
__author__ = "charles"
__email__ = "charleschen2013@163.com"

import argparse
import os
import torch
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import time
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader


def config():
    parser = argparse.ArgumentParser(description='Trains GAN on CIFAR',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Positional arguments
    parser.add_argument('--style_image', type=str, default='./style_imgs/TheStarryNight.jpg',
                        help='targe style image')
    parser.add_argument('--img_dir', type=str, default='./style_imgs/sea.png', help='image to be transferred')
    parser.add_argument('--style', type=str, default='TheStarryNight',
                        choices=['TheStarryNight', 'horse', 'flower', 'flowers'],
                        help='pre_trained built in style')
    parser.add_argument('--image_dataset', type=str, default='/Users/chenlinwei/dataset/SBD/benchmark_RELEASE/dataset/',
                        help='Root for the Cifar dataset.')
    # Optimization options
    parser.add_argument('--optimizer', '-op', type=str, default='adam', help='Optimizer to train model.')
    parser.add_argument('--epochs', '-e', type=int, default=50, help='Number of epochs to train.')
    parser.add_argument('--batch_size', '-b', type=int, default=2, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.001, help='The Learning Rate.')

    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
    parser.add_argument('--test_batch_size', type=int, default=10)
    parser.add_argument('--scheduler', type=str, default='multi_step')
    parser.add_argument('--milestones', type=int, nargs='+', default=[25, 40],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    # Checkpoints
    parser.add_argument('--save', '-s', type=str, default=None, help='Folder to save checkpoints.')
    parser.add_argument('--save_steps', '-ss', type=int, default=200, help='steps to save checkpoints.')
    parser.add_argument('--load', '-l', type=str, help='Checkpoint path to resume / test.')
    parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')

    parser.add_argument('--STYLE_WEIGHT', type=float, default=1e6, help='STYLE_WEIGHT')
    parser.add_argument('--CONTENT_WEIGHT', type=float, default=1e0, help='CONTENT_WEIGHT')
    parser.add_argument('--TV_WEIGHT', type=float, default=2e-2, help='TV_WEIGHT')
    # Acceleration
    parser.add_argument('--gpu', type=int, default=1, help='0 = CPU.')
    parser.add_argument('--prefetch', type=int, default=12, help='Pre-fetching threads.')
    # i/o
    parser.add_argument('--log', type=str, default=None, help='Log folder.')

    parser.add_argument('--crop_size', type=int, default=256, help='The size of image.')
    parser.add_argument('--aug', type=str, default='crop', help='The size of image.')

    parser.add_argument('--display', type=int, default=0, help='display or not')
    parser.add_argument('--content_layer', type=int, default=2,
                        help='choose No.content_layer of vgg layer for content loss')
    args = parser.parse_args()
    if args.save is None:
        args.save = f'../../transfer_model'
    if args.log is None:
        args.log = args.save
    args.scheduler = f'{args.optimizer}_{args.scheduler}'
    return args
