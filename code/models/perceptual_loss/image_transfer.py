__author__ = "charles"
__email__ = "charleschen2013@163.com"
from os import path as osp
import sys
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

sys.path.append(osp.join(sys.path[0], '../'))
sys.path.append(osp.join(sys.path[0], '../../'))
import argparse
import time
import numpy as np
import torch

from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from src.perceptual_loss.utils import normalize_tensor_transform, load_image, gram
from src.perceptual_loss.network import ImageTransformNet
from perceptual_loss.image_dataset import get_image_dataset, get_dataloader_from_dir
from perceptual_loss.train_config import config
from perceptual_loss.perceptual_loss import PerceptualLoss
from src.utils.visualizer import Visualizer
from src.utils.train_utils import model_accelerate, get_device
from tqdm import tqdm
from src.utils.logger import ModelSaver, Logger
from src.perceptual_loss.utils import save_image

if __name__ == '__main__':

    args = config()
    visualizer = Visualizer(keys=['img'])
    style_name = osp.split(args.style_image)[-1].split('.')[0]
    # style_name = osp.split(args.style_image)[-1].split('.')[0]
    logger = Logger(save_path=args.save, json_name=f'{style_name}')
    model_saver = ModelSaver(save_path=args.save, name_list=[
        f'{style_name}',
        f'{style_name}_{args.optimizer}',
        f'{style_name}_{args.scheduler}'
    ])
    criterion = PerceptualLoss(args)
    model = ImageTransformNet()
    model = model_accelerate(args, model)
    model_saver.load(f'{style_name}', model=model)

    optimizer = Adam(model.parameters(), lr=args.lr)
    model_saver.load(f'{style_name}_{args.optimizer}', model=optimizer)

    epoch_now = len(logger.get_data(key='loss'))
    device = get_device(args)

    ####
    data_loader = get_dataloader_from_dir(args)
    data_loader = tqdm(data_loader)
    model.eval()
    with torch.no_grad():
        # counter = 10
        for i, (imgs, path) in enumerate(data_loader):
            imgs = imgs.to(device)
            # counter -= 1
            # if counter < 0:
            #     break
            y_hat = model(imgs)
            for index in range(y_hat.size(0)):
                # data = torch.cat([y_hat[index], imgs[index]], dim=2)
                data = y_hat[index]
                transferred_image_name = path[0].split('.')[0]
                save_image(filename=f'{transferred_image_name}_{i}.jpg', data=data.cpu())
    ####
