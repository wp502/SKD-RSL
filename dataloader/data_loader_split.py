"""
   CIFAR-10 CIFAR-100, Tiny-ImageNet data loader
"""

import random
import os
import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from .data_loader_common import *
from .cub200_2 import CUB2011Metric
from .cars196_2 import Cars196Metric
from .cub200 import CUB200
from .cars196 import Cars
from .cars196_extract_folder import CarstoFolder
from .cub200_extract_folder import CUB200toFolder
from torch.utils.data import DataLoader, Dataset


__all__ = ['fetch_dataloader_split']

def fetch_dataloader_split(types, args):
    # 该方法用在 work2 Our 方法中时。在同一个epoch中，先用 数据集来训练学生，然后用 数据集来反馈训练教师。
    # 同时，该方法不同于 data_loader_1.py 中 fetch_valNeeded_dataloader 方法。按照设定的比例在同一个类中完成切分。
    # 详见 train_one_epoch.py 中  方法
    # using random crops and horizontal flip for train sets

    # if args.dataset == 'cifar10':
    #     trainset, devset = fetch_cifar10(args)
    # elif args.dataset == 'cifar100':
    #     trainset, devset = fetch_cifar100(args)
    # elif args.dataset == 'tiny_imagenet':
    #     trainset, devset = fetch_tiny_imagenet(args)
    # elif args.dataset == 'imagenet':
    #     trainset, devset = fetch_imagenet(args)
    # elif args.dataset == 'cub200':
    #     trainset, devset = fetch_cub200(args)
    # elif args.dataset == 'cars196':
    #     trainset, devset = fetch_cars(args)
    trainset, devset = fetch_dataset(args)

    if types == 'train':

        # np.random.shuffle(indices)
        fwd_indices = []
        fb_indices = []
        sample_target = trainset.targets
        # 找到每个类对应的index
        targets_where = [np.where(np.array(sample_target) == i) for i in range(args.num_class)]
        # train_sampler = [where[i][0][-(int(np.floor(0.05 * len(where[i][0])))):] for i in range(100)]
        for i in range(args.num_class):
            # 在每个类上进行比例切分
            split = int(np.floor(args.fb_set_percent * len(targets_where[i][0])))
            fwd_indices.extend(list(targets_where[i][0][:-split]))
            fb_indices.extend(list(targets_where[i][0][-split:]))

        # shuffle 只能对 index 进行打乱
        np.random.seed(args.seed)
        np.random.shuffle(fwd_indices)
        np.random.shuffle(fb_indices)
        sampler_fwd = SubsetRandomSampler(fwd_indices)
        sampler_fb = SubsetRandomSampler(fb_indices)

        trainloader_fwd = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                  sampler=sampler_fwd, num_workers=args.num_workers, pin_memory=args.pin_memory)
        trainloader_fb = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                sampler=sampler_fb, num_workers=args.num_workers, pin_memory=args.pin_memory)

        return trainloader_fwd, trainloader_fb
    else:
        devloader = torch.utils.data.DataLoader(devset, batch_size=args.batch_size,
                                                shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
        return devloader