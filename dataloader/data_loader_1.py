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
from .caltech import Caltech101, Caltech256

__all__ = ['fetch_dataloader']

def fetch_dataloader(types, args):
    # using random crops and horizontal flip for train sets

    trainset, devset = fetch_dataset(args)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory)

    if args.distill == 'our':
        feedback_trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                           shuffle=True, num_workers=args.num_workers,
                                                           pin_memory=args.pin_memory)

    devloader = torch.utils.data.DataLoader(devset, batch_size=args.batch_size,
                                            shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)

    if args.distill == 'our':
        if types == 'train':
            return trainloader, feedback_trainloader
        else:
            return devloader
    else:
        if types == 'train':
            return trainloader
        else:
            return devloader

    # return dl


def fetch_subset_dataloader(types, args):
    """
    Use only a subset of dataset for KD training, depending on args.subset_percent
    """
    trainset, devset = fetch_dataset(args)

    trainset_size = len(trainset)
    indices = list(range(trainset_size))
    split = int(np.floor(args.subset_percent * trainset_size))
    np.random.seed(args.seed)
    np.random.shuffle(indices)

    train_sampler = SubsetRandomSampler(indices[:split])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              sampler=train_sampler, num_workers=args.num_workers, pin_memory=args.pin_memory)

    devloader = torch.utils.data.DataLoader(devset, batch_size=args.batch_size,
                                            shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)

    if types == 'train':
        dl = trainloader
    else:
        dl = devloader

    return dl


def fetch_valNeeded_dataloader(types, args):
    # 可以根据比例划分出验证集。但是该方法是在整体上随机划分。而不是对每个类按照比例来划分。
    """
    Use only a subset of dataset for KD training, depending on args.subset_percent
    """

    trainset, devset = fetch_dataset(args)


    if types == 'trainANDval':
        # 从训练集中划分验证集
        trainset_size = len(trainset)
        indices = list(range(trainset_size))
        split = int(np.floor(args.subset_percent * trainset_size))
        np.random.seed(args.seed)
        np.random.shuffle(indices)

        train_sampler = SubsetRandomSampler(indices[:-split])
        val_sampler = SubsetRandomSampler(indices[-split:])

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                  sampler=train_sampler, num_workers=args.num_workers, pin_memory=args.pin_memory)
        valloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                sampler=val_sampler, num_workers=args.num_workers, pin_memory=args.pin_memory)
        # dl = trainloader
        return trainloader, valloader
    elif types == 'test':
        devloader = torch.utils.data.DataLoader(devset, batch_size=args.batch_size,
                                                shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
        # dl = devloader
        return devloader

    # return dl

