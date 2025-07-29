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
from .cub200_2 import CUB2011Metric
from .cars196_2 import Cars196Metric
from .cub200 import CUB200
from .cars196 import Cars
from .cars196_extract_folder import CarstoFolder
from .cub200_extract_folder import CUB200toFolder

__all__ = ['fetch_dataloader']

def fetch_cifar100(args):
    if args.augmentation == "yes":
        train_transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                 (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))])
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.240, 0.243, 0.261))

    # data augmentation can be turned off
    else:
        train_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                 (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))])

    # transformer for dev set
    dev_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                             (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))])

    if args.dataset_dir is not None:
        data_dir = args.dataset_dir
    else:
        data_dir = './data/data-cifar100'

    trainset = torchvision.datasets.CIFAR100(root=data_dir, train=True,
                                             download=True, transform=train_transformer)
    devset = torchvision.datasets.CIFAR100(root=data_dir, train=False,
                                           download=True, transform=dev_transformer)

    return trainset, devset


def fetch_cifar10(args):
    if args.augmentation == "yes":
        train_transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                 (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))])
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.240, 0.243, 0.261))

    # data augmentation can be turned off
    else:
        train_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                 (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))])

    # transformer for dev set
    dev_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                             (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))])

    if args.dataset_dir is not None:
        data_dir = args.dataset_dir
    else:
        data_dir = './data/data-cifar10'
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                            download=True, transform=train_transformer)
    devset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                          download=True, transform=dev_transformer)

    return trainset, devset


def fetch_tiny_imagenet(args):
    if args.dataset_dir is not None:
        data_dir = args.dataset_dir
    else:
        data_dir = './data/tiny-imagenet-200/tiny-imagenet-200'

    # data_dir = './data/tiny-imagenet-200/tiny-imagenet-200/'
    # data_dir = './data/tiny-imagenet-200/'
    train_dir = data_dir + '/train/'
    # test_dir = data_dir + 'val/images/'
    test_dir = data_dir + '/val/'
    if args.augmentation == "yes":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            # transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    trainset = torchvision.datasets.ImageFolder(train_dir, train_transform)
    devset = torchvision.datasets.ImageFolder(test_dir, val_transform)
    
    return trainset, devset

def fetch_imagenet(args):
    if args.dataset_dir is not None:
        data_dir = args.dataset_dir
    else:
        data_dir = './data/imagenet'
    # data_dir = './data/imagenet/'
    train_dir = data_dir + '/ILSVRC2012_img_train/'
    test_dir = data_dir + '/ILSVRC2012_img_val/'
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    trainset = torchvision.datasets.ImageFolder(train_dir, train_transform)
    devset = torchvision.datasets.ImageFolder(test_dir, val_transform)

    return trainset, devset


def fetch_cub200(args):
    if args.dataset_dir is not None:
        data_dir = args.dataset_dir
    else:
        data_dir = './data/CUB_200_2011'
    # data_dir = './data/CUB_200_2011'
    train_dir = data_dir + '/train/'
    test_dir = data_dir + '/test/'
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    CUB200toFolder(data_dir)

    trainset = torchvision.datasets.ImageFolder(train_dir, train_transform)
    assert (len(trainset) == 5994)
    devset = torchvision.datasets.ImageFolder(test_dir, val_transform)
    assert (len(devset) == 5794)

    # trainset = CUB200(data_dir, train=True, transform=train_transform, download=True)
    # devset = CUB200(data_dir, train=False, transform=val_transform, download=True)

    return trainset, devset

def fetch_cars(args):
    if args.dataset_dir is not None:
        data_dir = args.dataset_dir
    else:
        data_dir = './data/Cars196'
    # data_dir = './data/Cars196'
    train_dir = data_dir + '/train/'
    test_dir = data_dir + '/test/'
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    CarstoFolder(data_dir)

    trainset = torchvision.datasets.ImageFolder(train_dir, train_transform)
    assert (len(trainset) == 8144)
    devset = torchvision.datasets.ImageFolder(test_dir, val_transform)
    assert (len(devset) == 8041)
    #
    # trainset = Cars(data_dir, train=True, transform=train_transform, download=True)
    # devset = Cars(data_dir, train=False, transform=val_transform, download=True)

    return trainset, devset


def fetch_dataloader(types, args):
    # using random crops and horizontal flip for train sets

    if args.dataset == 'cifar10':
        trainset, devset = fetch_cifar10(args)
    elif args.dataset == 'cifar100':
        trainset, devset = fetch_cifar100(args)
    elif args.dataset == 'tiny_imagenet':
        trainset, devset = fetch_tiny_imagenet(args)
    elif args.dataset == 'imagenet':
        trainset, devset = fetch_imagenet(args)
    elif args.dataset == 'cub200':
        trainset, devset = fetch_cub200(args)
    elif args.dataset == 'cars196':
        trainset, devset = fetch_cars(args)

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

    # using random crops and horizontal flip for train set
    if args.augmentation == "yes":
        train_transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    # data augmentation can be turned off
    else:
        train_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    # transformer for dev set
    dev_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./data-cifar10', train=True,
                                                download=True, transform=train_transformer)
        devset = torchvision.datasets.CIFAR10(root='./data-cifar10', train=False,
                                              download=True, transform=dev_transformer)
    elif args.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='./data-cifar100', train=True,
                                                 download=True, transform=train_transformer)
        devset = torchvision.datasets.CIFAR100(root='./data-cifar100', train=False,
                                               download=True, transform=dev_transformer)
    elif args.dataset == 'tiny_imagenet':
        data_dir = './data/tiny-imagenet-200/tiny-imagenet-200/'
        # data_dir = './data/tiny-imagenet-200/'
        train_dir = data_dir + 'train/'
        # test_dir = data_dir + 'val/images/'
        test_dir = data_dir + 'val/'
        if args.augmentation == "yes":
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            train_transform = transforms.Compose([
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ])
            val_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ])

        trainset = torchvision.datasets.ImageFolder(train_dir, train_transform)
        devset = torchvision.datasets.ImageFolder(test_dir, val_transform)

    trainset_size = len(trainset)
    indices = list(range(trainset_size))
    split = int(np.floor(args.subset_percent * trainset_size))
    np.random.seed(230)
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
