#!/bin/bash

#CSKD需要跑他自己的代码，在CSKD文件夹中

# CIFAR100 SELF CSKD
#200 [60,120,160] lr0.1 decay0.1 batchsize128

# ResNet18_cifar_another - ResNet18_cifar_another
python train.py --lr 0.1 --model ResNet18_cifar_another --batch-size 128 --epoch 200 --decay 5e-4 --dataset cifar100 --cls --temp 4 --lamda 1 --sgpu 0 --name contrast-5-01_01_self

#WRN_40_2 - WRN_40_2
python train.py --lr 0.1 --model WRN_40_2 --batch-size 128 --epoch 200 --decay 5e-4 --dataset cifar100 --cls --temp 4 --lamda 1 --sgpu 0 --name contrast-5-02_01_self

#VGG11_BN - VGG11_BN
python train.py --lr 0.1 --model VGG11_BN --batch-size 128 --epoch 200 --decay 5e-4 --dataset cifar100 --cls --temp 4 --lamda 1 --sgpu 0 --name contrast-5-03_01_self