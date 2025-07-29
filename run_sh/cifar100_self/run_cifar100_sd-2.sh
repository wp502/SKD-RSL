#!/bin/bash

#SD需要跑他自己的代码，在SD文件夹中

# CIFAR100 SELF SD
#200 [60,120,160] lr0.1 decay0.1 batchsize128

# ResNet18_cifar_another - ResNet18_cifar_another
python train.py --model ResNet18 --dataset cifar100 --epoch 200 --dataset_path ../CSKD/data/ --autoaugment False --temperature 3 --init_lr 0.1 --batchsize 128 --trial contrast-2-01_01_self

#WRN_40_2 - WRN_40_2
python train.py --model WRN_40_2 --dataset cifar100 --epoch 200 --dataset_path ../CSKD/data/ --autoaugment False --temperature 3 --init_lr 0.1 --batchsize 128 --trial contrast-2-02_01_self

#VGG11_BN - VGG11_BN
python train.py --model VGG11_BN --dataset cifar100 --epoch 200 --dataset_path ../CSKD/data/ --autoaugment False --temperature 3 --init_lr 0.1 --batchsize 128 --trial contrast-2-03_01_self