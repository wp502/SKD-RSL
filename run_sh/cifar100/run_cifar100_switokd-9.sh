#!/bin/bash

# CIFAR100 Swit-OKD
#200 [60,120,160] lr0.1 decay0.1 batchsize128

# 1
# ResNet34_cifar_another - ResNet18_cifar_another 1-01_01
python train_online.py --epochs 200 --lr_decay_epochs 60,120,160 --dataset cifar100 --num_class 100 --batch_size 128 --learning_rate 0.1 --lr_decay_rate 0.1 --models_list ResNet34_cifar_another ResNet18_cifar_another --distill switokd -t contrast-9-01_01

# 2
# ResNet101_cifar_another - ResNet34_cifar_another 1-02_01
python train_online.py --epochs 200 --lr_decay_epochs 60,120,160 --dataset cifar100 --num_class 100 --batch_size 128 --learning_rate 0.1 --lr_decay_rate 0.1 --models_list ResNet101_cifar_another ResNet34_cifar_another --distill switokd -t contrast-9-02_01

# 3
# VGG13_BN - VGG8_BN
python train_online.py --epochs 200 --lr_decay_epochs 60,120,160 --dataset cifar100 --num_class 100 --batch_size 128 --learning_rate 0.1 --lr_decay_rate 0.1 --models_list VGG13_BN VGG8_BN --distill switokd -t contrast-9-03_01

# 4
# ResNet34_cifar_another - ShuffleV2
python train_online.py --epochs 200 --lr_decay_epochs 60,120,160 --dataset cifar100 --num_class 100 --batch_size 128 --learning_rate 0.1 --lr_decay_rate 0.1 --models_list ResNet34_cifar_another ShuffleV2 --distill switokd -t contrast-9-04_01

# 5
# VGG11_BN - MobileNetV2
python train_online.py --epochs 200 --lr_decay_epochs 60,120,160 --dataset cifar100 --num_class 100 --batch_size 128 --learning_rate 0.1 --lr_decay_rate 0.1 --models_list VGG11_BN MobileNetV2 --distill switokd -t contrast-9-05_01