#!/bin/bash

# TINY-IMAGENET
#100 [30,60,90] lr0.1 decay0.1 batchsize 128

# 1
# ResNet34_another - ResNet18_another 1-01_01
python train_online.py --epochs 100 --lr_decay_epochs 30,60,90 --dataset tiny_imagenet --num_class 200 --batch_size 128 --learning_rate 0.1 --lr_decay_rate 0.1 --models_list ResNet34_another ResNet18_another --distill ffl --kd_T 3 -t contrast-8-01_01

# 2
# ResNet101_another - ResNet34_another 1-02_01
python train_online.py --epochs 100 --lr_decay_epochs 30,60,90 --dataset tiny_imagenet --num_class 200 --batch_size 128 --learning_rate 0.1 --lr_decay_rate 0.1 --models_list ResNet101_another ResNet34_another --distill ffl --kd_T 3 -t contrast-8-02_01

# 3
# VGG13_BN_IMG - VGG8_BN_IMG
python train_online.py --epochs 100 --lr_decay_epochs 30,60,90 --dataset tiny_imagenet --num_class 200 --batch_size 128 --learning_rate 0.1 --lr_decay_rate 0.1 --models_list VGG13_BN_IMG VGG8_BN_IMG --distill ffl --kd_T 3 -t contrast-8-03_01

# 4
# ResNet34_another - ShuffleV2_img
python train_online.py --epochs 100 --lr_decay_epochs 30,60,90 --dataset tiny_imagenet --num_class 200 --batch_size 128 --learning_rate 0.1 --lr_decay_rate 0.1 --models_list ResNet34_another ShuffleV2_img --distill ffl --kd_T 3 -t contrast-8-04_01

# 5
# VGG11_BN_IMG - MobileNetV2_img
python train_online.py --epochs 100 --lr_decay_epochs 30,60,90 --dataset tiny_imagenet --num_class 200 --batch_size 128 --learning_rate 0.1 --lr_decay_rate 0.1 --models_list VGG11_BN_IMG MobileNetV2_img --distill ffl --kd_T 3 -t contrast-8-05_01