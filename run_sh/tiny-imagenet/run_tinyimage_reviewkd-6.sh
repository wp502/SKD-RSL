#!/bin/bash

# TINY-IMAGENET ReviewKD
#100 [30,60,90] lr0.1 decay0.1 batchsize 128

# 1
# ResNet34_another - ResNet18_another 1-01_01
python train_student.py --epochs 100 --lr_decay_epochs 30,60,90 --dataset tiny_imagenet --num_class 200 --batch_size 128 --learning_rate 0.1 --lr_decay_rate 0.1 --model_t ResNet34_another --model_s ResNet18_another --path_t ./save/base/tiny_imagenet/ResNet34_another_tiny_imagenet_base/ResNet34_another_teacher_90_checkpoint_64.59.pth.tar --distill reviewkd -a 5 --kd_T 4 -t contrast-6-01_01

# 2
# ResNet101_another - ResNet34_another 1-02_01
python train_student.py --epochs 100 --lr_decay_epochs 30,60,90 --dataset tiny_imagenet --num_class 200 --batch_size 128 --learning_rate 0.1 --lr_decay_rate 0.1 --model_t ResNet101_another --model_s ResNet34_another --path_t ./save/base/tiny_imagenet/ResNet101_another_tiny_imagenet_base/ResNet101_another_teacher_best.pth.tar --distill reviewkd -a 5 --kd_T 4 -t contrast-6-02_01

# 3
# VGG13_BN_IMG - VGG8_BN_IMG
python train_student.py --epochs 100 --lr_decay_epochs 30,60,90 --dataset tiny_imagenet --num_class 200 --batch_size 128 --learning_rate 0.1 --lr_decay_rate 0.1 --model_t VGG13_BN_IMG --model_s VGG8_BN_IMG --path_t ./save/base/tiny_imagenet/VGG13_BN_IMG_tiny_imagenet_base/VGG13_BN_IMG_teacher_best.pth.tar --distill reviewkd -a 5 --kd_T 4 -t contrast-6-03_01

# 4
# ResNet34_another - ShuffleV2_img
python train_student.py --epochs 100 --lr_decay_epochs 30,60,90 --dataset tiny_imagenet --num_class 200 --batch_size 128 --learning_rate 0.1 --lr_decay_rate 0.1 --model_t ResNet34_another --model_s ShuffleV2_img --path_t ./save/base/tiny_imagenet/ResNet34_another_tiny_imagenet_base/ResNet34_another_teacher_90_checkpoint_64.59.pth.tar --distill reviewkd -a 5 --kd_T 4 -t contrast-6-04_01

# 5
# VGG11_BN_IMG - MobileNetV2_img
python train_student.py --epochs 100 --lr_decay_epochs 30,60,90 --dataset tiny_imagenet --num_class 200 --batch_size 128 --learning_rate 0.1 --lr_decay_rate 0.1 --model_t VGG11_BN_IMG --model_s MobileNetV2_img --path_t ./save/base/tiny_imagenet/VGG11_BN_IMG_tiny_imagenet_base/VGG11_BN_IMG_teacher_best.pth.tar --distill reviewkd -a 5 --kd_T 4 -t contrast-6-05_01