#!/bin/bash

# CIFAR100 SOKD
#200 [60,120,160] lr0.1 decay0.1 batchsize128

# 1
# ResNet34_cifar_another - ResNet18_cifar_another 1-01_01
python train_online.py --epochs 200 --lr_decay_epochs 60,120,160 --dataset cifar100 --num_class 100 --batch_size 128 --learning_rate 0.1 --lr_decay_rate 0.1 --model_t ResNet34_cifar_another --path_t ./save/base/cifar100/ResNet34_cifar_another_base/ResNet34_cifar_another_teacher_best.pth.tar --models_list ResNet18_cifar_another --distill sokd -a 1 -b 1 -r 1 --kd_T 3 -t contrast-7-01_01

# 2
# ResNet101_cifar_another - ResNet34_cifar_another 1-02_01
python train_online.py --epochs 200 --lr_decay_epochs 60,120,160 --dataset cifar100 --num_class 100 --batch_size 128 --learning_rate 0.1 --lr_decay_rate 0.1 --model_t ResNet101_cifar_another --path_t ./save/base/cifar100/ResNet101_cifar_another_base/ResNet101_cifar_another_teacher_best.pth.tar --models_list ResNet34_cifar_another --distill sokd -a 1 -b 1 -r 1 --kd_T 3 -t contrast-7-02_01

# 3
# VGG13_BN - VGG8_BN
python train_online.py --epochs 200 --lr_decay_epochs 60,120,160 --dataset cifar100 --num_class 100 --batch_size 128 --learning_rate 0.1 --lr_decay_rate 0.1 --model_t VGG13_BN --path_t ./save/base/cifar100/VGG13_BN_base/VGG13_BN_teacher_best.pth.tar --models_list VGG8_BN --distill sokd -a 1 -b 1 -r 1 --kd_T 3 -t contrast-7-03_01

# 4
# ResNet34_cifar_another - ShuffleV2
python train_online.py --epochs 200 --lr_decay_epochs 60,120,160 --dataset cifar100 --num_class 100 --batch_size 128 --learning_rate 0.1 --lr_decay_rate 0.1 --model_t ResNet34_cifar_another --path_t ./save/base/cifar100/ResNet34_cifar_another_base/ResNet34_cifar_another_teacher_best.pth.tar --models_list ShuffleV2 --distill sokd -a 1 -b 1 -r 1 --kd_T 3 -t contrast-7-04_01

# 5
# VGG11_BN - MobileNetV2
python train_online.py --epochs 200 --lr_decay_epochs 60,120,160 --dataset cifar100 --num_class 100 --batch_size 128 --learning_rate 0.1 --lr_decay_rate 0.1 --model_t VGG11_BN --path_t ./save/base/cifar100/VGG11_BN_base/VGG11_BN_teacher_best.pth.tar --models_list MobileNetV2 --distill sokd -a 1 -b 1 -r 1 --kd_T 3 -t contrast-7-05_01