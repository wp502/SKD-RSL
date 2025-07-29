#!/bin/bash

# CIFAR100 FITNET
#200 [60,120,160] lr0.1 decay0.1 batchsize128

# 1
# ResNet34_cifar_another - ResNet18_cifar_another 1-01_01
python train_student.py --epochs 200 --lr_decay_epochs 60,120,160 --dataset cifar100 --num_class 100 --batch_size 128 --learning_rate 0.1 --lr_decay_rate 0.1 --model_t ResNet34_cifar_another --model_s ResNet18_cifar_another --path_t ./save/base/cifar100/ResNet34_cifar_another_base/ResNet34_cifar_another_teacher_best.pth.tar --distill fitnet -a 1 -b 1 -r 1 --kd_T 3 -t contrast-4-01_01

# 2
# ResNet101_cifar_another - ResNet34_cifar_another 1-02_01
python train_student.py --epochs 200 --lr_decay_epochs 60,120,160 --dataset cifar100 --num_class 100 --batch_size 128 --learning_rate 0.1 --lr_decay_rate 0.1 --model_t ResNet101_cifar_another --model_s ResNet34_cifar_another --path_t ./save/base/cifar100/ResNet101_cifar_another_base/ResNet101_cifar_another_teacher_best.pth.tar --distill fitnet -a 1 -b 1 -r 1 --kd_T 3 -t contrast-4-02_01

# 3
# VGG13_BN - VGG8_BN
python train_student.py --epochs 200 --lr_decay_epochs 60,120,160 --dataset cifar100 --num_class 100 --batch_size 128 --learning_rate 0.1 --lr_decay_rate 0.1 --model_t VGG13_BN --model_s VGG8_BN --path_t ./save/base/cifar100/VGG13_BN_base/VGG13_BN_teacher_best.pth.tar --distill fitnet -a 1 -b 1 -r 1 --kd_T 3 -t contrast-4-03_01

# 4
# ResNet34_cifar_another - ShuffleV2
python train_student.py --epochs 200 --lr_decay_epochs 60,120,160 --dataset cifar100 --num_class 100 --batch_size 128 --learning_rate 0.1 --lr_decay_rate 0.1 --model_t ResNet34_cifar_another --model_s ShuffleV2 --path_t ./save/base/cifar100/ResNet34_cifar_another_base/ResNet34_cifar_another_teacher_best.pth.tar --distill fitnet -a 1 -b 1 -r 1 --kd_T 3 -t contrast-4-04_01

# 5
# VGG11_BN - MobileNetV2
python train_student.py --epochs 200 --lr_decay_epochs 60,120,160 --dataset cifar100 --num_class 100 --batch_size 128 --learning_rate 0.1 --lr_decay_rate 0.1 --model_t VGG11_BN --model_s MobileNetV2 --path_t ./save/base/cifar100/VGG11_BN_base/VGG11_BN_teacher_best.pth.tar --distill fitnet -a 1 -b 1 -r 1 --kd_T 3 -t contrast-4-05_01