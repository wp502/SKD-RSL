#!/bin/bash

# CIFAR100 ReviewKD
#200 [60,120,160] lr0.1 decay0.1 batchsize128

# 1
# ResNet34_cifar_another - ResNet18_cifar_another 1-01_01
python train_student.py --epochs 200 --lr_decay_epochs 60,120,160 --dataset cifar100 --num_class 100 --batch_size 128 --learning_rate 0.1 --lr_decay_rate 0.1 --model_t ResNet34_cifar_another --model_s ResNet18_cifar_another --path_t ./save/base/cifar100/ResNet34_cifar_another_base/ResNet34_cifar_another_teacher_best.pth.tar --distill reviewkd -a 1 --kd_T 4 -t contrast-6-01_01
python train_student.py --epochs 200 --lr_decay_epochs 60,120,160 --dataset cifar100 --num_class 100 --batch_size 128 --learning_rate 0.1 --lr_decay_rate 0.1 --model_t ResNet34_cifar_another --model_s ResNet18_cifar_another --path_t ./save/base/cifar100/ResNet34_cifar_another_base/ResNet34_cifar_another_teacher_best.pth.tar --distill reviewkd -a 5 --kd_T 4 -t contrast-6-01_02

# 2
# ResNet101_cifar_another - ResNet34_cifar_another 1-02_01
python train_student.py --epochs 200 --lr_decay_epochs 60,120,160 --dataset cifar100 --num_class 100 --batch_size 128 --learning_rate 0.1 --lr_decay_rate 0.1 --model_t ResNet101_cifar_another --model_s ResNet34_cifar_another --path_t ./save/base/cifar100/ResNet101_cifar_another_base/ResNet101_cifar_another_teacher_best.pth.tar --distill reviewkd -a 1 --kd_T 4 -t contrast-6-02_01
python train_student.py --epochs 200 --lr_decay_epochs 60,120,160 --dataset cifar100 --num_class 100 --batch_size 128 --learning_rate 0.1 --lr_decay_rate 0.1 --model_t ResNet101_cifar_another --model_s ResNet34_cifar_another --path_t ./save/base/cifar100/ResNet101_cifar_another_base/ResNet101_cifar_another_teacher_best.pth.tar --distill reviewkd -a 5 --kd_T 4 -t contrast-6-02_02

# 3
# VGG13_BN - VGG8_BN
python train_student.py --epochs 200 --lr_decay_epochs 60,120,160 --dataset cifar100 --num_class 100 --batch_size 128 --learning_rate 0.1 --lr_decay_rate 0.1 --model_t VGG13_BN --model_s VGG8_BN --path_t ./save/base/cifar100/VGG13_BN_base/VGG13_BN_teacher_best.pth.tar --distill reviewkd -a 1 --kd_T 4 -t contrast-6-03_01
python train_student.py --epochs 200 --lr_decay_epochs 60,120,160 --dataset cifar100 --num_class 100 --batch_size 128 --learning_rate 0.1 --lr_decay_rate 0.1 --model_t VGG13_BN --model_s VGG8_BN --path_t ./save/base/cifar100/VGG13_BN_base/VGG13_BN_teacher_best.pth.tar --distill reviewkd -a 5 --kd_T 4 -t contrast-6-03_02

# 4
# ResNet34_cifar_another - ShuffleV2
python train_student.py --epochs 200 --lr_decay_epochs 60,120,160 --dataset cifar100 --num_class 100 --batch_size 128 --learning_rate 0.1 --lr_decay_rate 0.1 --model_t ResNet34_cifar_another --model_s ShuffleV2 --path_t ./save/base/cifar100/ResNet34_cifar_another_base/ResNet34_cifar_another_teacher_best.pth.tar --distill reviewkd -a 1 --kd_T 4 -t contrast-6-04_01
python train_student.py --epochs 200 --lr_decay_epochs 60,120,160 --dataset cifar100 --num_class 100 --batch_size 128 --learning_rate 0.1 --lr_decay_rate 0.1 --model_t ResNet34_cifar_another --model_s ShuffleV2 --path_t ./save/base/cifar100/ResNet34_cifar_another_base/ResNet34_cifar_another_teacher_best.pth.tar --distill reviewkd -a 5 --kd_T 4 -t contrast-6-04_02

# 5
# VGG11_BN - MobileNetV2
python train_student.py --epochs 200 --lr_decay_epochs 60,120,160 --dataset cifar100 --num_class 100 --batch_size 128 --learning_rate 0.1 --lr_decay_rate 0.1 --model_t VGG11_BN --model_s MobileNetV2 --path_t ./save/base/cifar100/VGG11_BN_base/VGG11_BN_teacher_best.pth.tar --distill reviewkd -a 1 --kd_T 4 -t contrast-6-05_01
python train_student.py --epochs 200 --lr_decay_epochs 60,120,160 --dataset cifar100 --num_class 100 --batch_size 128 --learning_rate 0.1 --lr_decay_rate 0.1 --model_t VGG11_BN --model_s MobileNetV2 --path_t ./save/base/cifar100/VGG11_BN_base/VGG11_BN_teacher_best.pth.tar --distill reviewkd -a 5 --kd_T 4 -t contrast-6-05_02