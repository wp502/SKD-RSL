#!/bin/bash

# CALTECH256 FITNET
#200 [60,120,160] lr0.1 decay0.10 batchsize 64 test_percent 0.25

# 1
# ResNet34_another - ResNet18_another 1-01_01
python train_student.py --epochs 200 --lr_decay_epochs 60,120,160 --dataset caltech256 --num_class 256 --batch_size 64 --dataset_test_percent 0.25 --learning_rate 0.1 --lr_decay_rate 0.1 --model_t ResNet34_another --model_s ResNet18_another --path_t ./save/base/caltech256/ResNet34_another_caltech256_base/ResNet34_another_teacher_67.71.pth.tar --distill fitnet -a 1 -b 1 -r 1 --kd_T 3 -t contrast-4-01_01

# 2
# ResNet101_another - ResNet34_another 1-02_01
python train_student.py --epochs 200 --lr_decay_epochs 60,120,160 --dataset caltech256 --num_class 256 --batch_size 64 --dataset_test_percent 0.25 --learning_rate 0.1 --lr_decay_rate 0.1 --model_t ResNet101_another --model_s ResNet34_another --path_t ./save/base/caltech256/ResNet101_another_caltech256_base/ResNet101_another_teacher_best.pth.tar --distill fitnet -a 1 -b 1 -r 1 --kd_T 3 -t contrast-4-02_01

# 3
# VGG13_BN_IMG - VGG8_BN_IMG
python train_student.py --epochs 200 --lr_decay_epochs 60,120,160 --dataset caltech256 --num_class 256 --batch_size 64 --dataset_test_percent 0.25 --learning_rate 0.1 --lr_decay_rate 0.1 --model_t VGG13_BN_IMG --model_s VGG8_BN_IMG --path_t ./save/base/caltech256/VGG13_BN_IMG_caltech256_base/VGG13_BN_IMG_teacher_best.pth.tar --distill fitnet -a 1 -b 1 -r 1 --kd_T 3 -t contrast-4-03_01

# 4
# ResNet34_another - ShuffleV2_img
python train_student.py --epochs 200 --lr_decay_epochs 60,120,160 --dataset caltech256 --num_class 256 --batch_size 64 --dataset_test_percent 0.25 --learning_rate 0.1 --lr_decay_rate 0.1 --model_t ResNet34_another --model_s ShuffleV2_img --path_t ./save/base/caltech256/ResNet34_another_caltech256_base/ResNet34_another_teacher_67.71.pth.tar --distill fitnet -a 1 -b 1 -r 1 --kd_T 3 -t contrast-4-04_01

# 5
# VGG11_BN_IMG - MobileNetV2_img
python train_student.py --epochs 200 --lr_decay_epochs 60,120,160 --dataset caltech256 --num_class 256 --batch_size 64 --dataset_test_percent 0.25 --learning_rate 0.1 --lr_decay_rate 0.1 --model_t VGG11_BN_IMG --model_s MobileNetV2_img --path_t ./save/base/caltech256/VGG11_BN_IMG_caltech256_base/VGG11_BN_IMG_teacher_best.pth.tar --distill fitnet -a 1 -b 1 -r 1 --kd_T 3 -t contrast-4-05_01