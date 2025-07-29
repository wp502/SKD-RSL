#!/bin/bash

# CALTECH256 SOKD

# 1
# ResNet34_another - ResNet18_another 1-01_01
python train_online.py --epochs 200 --lr_decay_epochs 60,120,160 --dataset caltech256 --num_class 256 --dataset_test_percent 0.25 --batch_size 64 --learning_rate 0.1 --lr_decay_rate 0.1 --model_t ResNet34_another --path_t ./save/base/caltech256/ResNet34_another_caltech256_base/ResNet34_another_teacher_67.71.pth.tar  --models_list ResNet18_another --distill sokd --kd_T 3 -t contrast-7-01_01

# 2
# ResNet101_another - ResNet34_another 1-02_01
python train_online.py --epochs 200 --lr_decay_epochs 60,120,160 --dataset caltech256 --num_class 256 --dataset_test_percent 0.25 --batch_size 64 --learning_rate 0.1 --lr_decay_rate 0.1 --model_t ResNet101_another --path_t ./save/base/caltech256/ResNet101_another_caltech256_base/ResNet101_another_teacher_best.pth.tar  --models_list ResNet34_another --distill sokd --kd_T 3 -t contrast-7-02_01

# 3
# VGG13_BN_IMG - VGG8_BN_IMG
python train_online.py --epochs 200 --lr_decay_epochs 60,120,160 --dataset caltech256 --num_class 256 --dataset_test_percent 0.25 --batch_size 64 --learning_rate 0.1 --lr_decay_rate 0.1 --model_t VGG13_BN_IMG --path_t ./save/base/caltech256/VGG13_BN_IMG_caltech256_base/VGG13_BN_IMG_teacher_best.pth.tar  --models_list VGG8_BN_IMG --distill sokd --kd_T 3 -t contrast-7-03_01

# 4
# ResNet34_another - ShuffleV2_img
python train_online.py --epochs 200 --lr_decay_epochs 60,120,160 --dataset caltech256 --num_class 256 --dataset_test_percent 0.25 --batch_size 64 --learning_rate 0.1 --lr_decay_rate 0.1 --model_t ResNet34_another --path_t ./save/base/caltech256/ResNet34_another_caltech256_base/ResNet34_another_teacher_67.71.pth.tar  --models_list ShuffleV2_img --distill sokd --kd_T 3 -t contrast-7-04_01

# 5
# VGG11_BN_IMG - MobileNetV2_img
python train_online.py --epochs 200 --lr_decay_epochs 60,120,160 --dataset caltech256 --num_class 256 --dataset_test_percent 0.25 --batch_size 64 --learning_rate 0.1 --lr_decay_rate 0.1 --model_t VGG11_BN_IMG --path_t ./save/base/caltech256/VGG11_BN_IMG_caltech256_base/VGG11_BN_IMG_teacher_best.pth.tar  --models_list MobileNetV2_img --distill sokd --kd_T 3 -t contrast-7-05_01