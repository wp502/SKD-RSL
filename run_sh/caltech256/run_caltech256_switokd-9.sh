#!/bin/bash

# CALTECH256 SwitOKD

# 1
# ResNet34_another - ResNet18_another 1-01_01
python train_online.py --epochs 200 --lr_decay_epochs 60,120,160 --dataset caltech256 --num_class 256 --dataset_test_percent 0.25 --batch_size 64 --learning_rate 0.1 --lr_decay_rate 0.1 --models_list ResNet34_another ResNet18_another --distill switokd --kd_T 1 -t contrast-9-01_01

# 2
# ResNet101_another - ResNet34_another 1-02_01
python train_online.py --epochs 200 --lr_decay_epochs 60,120,160 --dataset caltech256 --num_class 256 --dataset_test_percent 0.25 --batch_size 64 --learning_rate 0.1 --lr_decay_rate 0.1 --models_list ResNet101_another ResNet34_another --distill switokd --kd_T 1 -t contrast-9-02_01

# 3
# VGG13_BN_IMG - VGG8_BN_IMG
python train_online.py --epochs 200 --lr_decay_epochs 60,120,160 --dataset caltech256 --num_class 256 --dataset_test_percent 0.25 --batch_size 64 --learning_rate 0.1 --lr_decay_rate 0.1 --models_list VGG13_BN_IMG VGG8_BN_IMG --distill switokd --kd_T 1 -t contrast-9-03_01

# 4
# ResNet34_another - ShuffleV2_img
python train_online.py --epochs 200 --lr_decay_epochs 60,120,160 --dataset caltech256 --num_class 256 --dataset_test_percent 0.25 --batch_size 64 --learning_rate 0.1 --lr_decay_rate 0.1 --models_list ResNet34_another ShuffleV2_img --distill switokd --kd_T 1 -t contrast-9-04_01

# 5
# VGG11_BN_IMG - MobileNetV2_img
python train_online.py --epochs 200 --lr_decay_epochs 60,120,160 --dataset caltech256 --num_class 256 --dataset_test_percent 0.25 --batch_size 64 --learning_rate 0.1 --lr_decay_rate 0.1 --models_list VGG11_BN_IMG MobileNetV2_img --distill switokd --kd_T 1 -t contrast-9-05_01