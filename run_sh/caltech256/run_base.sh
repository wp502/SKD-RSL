#!/bin/bash

# CALTECH256

# FOR (3, 224, 224)

#ResNet18_another
python train_teacher.py --dataset caltech256 --dataset_test_percent 0.25 --num_class 256 --epochs 200 --learning_rate 0.1 --lr_decay_epochs 60,120,160 --lr_decay_rate 0.1 --batch_size 64 --model ResNet18_another -t 1

#ResNet34_another
python train_teacher.py --dataset caltech256 --dataset_test_percent 0.25 --num_class 256 --epochs 250 --learning_rate 0.1 --lr_decay_epochs 80,160,200 --lr_decay_rate 0.1 --batch_size 64 --model ResNet34_another -t 1 --is_checkpoint --save_checkpoint_amount 70

#ResNet101_another
python train_teacher.py --dataset caltech256 --dataset_test_percent 0.25 --num_class 256 --epochs 250 --learning_rate 0.1 --lr_decay_epochs 80,160,200 --lr_decay_rate 0.1 --batch_size 64 --model ResNet101_another -t 1 --is_checkpoint --save_checkpoint_amount 70

#VGG8_BN_IMG
python train_teacher.py --dataset caltech256 --dataset_test_percent 0.25 --num_class 256 --epochs 200 --learning_rate 0.1 --lr_decay_epochs 60,120,160 --lr_decay_rate 0.1 --batch_size 64 --model VGG8_BN_IMG -t 1

#VGG11_BN_IMG
python train_teacher.py --dataset caltech256 --dataset_test_percent 0.25 --num_class 256 --epochs 200 --learning_rate 0.1 --lr_decay_epochs 60,120,160 --lr_decay_rate 0.1 --batch_size 64 --model VGG11_BN_IMG -t 1

#VGG13_BN_IMG
python train_teacher.py --dataset caltech256 --dataset_test_percent 0.25 --num_class 256 --epochs 200 --learning_rate 0.1 --lr_decay_epochs 60,120,160 --lr_decay_rate 0.1 --batch_size 64 --model VGG13_BN_IMG -t 1

#ShuffleV2_img
python train_teacher.py --dataset caltech256 --dataset_test_percent 0.25 --num_class 256 --epochs 200 --learning_rate 0.1 --lr_decay_epochs 60,120,160 --lr_decay_rate 0.1 --batch_size 64 --model ShuffleV2_img -t 1

#MobileNetV2_img
python train_teacher.py --dataset caltech256 --dataset_test_percent 0.25 --num_class 256 --epochs 200 --learning_rate 0.1 --lr_decay_epochs 60,120,160 --lr_decay_rate 0.1 --batch_size 64 --model MobileNetV2_img -t 1

# FOR (3, 32, 32)

#ResNet18_cifar_another
python train_teacher.py --dataset caltech256 --augmentation no --dataset_test_percent 0.25 --num_class 256 --epochs 200 --learning_rate 0.1 --lr_decay_epochs 60,120,160 --lr_decay_rate 0.1 --batch_size 128 --model ResNet18_cifar_another -t 1

#ResNet34_cifar_another
python train_teacher.py --dataset caltech256 --augmentation no --dataset_test_percent 0.25 --num_class 256 --epochs 250 --learning_rate 0.1 --lr_decay_epochs 80,160,200 --lr_decay_rate 0.1 --batch_size 128 --model ResNet34_cifar_another -t 1 --is_checkpoint --save_checkpoint_amount 70

#ResNet101_cifar_another
python train_teacher.py --dataset caltech256 --augmentation no --dataset_test_percent 0.25 --num_class 256 --epochs 250 --learning_rate 0.1 --lr_decay_epochs 80,160,200 --lr_decay_rate 0.1 --batch_size 128 --model ResNet101_cifar_another -t 1 --is_checkpoint --save_checkpoint_amount 70

#VGG8_BN
python train_teacher.py --dataset caltech256 --augmentation no --dataset_test_percent 0.25 --num_class 256 --epochs 200 --learning_rate 0.1 --lr_decay_epochs 60,120,160 --lr_decay_rate 0.1 --batch_size 128 --model VGG8_BN -t 1

#VGG11_BN
python train_teacher.py --dataset caltech256 --augmentation no --dataset_test_percent 0.25 --num_class 256 --epochs 200 --learning_rate 0.1 --lr_decay_epochs 60,120,160 --lr_decay_rate 0.1 --batch_size 128 --model VGG11_BN -t 1

#VGG13_BN
python train_teacher.py --dataset caltech256 --augmentation no --dataset_test_percent 0.25 --num_class 256 --epochs 200 --learning_rate 0.1 --lr_decay_epochs 60,120,160 --lr_decay_rate 0.1 --batch_size 128 --model VGG13_BN -t 1
