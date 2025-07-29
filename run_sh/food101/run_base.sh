#!/bin/bash

# FOOD101

#ResNet18_another
python train_teacher.py --dataset food101 --num_class 101 --epochs 100 --learning_rate 0.1 --lr_decay_epochs 30,60,80 --lr_decay_rate 0.1 --batch_size 64 --model ResNet18_another -t 1

#ResNet34_another
python train_teacher.py --dataset food101 --num_class 101 --epochs 100 --learning_rate 0.1 --lr_decay_epochs 30,60,80 --lr_decay_rate 0.1 --batch_size 64 --model ResNet34_another -t 1 --is_checkpoint --save_checkpoint_amount 50

#ResNet101_another
python train_teacher.py --dataset food101 --num_class 101 --epochs 100 --learning_rate 0.1 --lr_decay_epochs 30,60,80 --lr_decay_rate 0.1 --batch_size 64 --model ResNet101_another -t 1 --is_checkpoint --save_checkpoint_amount 50

#VGG8_BN_IMG
python train_teacher.py --dataset food101 --num_class 101 --epochs 100 --learning_rate 0.1 --lr_decay_epochs 30,60,80 --lr_decay_rate 0.1 --batch_size 64 --model VGG8_BN_IMG -t 1

#VGG11_BN_IMG
python train_teacher.py --dataset food101 --num_class 101 --epochs 100 --learning_rate 0.1 --lr_decay_epochs 30,60,80 --lr_decay_rate 0.1 --batch_size 64 --model VGG11_BN_IMG -t 1

#VGG13_BN_IMG
python train_teacher.py --dataset food101 --num_class 101 --epochs 100 --learning_rate 0.1 --lr_decay_epochs 30,60,80 --lr_decay_rate 0.1 --batch_size 64 --model VGG13_BN_IMG -t 1

#ShuffleV2_img
python train_teacher.py --dataset food101 --num_class 101 --epochs 100 --learning_rate 0.1 --lr_decay_epochs 30,60,80 --lr_decay_rate 0.1 --batch_size 64 --model ShuffleV2_img -t 1

#MobileNetV2_img
python train_teacher.py --dataset food101 --num_class 101 --epochs 100 --learning_rate 0.1 --lr_decay_epochs 30,60,80 --lr_decay_rate 0.1 --batch_size 64 --model MobileNetV2_img -t 1

