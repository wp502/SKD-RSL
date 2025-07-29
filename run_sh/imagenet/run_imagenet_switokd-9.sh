#!/bin/bash

# IMAGENET
#90 [30,60,80] lr0.1 decay 0.1 weight-decay 1e-4 batchsize256

# 1
# ResNet34_another - ResNet18_another 1-01_01

python train_online.py --batch_size 256 --num_workers 8 --epochs 90 --learning_rate 0.1 --lr_decay_epochs 30,60,80 --lr_decay_rate 0.1 --weight_decay 1e-4 --models_list ResNet34_another ResNet18_another --distill switokd --kd_T 1 --dataset imagenet --num_class 1000 --dataset_dir /data1/mazc/gjp/xxm/NKD/data/imagenet/ -t contrast-9-01_01
