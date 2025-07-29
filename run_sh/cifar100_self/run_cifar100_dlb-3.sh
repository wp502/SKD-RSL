#!/bin/bash

#DLB需要跑他自己的代码，在DLB文件夹中

# CIFAR100 SELF DLB
#200 [60,120,160] lr0.1 decay0.1 batchsize128

# ResNet18_cifar_another - ResNet18_cifar_another
python dlb.py --model_names ResNet18_cifar_another --num_workers 4 --classes_num 100 --dataset cifar100 --T 3 --alpha 1 --batch_size 128 --epoch 200 --lr 0.1 --milestones 60 120 160 --gamma 0.1 --gpu-id 0 --print_freq 1 --exp_postfix contrast-3-01_01_self

#WRN_40_2 - WRN_40_2
python dlb.py --model_names WRN_40_2 --num_workers 4 --classes_num 100 --dataset cifar100 --T 3 --alpha 1 --batch_size 128 --epoch 200 --lr 0.1 --milestones 60 120 160 --gamma 0.1 --gpu-id 0 --print_freq 1 --exp_postfix contrast-3-02_01_self

#VGG11_BN - VGG11_BN
python dlb.py --model_names VGG11_BN --num_workers 4 --classes_num 100 --dataset cifar100 --T 3 --alpha 1 --batch_size 128 --epoch 200 --lr 0.1 --milestones 60 120 160 --gamma 0.1 --gpu-id 1 --print_freq 1 --exp_postfix contrast-3-03_01_self