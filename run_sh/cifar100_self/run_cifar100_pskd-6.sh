#!/bin/bash

# PSKD 需要跑他自己的代码，在 PSKD 文件夹中

# CIFAR100 SELF PSKD
#200 [60,120,160] lr0.1 decay0.1 batchsize128

# ResNet18_cifar_another - ResNet18_cifar_another
python main.py --lr 0.1 --end_epoch 200 --lr_decay_schedule 60 120 160 --PSKD --experiments_dir ./save/cifar100 --batch_size 128 --classifier_type ResNet18_cifar_another --data_path ../CSKD/data/ --data_type cifar100 --alpha_T 0.8 -t contrast-6-01_01_self

#WRN_40_2 - WRN_40_2
python main.py --lr 0.1 --end_epoch 200 --lr_decay_schedule 60 120 160 --PSKD --experiments_dir ./save/cifar100 --batch_size 128 --classifier_type WRN_40_2 --data_path ../CSKD/data/ --data_type cifar100 --alpha_T 0.8 -t contrast-6-02_01_self

#VGG11_BN - VGG11_BN
python main.py --lr 0.1 --end_epoch 200 --lr_decay_schedule 60 120 160 --PSKD --experiments_dir ./save/cifar100 --batch_size 128 --classifier_type VGG11_BN --data_path ../CSKD/data/ --data_type cifar100 --alpha_T 0.8 -t contrast-6-03_01_self