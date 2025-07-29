#!/bin/bash

# IMAGENET
#90 [30,60,80] lr0.1 decay 0.1 weight-decay 1e-4 batchsize256

# 1
# ResNet34_another - ResNet18_another 1-01_01

python train_student.py --batch_size 256 --num_workers 8 --epochs 90 --learning_rate 0.1 --lr_decay_epochs 30,60,80 --lr_decay_rate 0.1 --weight_decay 1e-4 --model_s ResNet18_another --model_t ResNet34_another --path_t ./save/base/imagenet/ResNet34_another_imagenet_base/ResNet34_another_teacher_best.pth.tar --distill reviewkd -a 1 --kd_T 4 --dataset imagenet --num_class 1000 --dataset_dir /data1/mazc/gjp/xxm/NKD/data/imagenet/ --kd_warm_up_rekd 0 -t contrast-6-01_01

#--is_resume --checkpoint_s ./save/distill/imagenet/S-ResNet18_another_T-ResNet34_another_imagenet_reviewkd_r-1_a-1.0_b-1_contrast-6-01_01/ResNet18_another_last.pth.tar
