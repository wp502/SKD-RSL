#!/bin/bash

# AT + MSE + TtoS
python train_ours.py --model_s ResNet18_cifar_another --model_t ResNet34_cifar_another --path_t ./save/base/cifar100/ResNet34_cifar_another_base/ResNet34_cifar_another_teacher_best.pth.tar --feedback_time 500 --in_method Attention --in_criterion MSE -t first_AT_MSE_TtoS &&


# AT + MSE + StoT
python train_ours.py --model_s ResNet18_cifar_another --model_t ResNet34_cifar_another --path_t ./save/base/cifar100/ResNet34_cifar_another_base/ResNet34_cifar_another_teacher_best.pth.tar --feedback_time 500 --in_method Attention --in_criterion MSE --fwd_in_TtoS -t first_AT_MSE_StoT