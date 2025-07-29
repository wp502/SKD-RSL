#!/bin/bash

basic='train_ours_12.py --model_s ShuffleV2 --model_t ResNet18_cifar_another --path_t ./save/base/cifar100/ResNet18_cifar_another_base/ResNet18_cifar_another_teacher_best.pth.tar --method 1 --in_method MEAN_STD --fusion_method Conv'
#--add_param_groups_method
#--infwd_loss
#--fusion_method
#--in_criterion
#-t

#mean_std + MSE Tpart2 init0.05 decay_rate0.1 method1 Conv fwdloss5 三block
python $basic  --feedback_time 0 --add_param_groups_method 2 --learning_rate 0.05 --lr_decay_rate 0.1 --in_criterion MSE --blocks_amount 3 --infwd_loss 5 -t other-01_01_fifth_MEANSTD_MSE_fwdloss5_threeBlock



