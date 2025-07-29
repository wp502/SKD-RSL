#!/bin/bash

basic='train_ours_12.py --model_s ResNet18_cifar_another --model_t ResNet34_cifar_another --path_t ./save/base/cifar100/ResNet34_cifar_another_base/ResNet34_cifar_another_teacher_best.pth.tar --method 1 --in_method MEAN_STD_KDCL --fusion_method Conv'
#--add_param_groups_method
#--infwd_loss
#--fusion_method
#--in_criterion
#-t

#base
#mean_std_kdcl + KL(softmax) Tpart12 init0.1 decay_rate0.1 method1 Conv 三block
#python $basic --feedback_time 0 --add_param_groups_method 12 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion KL_softmax --blocks_amount 3 --infwd_loss 1 -t 6-01_01_sixth_MEANSTDKDCL_KLsoftmax_feedback_Tpart12_ilr0.1_dlr0.1_method1_Conv_threeBlock &&


#测试 loss的计算方式
#mean_std_kdcl + MSE Tpart12 init0.1 decay_rate0.1 method1 Conv 三block
#python $basic --feedback_time 0 --add_param_groups_method 12 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE --blocks_amount 3 --infwd_loss 1 -t 6-02_01_sixth_MEANSTDKDCL_MSE_feedback_Tpart12_ilr0.1_dlr0.1_method1_Conv_threeBlock

#测试 Tpart 部分
#mean_std_kdcl + MSE Tpart2 init0.1 decay_rate0.1 method1 Conv 三block
python $basic --feedback_time 0 --add_param_groups_method 2 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE --blocks_amount 3 --infwd_loss 1 -t 6-03_01_sixth_MEANSTDKDCL_MSE_feedback_Tpart2_threeBlock







