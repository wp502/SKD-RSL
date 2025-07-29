#!/bin/bash

basic='train_ours.py --model_s ResNet18_cifar_another --model_t ResNet34_cifar_another --path_t ./save/base/cifar100/ResNet34_cifar_another_base/ResNet34_cifar_another_teacher_best.pth.tar --lr_decay_rate 0.1 --feedback_time 0 --method 1 --in_method MEAN_STD'
#--add_param_groups_method
#--infwd_loss
#--fusion_method
#--in_criterion
#-t

#mean_std + KL(softmax) Tpart2 0.1 method1 Conv
python $basic --add_param_groups_method 2 --fusion_method Conv --in_criterion KL_softmax -t forth_MEANSTD_KLsoftmax_feedback_Tpart2_lr0.1_method1_Conv &&

#mean_std + MSE(softmax) Tpart12 0.1 method1 Conv
python $basic --add_param_groups_method 12 --fusion_method Conv --in_criterion MSE_softmax -t forth_MEANSTD_MSEsoftmax_feedback_Tpart12_lr0.1_method1_Conv &&

#mean_std + KL(softmax) Tpart12 0.1 method1 Conv
python $basic --add_param_groups_method 12 --fusion_method Conv --in_criterion KL_softmax -t forth_MEANSTD_KLsoftmax_feedback_Tpart12_lr0.1_method1_Conv &&

#mean_std + KL(softmax) Tpart1 0.1 method1 Conv
python $basic --add_param_groups_method 1 --fusion_method Conv --in_criterion KL_softmax -t forth_MEANSTD_KLsoftmax_feedback_Tpart1_lr0.1_method1_Conv