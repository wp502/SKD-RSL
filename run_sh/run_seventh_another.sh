#!/bin/bash

basic='train_ours_13.py --model_s ResNet18_cifar_another --model_t ResNet34_cifar_another --path_t ./save/base/cifar100/ResNet34_cifar_another_base/ResNet34_cifar_another_teacher_best.pth.tar --method 2 --in_method MEAN_STD_allAE --fusion_method Conv --feedback_time 0'
#--add_param_groups_method
#--infwd_loss
#--fusion_method
#--in_criterion
#-t

#另类测试 使用method 2
#mean_std_allAE + MSE Tpart2 init0.1 decay_rate0.1 method2 Conv fwdloss1 三block ilrT0.1 fbSet0.3
python $basic --add_param_groups_method 2 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE --blocks_amount 3 --infwd_loss 1 --learning_rate_t 0.1 --fb_set_percent 0.3 --scheduler_method MultiStepLR -t 7-02_01_method2_fbSetPer0.3 &&

#mean_std_allAE + MSE Tpart2 init0.1 decay_rate0.1 method2 Conv fwdloss1 三block ilrT0.1 fbSet0.4
python $basic --add_param_groups_method 2 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE --blocks_amount 3 --infwd_loss 1 --learning_rate_t 0.1 --fb_set_percent 0.4 --scheduler_method MultiStepLR -t 7-02_02_method2_fbSetPer0.4 &&

#mean_std_allAE + MSE Tpart2 init0.1 decay_rate0.1 method2 Conv fwdloss1 三block ilrT0.1 fbSet0.5
python $basic --add_param_groups_method 2 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE --blocks_amount 3 --infwd_loss 1 --learning_rate_t 0.1 --fb_set_percent 0.5 --scheduler_method MultiStepLR -t 7-02_03_method2_fbSetPer0.5