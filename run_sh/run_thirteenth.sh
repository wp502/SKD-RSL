#!/bin/bash

basic='train_ours_17.py --model_s ResNet18_cifar_another --model_t ResNet34_cifar_another --path_t ./save/base/cifar100/ResNet34_cifar_another_base/ResNet34_cifar_another_teacher_best.pth.tar --method 1 --in_method inAuxCF_logitsNToAE --fusion_method AUXCF --feedback_time 0'


#base

#inAuxCF_logitsNToAE + MSEsoftmaxT TpartAll init0.1 decay_rate0.1 method1 AUXCF fwdloss1 三block ilrT5e-3 ConsineSch AEConv
python $basic --add_param_groups_method all --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE_softmax_T --blocks_amount 3 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method CosineAnnealingLR --fusion_method_AUXCF AEConv -t 13-01_01

#inAuxCF_logitsNToAE + KLsoftmax TpartAll init0.1 decay_rate0.1 method1 AUXCF fwdloss1 四block ilrT5e-3 ConsineSch AEConv
python $basic --add_param_groups_method all --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion KL_softmax --blocks_amount 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method CosineAnnealingLR --fusion_method_AUXCF AEConv -t 13-01_02

#inAuxCF_logitsNToAE + KLsoftmax TpartAll init0.1 decay_rate0.1 method1 AUXCF fwdloss1 三block ilrT1e-4 ConsineSch AEConv
python $basic --add_param_groups_method all --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion KL_softmax --blocks_amount 3 --infwd_loss 1 --learning_rate_t 1e-4 --scheduler_method CosineAnnealingLR --fusion_method_AUXCF AEConv -t 13-01_03

#inAuxCF_logitsNToAE + KLsoftmax TpartAll init0.1 decay_rate0.1 method1 AUXCF fwdloss1 三block ilrT5e-3 ConsineSch AEConv
python $basic --add_param_groups_method all --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion KL_softmax --blocks_amount 3 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method CosineAnnealingLR --fusion_method_AUXCF AEConv -t 13-01_04

#inAuxCF_logitsNToAE + KLsoftmax TpartAll init0.1 decay_rate0.1 method1 AUXCF fwdloss1 四block ilrT5e-3 ConsineSch AELinear
python $basic --add_param_groups_method all --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion KL_softmax --blocks_amount 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method CosineAnnealingLR --fusion_method_AUXCF AELinear -t 13-01_05

#inAuxCF_logitsNToAE + KLsoftmax TpartAll init0.1 decay_rate0.1 method1 AUXCF fwdloss1 四block ilrT5e-3 ConsineSch AEConv fb不用开关
python $basic --add_param_groups_method all --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion KL_softmax --blocks_amount 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method CosineAnnealingLR --fusion_method_AUXCF AEConv --fbUseGradSim -t 13-01_06

