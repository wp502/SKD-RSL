#!/bin/bash

basic='train_ours_19_2.py --model_s ResNet18_cifar_another --model_t ResNet34_cifar_another --path_t ./save/base/cifar100/ResNet34_cifar_another_base/ResNet34_cifar_another_teacher_best.pth.tar --method 1 --in_method inAuxCF_logitsNToAE_OnlyOne-2 --fusion_method AUXCF --feedback_time 0'


#base

##inAuxCF_logitsNToAE_OnlyOne-2 + KLsoftmax TpartAll init0.1 decay_rate0.1 method1 AUXCF fwdloss1 三block ilrT5e-3 ConsineSch AEConv
#python $basic --add_param_groups_method all --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion KL_softmax --blocks_amount 3 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method CosineAnnealingLR --fusion_method_AUXCF AEConv -t 17-01_01
#
##inAuxCF_logitsNToAE_OnlyOne-2 + KLsoftmax TpartAll init0.1 decay_rate0.1 method1 AUXCF fwdloss1 四block ilrT5e-3 ConsineSch AEConv3x3Linear
#python $basic --add_param_groups_method all --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion KL_softmax --blocks_amount 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method CosineAnnealingLR --fusion_method_AUXCF AEConv3x3Linear -t 17-01_02
#
##inAuxCF_logitsNToAE_OnlyOne-2 + KLsoftmax TpartAll init0.1 decay_rate0.1 method1 AUXCF fwdloss1 三block ilrT5e-3 MultiStep(60,120,160) AEConv
#python $basic --add_param_groups_method all --lr_decay_epochs 60,120,160 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion KL_softmax --blocks_amount 3 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --fusion_method_AUXCF AEConv -t 17-01_03
#
##inAuxCF_logitsNToAE_OnlyOne-2 + KLsoftmax TpartAll init0.1 decay_rate0.1 method1 AUXCF fwdloss1 三block ilrT1e-3 MultiStep(60,120,160) AEConv
#python $basic --add_param_groups_method all --lr_decay_epochs 60,120,160 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion KL_softmax --blocks_amount 3 --infwd_loss 1 --learning_rate_t 1e-3 --scheduler_method MultiStepLR --fusion_method_AUXCF AEConv -t 17-01_04
#
##inAuxCF_logitsNToAE_OnlyOne-2 + KLsoftmax TpartAll init0.1 decay_rate0.1 method1 AUXCF fwdloss1 三block ilrT5e-4 MultiStep(60,120,160) AEConv
#python $basic --add_param_groups_method all --lr_decay_epochs 60,120,160 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion KL_softmax --blocks_amount 3 --infwd_loss 1 --learning_rate_t 5e-4 --scheduler_method MultiStepLR --fusion_method_AUXCF AEConv -t 17-01_05
#
##inAuxCF_logitsNToAE_OnlyOne-2 + KLsoftmax TpartAll init0.1 decay_rate0.1 method1 AUXCF fwdloss1 三block ilrT1e-4 MultiStep(60,120,160) AEConv
#python $basic --add_param_groups_method all --lr_decay_epochs 60,120,160 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion KL_softmax --blocks_amount 3 --infwd_loss 1 --learning_rate_t 1e-4 --scheduler_method MultiStepLR --fusion_method_AUXCF AEConv -t 17-01_06

#inAuxCF_logitsNToAE_OnlyOne-2 + KLsoftmax TpartAll init0.1 decay_rate0.1 method1 AUXCF fwdloss1 三block ilrT5e-3 ConsineSch AEConv
python $basic --add_param_groups_method all --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion KL_softmax --blocks_amount 3 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method CosineAnnealingLR --fusion_method_AUXCF AEConv -t 17-01_01_True

#inAuxCF_logitsNToAE_OnlyOne-2 + KLsoftmax TpartAll init0.1 decay_rate0.1 method1 AUXCF fwdloss1 三block ilrT5e-3 MultiStep(60,120,160) AEConv
python $basic --add_param_groups_method all --lr_decay_epochs 60,120,160 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion KL_softmax --blocks_amount 3 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --fusion_method_AUXCF AEConv -t 17-01_02_True

#inAuxCF_logitsNToAE_OnlyOne-2 + KLsoftmax TpartAll init0.1 decay_rate0.1 method1 AUXCF fwdloss1 四block ilrT5e-3 MultiStep(60,120,160) AEConv
python $basic --add_param_groups_method all --lr_decay_epochs 60,120,160 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion KL_softmax --blocks_amount 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --fusion_method_AUXCF AEConv -t 17-01_03_True

