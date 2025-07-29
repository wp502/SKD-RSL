#!/bin/bash

# CIFAR-100

# 1
# ResNet34_cifar_another - ResNet18_cifar_another 1-01_01
#basicBig='train_ours_111.py --model_s ResNet18_cifar_another --model_t ResNet34_cifar_another --path_t ./save/base/cifar100/ResNet34_cifar_another_base/ResNet34_cifar_another_teacher_best.pth.tar --method 1 --in_method inAuxCF_logitsNToAE_OnlyOne-5 --fusion_method AUXCF --feedback_time 0 --learning_rate 0.1 --lr_decay_rate 0.1 --add_param_groups_method all --aux_method_t Bottle_big --aux_method_s Bottle_big --self_method_t bi_directional --self_method_s bi_directional'
#basicBig3='train_ours_111.py --model_s ResNet18_cifar_another --model_t ResNet34_cifar_another --path_t ./save/base/cifar100/ResNet34_cifar_another_base/ResNet34_cifar_another_teacher_best.pth.tar --method 3 --in_method inAuxCF_logitsNToAE_OnlyOne-5 --fusion_method AUXCF --feedback_time 0 --learning_rate 0.1 --lr_decay_rate 0.1 --add_param_groups_method all --aux_method_t Bottle_big --aux_method_s Bottle_big --self_method_t bi_directional --self_method_s bi_directional'
#basicBottle='train_ours_111.py --model_s ResNet18_cifar_another --model_t ResNet34_cifar_another --path_t ./save/base/cifar100/ResNet34_cifar_another_base/ResNet34_cifar_another_teacher_best.pth.tar --method 1 --in_method inAuxCF_logitsNToAE_OnlyOne-5 --fusion_method AUXCF --feedback_time 0 --learning_rate 0.1 --lr_decay_rate 0.1 --add_param_groups_method all --aux_method_t Bottle --aux_method_s Bottle --self_method_t bi_directional --self_method_s bi_directional'
basicBasic='train_ours_111.py --model_s ResNet18_cifar_another --model_t ResNet34_cifar_another --path_t ./save/base/cifar100/ResNet34_cifar_another_base/ResNet34_cifar_another_teacher_best.pth.tar --method 1 --in_method inAuxCF_logitsNToAE_OnlyOne-5 --fusion_method AUXCF --feedback_time 0 --learning_rate 0.1 --lr_decay_rate 0.1 --add_param_groups_method all --aux_method_t Basic --aux_method_s Basic --self_method_t bi_directional --self_method_s bi_directional'

## inAuxCF_logitsNToAE_OnlyOne-5 + KLsoftmax TpartAll init0.1 decay_rate0.1 method1 AUXCF fwdloss1 四Tblock 四Sblock ilrT5e-3 MultiStep(60,120,160) AELinear Basic
python $basicBasic --lr_decay_epochs 60,120,160 --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --fusion_method_AE_t AELinear --fusion_method_AE_s AELinear --fusion_size Mean -t 19-01_21


# 2
# ResNet101_cifar_another - ResNet34_cifar_another 1-02_01
basicR101R34='train_ours_111.py --model_s ResNet34_cifar_another --model_t ResNet101_cifar_another --path_t ./save/base/cifar100/ResNet101_cifar_another_base/ResNet101_cifar_another_teacher_best.pth.tar --method 1 --in_method inAuxCF_logitsNToAE_OnlyOne-5 --fusion_method AUXCF --feedback_time 0 --learning_rate 0.1 --lr_decay_rate 0.1 --add_param_groups_method all --self_method_t bi_directional --self_method_s bi_directional --lr_decay_epochs 60,120,160 --scheduler_method MultiStepLR --in_criterion KL_softmax --learning_rate_t 5e-3'

## ResNet101_cifar_another - ResNet34_cifar_another 四Tblock 四Sblock ilrT5e-3 MultiStep(60,120,160) AELinear TBasic SBasic
python $basicR101R34 --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --fusion_method_AE_t AELinear --fusion_method_AE_s AELinear --aux_method_t Basic --aux_method_s Basic --fusion_size Mean -t 19-02_01


# 3
# VGG13_BN - VGG8_BN
basicV13V8='train_ours_111.py --model_s VGG8_BN --model_t VGG13_BN --path_t ./save/base/cifar100/VGG13_BN_base/VGG13_BN_teacher_best.pth.tar --method 1 --in_method inAuxCF_logitsNToAE_OnlyOne-5 --fusion_method AUXCF --feedback_time 0 --learning_rate 0.1 --lr_decay_rate 0.1 --add_param_groups_method all --self_method_t bi_directional --self_method_s bi_directional --lr_decay_epochs 60,120,160 --scheduler_method MultiStepLR --in_criterion KL_softmax --learning_rate_t 5e-3 --blocks_amount_t 4 --blocks_amount_s 4 --aux_method_t Basic --aux_method_s Basic'
python $basicV13V8 --infwd_loss 1 --fusion_method_AE_t AELinear --fusion_method_AE_s AELinear --fusion_size Mean -t 1-03_01

python $basicV13V8 --infwd_loss 1 --fusion_method_AE_t AEConv3x3Linear --fusion_method_AE_s AEConv3x3Linear --fusion_size Mean -t 1-03_02

#python $basicV13V8 --infwd_loss 1 --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size Mean -t 1-03_03


# 4
# ResNet34_cifar_another - ShuffleV2
basicR34ShuV2='train_ours_111.py --model_s ShuffleV2 --model_t ResNet34_cifar_another --path_t ./save/base/cifar100/ResNet34_cifar_another_base/ResNet34_cifar_another_teacher_best.pth.tar --method 1 --in_method inAuxCF_logitsNToAE_OnlyOne-5 --fusion_method AUXCF --feedback_time 0 --learning_rate 0.1 --lr_decay_rate 0.1 --add_param_groups_method all --self_method_t bi_directional --self_method_s bi_directional --lr_decay_epochs 60,120,160 --scheduler_method MultiStepLR --in_criterion KL_softmax --learning_rate_t 5e-3 --blocks_amount_t 4 --blocks_amount_s 4 --aux_method_t Basic --aux_method_s Basic'
python $basicR34ShuV2 --infwd_loss 1 --fusion_method_AE_t AELinear --fusion_method_AE_s AELinear --fusion_size Mean -t 1-04_01

python $basicR34ShuV2 --infwd_loss 1 --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size littleSmall -t 1-04_02


# 5
# VGG11_BN - MobileNetV2
basicV11MobV2='train_ours_111.py --model_s MobileNetV2 --model_t VGG11_BN --path_t ./save/base/cifar100/VGG11_BN_base/VGG11_BN_teacher_best.pth.tar --method 1 --in_method inAuxCF_logitsNToAE_OnlyOne-5 --fusion_method AUXCF --feedback_time 0 --learning_rate 0.1 --lr_decay_rate 0.1 --add_param_groups_method all --self_method_t bi_directional --self_method_s bi_directional --lr_decay_epochs 60,120,160 --scheduler_method MultiStepLR --in_criterion KL_softmax --learning_rate_t 5e-3 --blocks_amount_t 4 --blocks_amount_s 4 --aux_method_t Basic --aux_method_s Basic'
python $basicV11MobV2 --infwd_loss 1 --fusion_method_AE_t AELinear --fusion_method_AE_s AELinear --fusion_size Mean -t 1-05_01

python $basicV11MobV2 --infwd_loss 1 --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size littleSmall -t 1-05_02
