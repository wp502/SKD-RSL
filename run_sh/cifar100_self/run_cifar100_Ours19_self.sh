#!/bin/bash

# CIFAR-100 SELF-DISTILLATION

# 1
# ResNet18_cifar_another - ResNet18_cifar_another
basicBasic='train_ours_111.py --model_s ResNet18_cifar_another --model_t ResNet18_cifar_another --path_t ./save/base/cifar100/ResNet18_cifar_another_base/ResNet18_cifar_another_teacher_best.pth.tar --method 1 --in_method inAuxCF_logitsNToAE_OnlyOne-5 --fusion_method AUXCF --feedback_time 0 --learning_rate 0.1 --lr_decay_rate 0.1 --add_param_groups_method all --aux_method_t Basic --aux_method_s Basic --self_method_t bi_directional --self_method_s bi_directional --lr_decay_epochs 60,120,160 --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --learning_rate_t 5e-3 --scheduler_method MultiStepLR'

python $basicBasic --infwd_loss 1 --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size littleSmall -t 1-01_01_self


# 2
# WRN_40_2 - WRN_40_2
basicwrn402self='train_ours_111.py --model_s WRN_40_2 --model_t WRN_40_2 --path_t ./save/base/cifar100/WRN_40_2_cifar100_base/WRN_40_2_teacher_best.pth.tar --method 1 --in_method inAuxCF_logitsNToAE_OnlyOne-5 --fusion_method AUXCF --feedback_time 0 --learning_rate 0.1 --lr_decay_rate 0.1 --add_param_groups_method all --self_method_t bi_directional --self_method_s bi_directional --lr_decay_epochs 60,120,160 --scheduler_method MultiStepLR --in_criterion KL_softmax --learning_rate_t 5e-3 --blocks_amount_t 4 --blocks_amount_s 4 --aux_method_t Basic --aux_method_s Basic'

python $basicwrn402self --infwd_loss 1 --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size littleSmall -t 1-02_01_self


# 3
# VGG11_BN - VGG11_BN
basicV11self='train_ours_111.py --model_s VGG11_BN --model_t VGG11_BN --path_t ./save/base/cifar100/VGG11_BN_base/VGG11_BN_teacher_best.pth.tar --method 1 --in_method inAuxCF_logitsNToAE_OnlyOne-5 --fusion_method AUXCF --feedback_time 0 --learning_rate 0.1 --lr_decay_rate 0.1 --add_param_groups_method all --self_method_t bi_directional --self_method_s bi_directional --lr_decay_epochs 60,120,160 --scheduler_method MultiStepLR --in_criterion KL_softmax --learning_rate_t 5e-3 --blocks_amount_t 4 --blocks_amount_s 4 --aux_method_t Basic --aux_method_s Basic'

python $basicV11self --infwd_loss 1 --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size littleSmall -t 1-03_01_self

