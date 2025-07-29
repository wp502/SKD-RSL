#!/bin/bash
#实验一：传统消融
#BASE:  19-01_30 largeSmall

basicBasic='train_ours_111.py --model_s ResNet18_cifar_another --model_t ResNet34_cifar_another --path_t ./save/base/cifar100/ResNet34_cifar_another_base/ResNet34_cifar_another_teacher_best.pth.tar --method 1 --in_method inAuxCF_logitsNToAE_OnlyOne-5 --fusion_method AUXCF --feedback_time 0 --learning_rate 0.1 --lr_decay_rate 0.1 --add_param_groups_method all --aux_method_t Basic --aux_method_s Basic --self_method_t bi_directional --self_method_s bi_directional --lr_decay_epochs 60,120,160 --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size largeSmall'
#abla-01_01(19-01_30)
python $basicBasic -t 19-01_30

python $basicBasic --fbUseGradSim -t abla-01_02

python $basicBasic --NO_SELF -t abla-01_03

python $basicBasic --NO_FUSION -t abla-01_04

python $basicBasic --NO_FUSION --NO_SELF -t abla-01_05

python $basicBasic --NO_FUSION --fbUseGradSim -t abla-01_06

python $basicBasic --NO_SELF --fbUseGradSim -t abla-01_07

python $basicBasic --NO_FUSION --NO_SELF --fbUseGradSim -t abla-01_08


