#!/bin/bash
#实验二：比对自蒸馏的路径
#BASE:  19-01_30 largeSmall

basicBasic='train_ours_111.py --model_s ResNet18_cifar_another --model_t ResNet34_cifar_another --path_t ./save/base/cifar100/ResNet34_cifar_another_base/ResNet34_cifar_another_teacher_best.pth.tar --method 1 --in_method inAuxCF_logitsNToAE_OnlyOne-5 --fusion_method AUXCF --feedback_time 0 --learning_rate 0.1 --lr_decay_rate 0.1 --add_param_groups_method all --aux_method_t Basic --aux_method_s Basic --lr_decay_epochs 60,120,160 --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size largeSmall'
#abla-02_03(19-01_30)
python $basicBasic --self_method_t bi_directional --self_method_s bi_directional -t 19-01_30


python $basicBasic --self_method_t deep_shallow --self_method_s deep_shallow -t abla-02_01

python $basicBasic --self_method_t shallow_deep --self_method_s shallow_deep -t abla-02_02

python $basicBasic --self_method_t deep_shallow_single --self_method_s deep_shallow_single -t abla-02_04

python $basicBasic --self_method_t shallow_deep_single --self_method_s shallow_deep_single -t abla-02_05

