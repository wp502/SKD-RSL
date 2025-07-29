#!/bin/bash
#实验四：比对auxcf块
#BASE:  19-01_30 largeSmall

basicBasic='train_ours_111.py --model_s ResNet18_cifar_another --model_t ResNet34_cifar_another --path_t ./save/base/cifar100/ResNet34_cifar_another_base/ResNet34_cifar_another_teacher_best.pth.tar --method 1 --in_method inAuxCF_logitsNToAE_OnlyOne-5 --fusion_method AUXCF --feedback_time 0 --learning_rate 0.1 --lr_decay_rate 0.1 --add_param_groups_method all --lr_decay_epochs 60,120,160 --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size largeSmall --self_method_t bi_directional --self_method_s bi_directional'
#abla-04_01(19-01_30)
python $basicBasic --aux_method_t Basic --aux_method_s Basic -t 19-01_30


python $basicBasic --aux_method_t Basic_big --aux_method_s Basic_big -t abla-04_02

python $basicBasic --aux_method_t Bottle --aux_method_s Bottle -t abla-04_03

python $basicBasic --aux_method_t Bottle_big --aux_method_s Bottle_big -t abla-04_04


