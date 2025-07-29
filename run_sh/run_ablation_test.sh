#!/bin/bash

basic='train_ours_111.py --model_s ResNet18_cifar_another --model_t ResNet34_cifar_another --path_t ./save/base/cifar100/ResNet34_cifar_another_base/ResNet34_cifar_another_teacher_best.pth.tar --method 1 --in_method inAuxCF_logitsNToAE_OnlyOne-5 --fusion_method AUXCF --feedback_time 0 --learning_rate 0.1 --lr_decay_rate 0.1 --add_param_groups_method all'

basicBasic='train_ours_111.py --model_s ResNet18_cifar_another --model_t ResNet34_cifar_another --path_t ./save/base/cifar100/ResNet34_cifar_another_base/ResNet34_cifar_another_teacher_best.pth.tar --method 1 --in_method inAuxCF_logitsNToAE_OnlyOne-5 --fusion_method AUXCF --feedback_time 0 --learning_rate 0.1 --lr_decay_rate 0.1 --add_param_groups_method all --aux_method_t Basic --aux_method_s Basic --self_method_t bi_directional --self_method_s bi_directional --lr_decay_epochs 60,120,160 --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --fusion_method_AUXCF AEConv3x3Linear'
#inAuxCF_logitsNToAE_OnlyOne-5 + KLsoftmax TpartAll init0.1 decay_rate0.1 method1 AUXCF fwdloss1 四Tblock 四Sblock ilrT5e-3 MultiStep(60,120,160) AEConv3x3LinearNew Basic
python $basicBasic -t 19-01_20

#ABLATION
#--NO_SELF
#--NO_FUSION
#--fbUseGradSim

#abla_01
python $basicBasic --fbUseGradSim -t abla_01
#abla_02
python $basicBasic --NO_SELF -t abla_02
#abla_03
python $basicBasic --NO_FUSION -t abla_03
#abla_04
python $basicBasic --NO_FUSION --NO_SELF -t abla_04
#abla_05
python $basicBasic --NO_FUSION --fbUseGradSim -t abla_05
#abla_06
python $basicBasic --NO_SELF --fbUseGradSim -t abla_06
#abla_07
python $basicBasic --NO_FUSION --NO_SELF --fbUseGradSim -t abla_07



#inAuxCF_logitsNToAE_OnlyOne-5 + KLsoftmax TpartAll init0.1 decay_rate0.1 method1 AUXCF fwdloss1 四Tblock 四Sblock ilrT5e-3 MultiStep(60,120,160) AEConv3x3Single
python $basic --lr_decay_epochs 60,120,160 --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --fusion_method_AUXCF AEConv3x3Single --self_method_t bi_directional --self_method_s bi_directional --aux_method_t Bottle_big --aux_method_s Bottle_big -t 19-01_11

#19-01_11 Bottle deep-shallow
python $basic --lr_decay_epochs 60,120,160 --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --fusion_method_AUXCF AEConv3x3Single --self_method_t deep_shallow --self_method_s deep_shallow --aux_method_t Bottle_big --aux_method_s Bottle_big -t 19-01_11_selfMethod_1

#19-01_11 Bottle shallow-deep
python $basic --lr_decay_epochs 60,120,160 --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --fusion_method_AUXCF AEConv3x3Single --self_method_t shallow_deep --self_method_s shallow_deep --aux_method_t Bottle_big --aux_method_s Bottle_big -t 19-01_11_selfMethod_2

#19-01_11 Bottle deep_shallow_single
python $basic --lr_decay_epochs 60,120,160 --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --fusion_method_AUXCF AEConv3x3Single --self_method_t deep_shallow_single --self_method_s deep_shallow_single --aux_method_t Bottle_big --aux_method_s Bottle_big -t 19-01_11_selfMethod_3

#19-01_11 Bottle shallow_deep_single
python $basic --lr_decay_epochs 60,120,160 --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --fusion_method_AUXCF AEConv3x3Single --self_method_t shallow_deep_single --self_method_s shallow_deep_single --aux_method_t Bottle_big --aux_method_s Bottle_big -t 19-01_11_selfMethod_4


#19-01_11 Bi-directional Basic
python $basic --lr_decay_epochs 60,120,160 --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --fusion_method_AUXCF AEConv3x3Single --aux_method_t Basic_big --aux_method_s Basic_big -t 19-01_11_auxMethod_1


