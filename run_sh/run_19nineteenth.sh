#!/bin/bash

basicBig='train_ours_111.py --model_s ResNet18_cifar_another --model_t ResNet34_cifar_another --path_t ./save/base/cifar100/ResNet34_cifar_another_base/ResNet34_cifar_another_teacher_best.pth.tar --method 1 --in_method inAuxCF_logitsNToAE_OnlyOne-5 --fusion_method AUXCF --feedback_time 0 --learning_rate 0.1 --lr_decay_rate 0.1 --add_param_groups_method all --aux_method_t Bottle_big --aux_method_s Bottle_big --self_method_t bi_directional --self_method_s bi_directional'
basicBig3='train_ours_111.py --model_s ResNet18_cifar_another --model_t ResNet34_cifar_another --path_t ./save/base/cifar100/ResNet34_cifar_another_base/ResNet34_cifar_another_teacher_best.pth.tar --method 3 --in_method inAuxCF_logitsNToAE_OnlyOne-5 --fusion_method AUXCF --feedback_time 0 --learning_rate 0.1 --lr_decay_rate 0.1 --add_param_groups_method all --aux_method_t Bottle_big --aux_method_s Bottle_big --self_method_t bi_directional --self_method_s bi_directional'
basicBottle='train_ours_111.py --model_s ResNet18_cifar_another --model_t ResNet34_cifar_another --path_t ./save/base/cifar100/ResNet34_cifar_another_base/ResNet34_cifar_another_teacher_best.pth.tar --method 1 --in_method inAuxCF_logitsNToAE_OnlyOne-5 --fusion_method AUXCF --feedback_time 0 --learning_rate 0.1 --lr_decay_rate 0.1 --add_param_groups_method all --aux_method_t Bottle --aux_method_s Bottle --self_method_t bi_directional --self_method_s bi_directional'
basicBasic='train_ours_111.py --model_s ResNet18_cifar_another --model_t ResNet34_cifar_another --path_t ./save/base/cifar100/ResNet34_cifar_another_base/ResNet34_cifar_another_teacher_best.pth.tar --method 1 --in_method inAuxCF_logitsNToAE_OnlyOne-5 --fusion_method AUXCF --feedback_time 0 --learning_rate 0.1 --lr_decay_rate 0.1 --add_param_groups_method all --aux_method_t Basic --aux_method_s Basic --self_method_t bi_directional --self_method_s bi_directional'


#base

#inAuxCF_logitsNToAE_OnlyOne-5 + KLsoftmax TpartAll init0.1 decay_rate0.1 method1 AUXCF fwdloss1 三block ilrT5e-3 ConsineSch AEConv
python $basicBig --CosineSch_Tmax 200 --in_criterion KL_softmax --blocks_amount_t 3 --blocks_amount_s 3 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method CosineAnnealingLR --fusion_method_AE_t AEConv - --fusion_method_AE_s AEConv --fusion_size Mean -t 19-01_01

#inAuxCF_logitsNToAE_OnlyOne-5 + KLsoftmax TpartAll init0.1 decay_rate0.1 method1 AUXCF fwdloss1 三block ilrT5e-3 MultiStep(60,120,160) AEConv
python $basicBig --lr_decay_epochs 60,120,160 --in_criterion KL_softmax --blocks_amount_t 3 --blocks_amount_s 3 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --fusion_method_AE_t AEConv - --fusion_method_AE_s AEConv --fusion_size Mean -t 19-01_02

#inAuxCF_logitsNToAE_OnlyOne-5 + KLsoftmax TpartAll init0.1 decay_rate0.1 method1 AUXCF fwdloss1 四block ilrT5e-3 ConsineSch AEConv3x3Linear
python $basicBig --CosineSch_Tmax 200 --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method CosineAnnealingLR --fusion_method_AE_t AEConv3x3Linear --fusion_method_AE_s AEConv3x3Linear --fusion_size Mean -t 19-01_03

#inAuxCF_logitsNToAE_OnlyOne-5 + KLsoftmax TpartAll init0.1 decay_rate0.1 method1 AUXCF fwdloss1 四block ilrT5e-3 MultiStep(60,120,160) AEConv
python $basicBig --lr_decay_epochs 60,120,160 --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --fusion_method_AE_t AEConv - --fusion_method_AE_s AEConv --fusion_size Mean -t 19-01_04

#inAuxCF_logitsNToAE_OnlyOne-5 + KLsoftmax TpartAll init0.1 decay_rate0.1 method1 AUXCF fwdloss1 四block ilrT5e-3 MultiStep(60,120,160) AEConv3x3Linear
python $basicBig --lr_decay_epochs 60,120,160 --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --fusion_method_AE_t AEConv3x3Linear --fusion_method_AE_s AEConv3x3Linear --fusion_size Mean -t 19-01_05

#inAuxCF_logitsNToAE_OnlyOne-5 + KLsoftmax TpartAll init0.1 decay_rate0.1 method1 AUXCF fwdloss1 三block ilrT5e-3 MultiStep(60,120,160) AEConv3x3Linear
python $basicBig --lr_decay_epochs 60,120,160 --in_criterion KL_softmax --blocks_amount_t 3 --blocks_amount_s 3 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --fusion_method_AE_t AEConv3x3Linear --fusion_method_AE_s AEConv3x3Linear --fusion_size Mean -t 19-01_06

#inAuxCF_logitsNToAE_OnlyOne-5 + KLsoftmax TpartAll init0.1 decay_rate0.1 method1 AUXCF fwdloss1 三block ilrT5e-3 MultiStep(60,120,160) AEConv AENoRelu
python $basicBig --lr_decay_epochs 60,120,160 --in_criterion KL_softmax --blocks_amount_t 3 --blocks_amount_s 3 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --fusion_method_AE_t AEConv - --fusion_method_AE_s AEConv --ende_use_relu --fusion_size Mean -t 19-01_07

#inAuxCF_logitsNToAE_OnlyOne-5 + KLsoftmax TpartAll init0.1 decay_rate0.1 method1 AUXCF fwdloss1 四Tblock 四Sblock ilrT5e-3 MultiStep(60,120,160) AEConv AENoRelu
python $basicBig --lr_decay_epochs 60,120,160 --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --fusion_method_AE_t AEConv - --fusion_method_AE_s AEConv --ende_use_relu --fusion_size Mean -t 19-01_08

#inAuxCF_logitsNToAE_OnlyOne-5 + KLsoftmax TpartAll init0.1 decay_rate0.1 method3 AUXCF fwdloss1 四Tblock 四Sblock ilrT5e-3 MultiStep(60,120,160) AEConv
python $basicBig3 --lr_decay_epochs 60,120,160 --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --fusion_method_AE_t AEConv - --fusion_method_AE_s AEConv --fusion_size Mean -t 19-01_09

#inAuxCF_logitsNToAE_OnlyOne-5 + KLsoftmax TpartAll init0.1 decay_rate0.1 method1 AUXCF fwdloss1 三Tblock 四Sblock ilrT5e-3 MultiStep(60,120,160) AEConv
python $basicBig --lr_decay_epochs 60,120,160 --in_criterion KL_softmax --blocks_amount_t 3 --blocks_amount_s 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --fusion_method_AE_t AEConv - --fusion_method_AE_s AEConv --fusion_size Mean -t 19-01_10

#inAuxCF_logitsNToAE_OnlyOne-5 + KLsoftmax TpartAll init0.1 decay_rate0.1 method1 AUXCF fwdloss1 四Tblock 四Sblock ilrT5e-3 MultiStep(60,120,160) AEConv3x3Single
python $basicBig --lr_decay_epochs 60,120,160 --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --fusion_method_AE_t AEConv3x3Single --fusion_method_AE_s AEConv3x3Single --fusion_size Mean -t 19-01_11

#inAuxCF_logitsNToAE_OnlyOne-5 + KLsoftmax TpartAll init0.1 decay_rate0.1 method1 AUXCF fwdloss1 四Tblock 四Sblock ilrT5e-3 MultiStep(60,120,160) AELinear
python $basicBig --lr_decay_epochs 60,120,160 --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --fusion_method_AE_t AELinear --fusion_method_AE_s AELinear --fusion_size Mean -t 19-01_12

#inAuxCF_logitsNToAE_OnlyOne-5 + KLsoftmax TpartAll init0.1 decay_rate0.1 method1 AUXCF fwdloss1 四Tblock 四Sblock ilrT5e-3 MultiStep(60,120,160) AEConv3x3LinearNew
python $basicBig --lr_decay_epochs 60,120,160 --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --fusion_method_AE_t AEConv3x3Linear --fusion_method_AE_s AEConv3x3Linear --fusion_size Mean -t 19-01_13

#上面的aux_method用的都是 Bottle_Big stride=1，下面的是stride=2的Bottle

#inAuxCF_logitsNToAE_OnlyOne-5 + KLsoftmax TpartAll init0.1 decay_rate0.1 method1 AUXCF fwdloss1 四Tblock 四Sblock ilrT5e-3 MultiStep(60,120,160) AEConv3x3Single Bottle
python $basicBottle --lr_decay_epochs 60,120,160 --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --fusion_method_AE_t AEConv3x3Single --fusion_method_AE_s AEConv3x3Single --fusion_size Mean -t 19-01_14

#inAuxCF_logitsNToAE_OnlyOne-5 + KLsoftmax TpartAll init0.1 decay_rate0.1 method1 AUXCF fwdloss1 四Tblock 四Sblock ilrT5e-3 MultiStep(60,120,160) AEConv Bottle
python $basicBottle --lr_decay_epochs 60,120,160 --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --fusion_method_AE_t AEConv - --fusion_method_AE_s AEConv --fusion_size Mean -t 19-01_15

#inAuxCF_logitsNToAE_OnlyOne-5 + KLsoftmax TpartAll init0.1 decay_rate0.1 method1 AUXCF fwdloss1 四Tblock 四Sblock ilrT5e-3 MultiStep(60,120,160) AEConv3x3LinearNew Bottle
python $basicBottle --lr_decay_epochs 60,120,160 --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --fusion_method_AE_t AEConv3x3Linear --fusion_method_AE_s AEConv3x3Linear --fusion_size Mean -t 19-01_16

#inAuxCF_logitsNToAE_OnlyOne-5 + KLsoftmax TpartAll init0.1 decay_rate0.1 method1 AUXCF fwdloss1 四Tblock 四Sblock ilrT5e-3 MultiStep(60,120,160) AELinear Bottle
python $basicBottle --lr_decay_epochs 60,120,160 --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --fusion_method_AE_t AELinear --fusion_method_AE_s AELinear --fusion_size Mean -t 19-01_17

#inAuxCF_logitsNToAE_OnlyOne-5 + KLsoftmax TpartAll init0.1 decay_rate0.1 method1 AUXCF fwdloss1 四Tblock 四Sblock ilrT5e-3 MultiStep(60,120,160) AEConv3x3Single Basic
python $basicBasic --lr_decay_epochs 60,120,160 --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --fusion_method_AE_t AEConv3x3Single --fusion_method_AE_s AEConv3x3Single --fusion_size Mean -t 19-01_18

#inAuxCF_logitsNToAE_OnlyOne-5 + KLsoftmax TpartAll init0.1 decay_rate0.1 method1 AUXCF fwdloss1 四Tblock 四Sblock ilrT5e-3 MultiStep(60,120,160) AEConv Basic
python $basicBasic --lr_decay_epochs 60,120,160 --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --fusion_method_AE_t AEConv - --fusion_method_AE_s AEConv --fusion_size Mean -t 19-01_19

#inAuxCF_logitsNToAE_OnlyOne-5 + KLsoftmax TpartAll init0.1 decay_rate0.1 method1 AUXCF fwdloss1 四Tblock 四Sblock ilrT5e-3 MultiStep(60,120,160) AEConv3x3LinearNew Basic
python $basicBasic --lr_decay_epochs 60,120,160 --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --fusion_method_AE_t AEConv3x3Linear --fusion_method_AE_s AEConv3x3Linear --fusion_size Mean -t 19-01_20

#inAuxCF_logitsNToAE_OnlyOne-5 + KLsoftmax TpartAll init0.1 decay_rate0.1 method1 AUXCF fwdloss1 四Tblock 四Sblock ilrT5e-3 MultiStep(60,120,160) AELinear Basic
python $basicBasic --lr_decay_epochs 60,120,160 --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --fusion_method_AE_t AELinear --fusion_method_AE_s AELinear --fusion_size Mean -t 19-01_21

#inAuxCF_logitsNToAE_OnlyOne-5 + MSE_normalize TpartAll init0.1 decay_rate0.1 method1 AUXCF fwdloss1 四Tblock 四Sblock ilrT5e-3 MultiStep(60,120,160) AELinear Basic
python $basicBasic --lr_decay_epochs 60,120,160 --in_criterion MSE_normalize --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --fusion_method_AE_t AELinear --fusion_method_AE_s AELinear --fusion_size Mean -t 19-01_22

#inAuxCF_logitsNToAE_OnlyOne-5 + L1 TpartAll init0.1 decay_rate0.1 method1 AUXCF fwdloss1 四Tblock 四Sblock ilrT5e-3 MultiStep(60,120,160) AELinear Basic
python $basicBasic --lr_decay_epochs 60,120,160 --in_criterion L1 --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --fusion_method_AE_t AELinear --fusion_method_AE_s AELinear --fusion_size Mean -t 19-01_23

#inAuxCF_logitsNToAE_OnlyOne-5 + MSE_softmax TpartAll init0.1 decay_rate0.1 method1 AUXCF fwdloss1 四Tblock 四Sblock ilrT5e-3 MultiStep(60,120,160) AELinear Basic
python $basicBasic --lr_decay_epochs 60,120,160 --in_criterion MSE_softmax --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --fusion_method_AE_t AELinear --fusion_method_AE_s AELinear --fusion_size Mean -t 19-01_24

#inAuxCF_logitsNToAE_OnlyOne-5 + MSE_softmax_T TpartAll init0.1 decay_rate0.1 method1 AUXCF fwdloss1 四Tblock 四Sblock ilrT5e-3 MultiStep(60,120,160) AELinear Basic
python $basicBasic --lr_decay_epochs 60,120,160 --in_criterion MSE_softmax_T --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --fusion_method_AE_t AELinear --fusion_method_AE_s AELinear --fusion_size Mean -t 19-01_25

#inAuxCF_logitsNToAE_OnlyOne-5 + KLsoftmax TpartAll init0.1 decay_rate0.1 method1 AUXCF fwdloss1 四Tblock 四Sblock ilrT5e-3 MultiStep(60,120,160) ADPAELinear Basic fSizeMean
python $basicBasic --lr_decay_epochs 60,120,160 --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size Mean -t 19-01_26

#inAuxCF_logitsNToAE_OnlyOne-5 + KLsoftmax TpartAll init0.1 decay_rate0.1 method1 AUXCF fwdloss1 四Tblock 四Sblock ilrT5e-3 MultiStep(60,120,160) ADPAELinear Basic fSizeBig
python $basicBasic --lr_decay_epochs 60,120,160 --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size Big -t 19-01_27

#inAuxCF_logitsNToAE_OnlyOne-5 + KLsoftmax TpartAll init0.1 decay_rate0.1 method1 AUXCF fwdloss1 四Tblock 四Sblock ilrT5e-3 MultiStep(60,120,160) ADPAELinear Basic fSizeSmall
python $basicBasic --lr_decay_epochs 60,120,160 --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size Small -t 19-01_28

python $basicBasic --lr_decay_epochs 60,120,160 --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size littleSmall -t 19-01_29

python $basicBasic --lr_decay_epochs 60,120,160 --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size largeSmall -t 19-01_30

python $basicBasic --lr_decay_epochs 60,120,160 --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size hugeSmall -t 19-01_31

python $basicBasic --lr_decay_epochs 60,120,160 --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size numClass -t 19-01_32

python $basicBasic --lr_decay_epochs 60,120,160 --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 5 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size littleSmall -t 19-01_33

python $basicBasic --lr_decay_epochs 60,120,160 --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 5 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size largeSmall -t 19-01_34

# 全新的grad sim
python $basicBasic --lr_decay_epochs 60,120,160 --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size littleSmall -t 19-01_35

python $basicBasic --lr_decay_epochs 60,120,160 --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --learning_rate_t 1e-1 --scheduler_method MultiStepLR --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size littleSmall -t 19-01_36

python $basicBasic --lr_decay_epochs 60,120,160 --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --learning_rate_t 1e-2 --scheduler_method MultiStepLR --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size littleSmall -t 19-01_37

python $basicBasic --lr_decay_epochs 60,120,160 --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --learning_rate_t 1e-3 --scheduler_method MultiStepLR --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size littleSmall -t 19-01_38

python $basicBasic --lr_decay_epochs 60,120,160 --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --learning_rate_t 5e-2 --scheduler_method MultiStepLR --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size littleSmall -t 19-01_39

python $basicBasic --lr_decay_epochs 60,120,160 --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 10 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size littleSmall -t 19-01_40

python $basicBasic --lr_decay_epochs 60,120,160 --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 10 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size largeSmall -t 19-01_41

# 测试是否是模型更大，更适合用Bottle，模型更小，更适合用Basic
basicR101R34='train_ours_111.py --model_s ResNet34_cifar_another --model_t ResNet101_cifar_another --path_t ./save/base/cifar100/ResNet101_cifar_another_base/ResNet101_cifar_another_teacher_best.pth.tar --method 1 --in_method inAuxCF_logitsNToAE_OnlyOne-5 --fusion_method AUXCF --feedback_time 0 --learning_rate 0.1 --lr_decay_rate 0.1 --add_param_groups_method all --self_method_t bi_directional --self_method_s bi_directional --lr_decay_epochs 60,120,160 --scheduler_method MultiStepLR --in_criterion KL_softmax --learning_rate_t 5e-3'

# ResNet101_cifar_another - ResNet34_cifar_another 四Tblock 四Sblock ilrT5e-3 MultiStep(60,120,160) AELinear TBasic SBasic
python $basicR101R34 --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --fusion_method_AE_t AELinear --fusion_method_AE_s AELinear --aux_method_t Basic --aux_method_s Basic --fusion_size Mean -t 19-02_01

# ResNet101_cifar_another - ResNet34_cifar_another 四Tblock 四Sblock ilrT5e-3 MultiStep(60,120,160) AELinear TBottle SBottle
python $basicR101R34 --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --fusion_method_AE_t AELinear --fusion_method_AE_s AELinear --aux_method_t Bottle --aux_method_s Bottle --fusion_size Mean -t 19-02_02

# ResNet101_cifar_another - ResNet34_cifar_another 四Tblock 四Sblock ilrT5e-3 MultiStep(60,120,160) AELinear TBottle SBasic
python $basicR101R34 --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --fusion_method_AE_t AELinear --fusion_method_AE_s AELinear --aux_method_t Bottle --aux_method_s Basic --fusion_size Mean -t 19-02_03

python $basicR101R34 --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --aux_method_t Basic --aux_method_s Basic --fusion_size littleSmall -t 19-02_04

# ResNet101_cifar_another - ResNet34_cifar_another 四Tblock 四Sblock ilrT5e-3 MultiStep(60,120,160) AELinear TBottle SBasic
#python $basicR101R34 --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --fusion_method_AUXCF AELinear --aux_method_t Bottle --aux_method_s Basic -t 19-02_04


basicV13V8='train_ours_111.py --model_s VGG8_BN --model_t VGG13_BN --path_t ./save/base/cifar100/VGG13_BN_base/VGG13_BN_teacher_best.pth.tar --method 1 --in_method inAuxCF_logitsNToAE_OnlyOne-5 --fusion_method AUXCF --feedback_time 0 --learning_rate 0.1 --lr_decay_rate 0.1 --add_param_groups_method all --self_method_t bi_directional --self_method_s bi_directional --lr_decay_epochs 60,120,160 --scheduler_method MultiStepLR --in_criterion KL_softmax --learning_rate_t 5e-3 --blocks_amount_t 4 --blocks_amount_s 4 --aux_method_t Basic --aux_method_s Basic'
python $basicV13V8 --infwd_loss 1 --fusion_method_AE_t AELinear --fusion_method_AE_s AELinear --fusion_size Mean -t 19-03_01

python $basicV13V8 --infwd_loss 1 --fusion_method_AE_t AEConv3x3Linear --fusion_method_AE_s AEConv3x3Linear --fusion_size Mean -t 19-03_02

python $basicV13V8 --infwd_loss 1 --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size Mean -t 19-03_03

python $basicV13V8 --infwd_loss 1 --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size Big -t 19-03_04

python $basicV13V8 --infwd_loss 1 --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size Small -t 19-03_05

python $basicV13V8 --infwd_loss 1 --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size littleSmall -t 19-03_06

python $basicV13V8 --infwd_loss 1 --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size largeSmall -t 19-03_07

python $basicV13V8 --infwd_loss 1 --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size hugeSmall -t 19-03_08

python $basicV13V8 --infwd_loss 1 --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size numClass -t 19-03_09

#python $basicV13V8 --infwd_loss 1 --fusion_method_AE_t ADPAEConv3x3Linear --fusion_method_AE_s ADPAEConv3x3Linear --fusion_size Mean -t 19-03_10

#python $basicV13V8 --infwd_loss 1 --fusion_method_AE_t ADPAEConv3x3Linear --fusion_method_AE_s ADPAEConv3x3Linear --fusion_size Small -t 19-03_11



