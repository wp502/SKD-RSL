#!/bin/bash
#实验三：比对fusion方法和fusionsize

#ResNet34-ResNet18  512 - 512
basicR34R18='train_ours_111.py --model_s ResNet18_cifar_another --model_t ResNet34_cifar_another --path_t ./save/base/cifar100/ResNet34_cifar_another_base/ResNet34_cifar_another_teacher_best.pth.tar --method 1 --in_method inAuxCF_logitsNToAE_OnlyOne-5 --fusion_method AUXCF --feedback_time 0 --learning_rate 0.1 --lr_decay_rate 0.1 --add_param_groups_method all --aux_method_t Basic --aux_method_s Basic --self_method_t bi_directional --self_method_s bi_directional --lr_decay_epochs 60,120,160 --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method MultiStepLR'
#abla-01_01(19-01_30)
python $basicR34R18 --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size largeSmall -t 19-01_30


python $basicR34R18 --fusion_method_AE_t AELinear --fusion_method_AE_s AELinear --fusion_size Mean -t abla-03-01_01
python $basicR34R18 --fusion_method_AE_t AELinear --fusion_method_AE_s AELinear --fusion_size littleSmall -t abla-03-01_08
python $basicR34R18 --fusion_method_AE_t AELinear --fusion_method_AE_s AELinear --fusion_size largeSmall -t abla-03-01_09


#ResNet18-MobileNetV2 512 - 320
basicR18MV2='train_ours_111.py --model_s MobileNetV2 --model_t ResNet18_cifar_another --path_t ./save/base/cifar100/ResNet18_cifar_another_base/ResNet18_cifar_another_teacher_best.pth.tar --method 1 --in_method inAuxCF_logitsNToAE_OnlyOne-5 --fusion_method AUXCF --feedback_time 0 --learning_rate 0.1 --lr_decay_rate 0.1 --add_param_groups_method all --aux_method_t Basic --aux_method_s Basic --self_method_t bi_directional --self_method_s bi_directional --lr_decay_epochs 60,120,160 --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method MultiStepLR'

python $basicR18MV2 --fusion_method_AE_t AELinear --fusion_method_AE_s AELinear --fusion_size Mean -t abla-03-02_01
python $basicR18MV2 --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size Big -t abla-03-02_02
python $basicR18MV2 --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size Mean -t abla-03-02_03
python $basicR18MV2 --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size littleSmall -t abla-03-02_04
python $basicR18MV2 --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size Small -t abla-03-02_05
python $basicR18MV2 --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size largeSmall -t abla-03-02_06
python $basicR18MV2 --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size hugeSmall -t abla-03-02_07
python $basicR18MV2 --fusion_method_AE_t AELinear --fusion_method_AE_s AELinear --fusion_size littleSmall -t abla-03-02_08
python $basicR18MV2 --fusion_method_AE_t AELinear --fusion_method_AE_s AELinear --fusion_size largeSmall -t abla-03-02_09


#VGG13_BN-ShuffleV2  512 - 1024
basicV13BNSV2='train_ours_111.py --model_s ShuffleV2 --model_t VGG13_BN --path_t ./save/base/cifar100/VGG13_BN_base/VGG13_BN_teacher_best.pth.tar --method 1 --in_method inAuxCF_logitsNToAE_OnlyOne-5 --fusion_method AUXCF --feedback_time 0 --learning_rate 0.1 --lr_decay_rate 0.1 --add_param_groups_method all --aux_method_t Basic --aux_method_s Basic --self_method_t bi_directional --self_method_s bi_directional --lr_decay_epochs 60,120,160 --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method MultiStepLR'

python $basicV13BNSV2 --fusion_method_AE_t AELinear --fusion_method_AE_s AELinear --fusion_size Mean -t abla-03-03_01
python $basicV13BNSV2 --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size Big -t abla-03-03_02
python $basicV13BNSV2 --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size Mean -t abla-03-03_03
python $basicV13BNSV2 --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size littleSmall -t abla-03-03_04
python $basicV13BNSV2 --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size Small -t abla-03-03_05
python $basicV13BNSV2 --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size largeSmall -t abla-03-03_06
python $basicV13BNSV2 --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size hugeSmall -t abla-03-03_07
python $basicV13BNSV2 --fusion_method_AE_t AELinear --fusion_method_AE_s AELinear --fusion_size littleSmall -t abla-03-03_08
python $basicV13BNSV2 --fusion_method_AE_t AELinear --fusion_method_AE_s AELinear --fusion_size largeSmall -t abla-03-03_09



