#!/bin/bash

# TINY-IMAGENET
#100 [30,60,90] lr0.1 decay0.1 batchsize 128

# 1
# ResNet34_another - ResNet18_another 1-01_01
basicBasic='train_ours_111.py --dataset tiny_imagenet --num_class 200 --batch_size 128 --model_s ResNet18_another --model_t ResNet34_another --path_t ./save/base/tiny_imagenet/ResNet34_another_tiny_imagenet_base/ResNet34_another_teacher_90_checkpoint_64.59.pth.tar --method 1 --in_method inAuxCF_logitsNToAE_OnlyOne-5 --fusion_method AUXCF --feedback_time 0 --learning_rate 0.1 --lr_decay_rate 0.1 --add_param_groups_method all --self_method_t bi_directional --self_method_s bi_directional --epochs 100 --lr_decay_epochs 30,60,90 --scheduler_method MultiStepLR --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --aux_method_t Basic --aux_method_s Basic'
python $basicBasic --learning_rate_t 5e-3 --infwd_loss 1 --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size littleSmall -t 1-01_01
python $basicBasic --learning_rate_t 5e-3 --infwd_loss 1 --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size largeSmall -t 1-01_02

# 2
# ResNet101_another - ResNet34_another 1-02_01
basicR101R34='train_ours_111.py --dataset tiny_imagenet --num_class 200 --batch_size 128 --model_s ResNet34_another --model_t ResNet101_another --path_t ./save/base/tiny_imagenet/ResNet101_another_tiny_imagenet_base/ResNet101_another_teacher_best.pth.tar --method 1 --in_method inAuxCF_logitsNToAE_OnlyOne-5 --fusion_method AUXCF --feedback_time 0 --learning_rate 0.1 --lr_decay_rate 0.1 --add_param_groups_method all --self_method_t bi_directional --self_method_s bi_directional --epochs 100 --lr_decay_epochs 30,60,90 --scheduler_method MultiStepLR --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --aux_method_t Basic --aux_method_s Basic'
python $basicR101R34 --learning_rate_t 5e-3 --infwd_loss 1 --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size littleSmall -t 1-02_01
python $basicR101R34 --learning_rate_t 5e-3 --infwd_loss 1 --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size largeSmall -t 1-02_02

# 3
# VGG13_BN_IMG - VGG8_BN_IMG
basicV13V8='train_ours_111.py --dataset tiny_imagenet --num_class 200 --batch_size 128 --model_s VGG8_BN_IMG --model_t VGG13_BN_IMG --path_t ./save/base/tiny_imagenet/VGG13_BN_IMG_tiny_imagenet_base/VGG13_BN_IMG_teacher_best.pth.tar --method 1 --in_method inAuxCF_logitsNToAE_OnlyOne-5 --fusion_method AUXCF --feedback_time 0 --learning_rate 0.1 --lr_decay_rate 0.1 --add_param_groups_method all --self_method_t bi_directional --self_method_s bi_directional --epochs 100 --lr_decay_epochs 30,60,90 --scheduler_method MultiStepLR --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --aux_method_t Basic --aux_method_s Basic'
python $basicV13V8 --learning_rate_t 5e-3 --infwd_loss 1 --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size littleSmall -t 1-03_01
python $basicV13V8 --learning_rate_t 5e-3 --infwd_loss 1 --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size largeSmall -t 1-03_02

# 4
# ResNet34_another - ShuffleV2_img
basicR34ShuV2='train_ours_111.py --dataset tiny_imagenet --num_class 200 --batch_size 128 --model_s ShuffleV2_img --model_t ResNet34_another --path_t ./save/base/tiny_imagenet/ResNet34_another_tiny_imagenet_base/ResNet34_another_teacher_90_checkpoint_64.59.pth.tar --method 1 --in_method inAuxCF_logitsNToAE_OnlyOne-5 --fusion_method AUXCF --feedback_time 0 --learning_rate 0.1 --lr_decay_rate 0.1 --add_param_groups_method all --self_method_t bi_directional --self_method_s bi_directional --epochs 100 --lr_decay_epochs 30,60,90 --scheduler_method MultiStepLR --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --aux_method_t Basic --aux_method_s Basic'
python $basicR34ShuV2 --learning_rate_t 5e-3 --infwd_loss 1 --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size littleSmall -t 1-04_01

# 5
# VGG11_BN_IMG - MobileNetV2_img
basicV11MobV2='train_ours_111.py --dataset tiny_imagenet --num_class 200 --batch_size 128 --model_s MobileNetV2_img --model_t VGG11_BN_IMG --path_t ./save/base/tiny_imagenet/VGG11_BN_IMG_tiny_imagenet_base/VGG11_BN_IMG_teacher_best.pth.tar --method 1 --in_method inAuxCF_logitsNToAE_OnlyOne-5 --fusion_method AUXCF --feedback_time 0 --learning_rate 0.1 --lr_decay_rate 0.1 --add_param_groups_method all --self_method_t bi_directional --self_method_s bi_directional --epochs 100 --lr_decay_epochs 30,60,90 --scheduler_method MultiStepLR --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --aux_method_t Basic --aux_method_s Basic'
python $basicV11MobV2 --learning_rate_t 5e-3 --infwd_loss 1 --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size littleSmall -t 1-05_01
