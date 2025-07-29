#!/bin/bash

basic='train_ours_15.py --model_s ResNet18_cifar_another --model_t ResNet34_cifar_another --path_t ./save/base/cifar100/ResNet34_cifar_another_base/ResNet34_cifar_another_teacher_best.pth.tar --method 1 --in_method MEAN_STD_aeFeatCF --fusion_method Conv --feedback_time 0'


#base
#mean_std_aeFeatCF + MSEsoftmaxT TpartAll init0.1 decay_rate0.1 method1 Conv fwdloss1 三block ilrT5e-3 ConsineSch
#python $basic --add_param_groups_method all --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE_softmax_T --blocks_amount 3 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method CosineAnnealingLR -t 10-01_01

#mean_std_aeFeatCF + KLsoftmax TpartAll init0.1 decay_rate0.1 method1 Conv fwdloss1 三block ilrT5e-3 ConsineSch
#python $basic --add_param_groups_method all --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion KL_softmax --blocks_amount 3 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method CosineAnnealingLR -t 10-01_02

#mean_std_aeFeatCF + KLsoftmax TpartAll init0.1 decay_rate0.1 method1 Conv fwdloss1 三block ilrT1e-4 ConsineSch
python $basic --add_param_groups_method all --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion KL_softmax --blocks_amount 3 --infwd_loss 1 --learning_rate_t 1e-4 --scheduler_method CosineAnnealingLR -t 10-01_03

#mean_std_aeFeatCF + KLsoftmax TpartAll init0.1 decay_rate0.1 method1 Conv fwdloss1 四block ilrT5e-3 ConsineSch
python $basic --add_param_groups_method all --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion KL_softmax --blocks_amount 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method CosineAnnealingLR -t 10-01_04

#mean_std_aeFeatCF + KLsoftmax TpartAll init0.1 decay_rate0.1 method1 Conv fwdloss1 三block ilrT1e-3 ConsineSch
python $basic --add_param_groups_method all --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion KL_softmax --blocks_amount 3 --infwd_loss 1 --learning_rate_t 1e-3 --scheduler_method CosineAnnealingLR -t 10-01_05

#mean_std_aeFeatCF + KLsoftmax TpartAll init0.1 decay_rate0.1 method1 Conv fwdloss1 三block ilrT1e-2 ConsineSch
python $basic --add_param_groups_method all --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion KL_softmax --blocks_amount 3 --infwd_loss 1 --learning_rate_t 1e-2 --scheduler_method CosineAnnealingLR -t 10-01_06

#mean_std_aeFeatCF + KLsoftmax TpartAll init0.1 decay_rate0.1 method1 Conv fwdloss1 三block ilrT1e-5 ConsineSch
python $basic --add_param_groups_method all --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion KL_softmax --blocks_amount 3 --infwd_loss 1 --learning_rate_t 1e-5 --scheduler_method CosineAnnealingLR -t 10-01_07

#mean_std_aeFeatCF + KLsoftmax TpartAll init0.1 decay_rate0.1 method1 Conv fwdloss1 四block ilrT1e-3 ConsineSch
python $basic --add_param_groups_method all --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion KL_softmax --blocks_amount 4 --infwd_loss 1 --learning_rate_t 1e-3 --scheduler_method CosineAnnealingLR -t 10-01_08

#mean_std_aeFeatCF + KLsoftmax TpartAll init0.1 decay_rate0.1 method1 Conv fwdloss1 三block ilrT1e-1 ConsineSch
python $basic --add_param_groups_method all --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion KL_softmax --blocks_amount 3 --infwd_loss 1 --learning_rate_t 1e-1 --scheduler_method CosineAnnealingLR -t 10-01_09

