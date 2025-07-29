#!/bin/bash

basic='train_ours_13.py --model_s ResNet18_cifar_another --model_t ResNet34_cifar_another --path_t ./save/base/cifar100/ResNet34_cifar_another_base/ResNet34_cifar_another_teacher_best.pth.tar --method 3 --in_method MEAN_STD_allAE --fusion_method Conv --feedback_time 0'
#--add_param_groups_method
#--infwd_loss
#--fusion_method
#--in_criterion
#-t

#base
#mean_std_allAE + MSEsoftmaxT Tpart1 init0.1 decay_rate0.1 method3 Conv fwdloss1 三block ilrT0.05 ConsineSch
#python $basic --add_param_groups_method 1 --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE_softmax_T --blocks_amount 3 --infwd_loss 1 --learning_rate_t 0.05 --scheduler_method CosineAnnealingLR --fb_set_percent 0.3 -t 8-01_01 &&

#mean_std_allAE + MSEsoftmaxT Tpart5 init0.1 decay_rate0.1 method3 Conv fwdloss1 四block ilrT0.05 ConsineSch
#python $basic --add_param_groups_method 5 --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE_softmax_T --blocks_amount 4 --infwd_loss 1 --learning_rate_t 0.05 --scheduler_method CosineAnnealingLR --fb_set_percent 0.3 -t 8-01_02

#mean_std_allAE + MSEsoftmaxT Tpart1 init0.1 decay_rate0.1 method3 Conv fwdloss1 三block ilrT0.05 MultiStep[60,120,160] fbPercent0.3
#python $basic --add_param_groups_method 1 --lr_decay_epochs 60,120,160 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE_softmax_T --blocks_amount 3 --infwd_loss 1 --learning_rate_t 0.05 --scheduler_method MultiStepLR --fb_set_percent 0.3 -t 8-01_03 &&


#测试 ilrT
#mean_std_allAE + MSEsoftmaxT Tpart1 init0.1 decay_rate0.1 method3 Conv fwdloss1 三block ilrT0.1 ConsineSch fbPercent0.3
#python $basic --add_param_groups_method 1 --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE_softmax_T --blocks_amount 3 --infwd_loss 1 --learning_rate_t 0.1 --scheduler_method CosineAnnealingLR --fb_set_percent 0.3 -t 8-02_01 &&

#mean_std_allAE + MSEsoftmaxT Tpart1 init0.1 decay_rate0.1 method3 Conv fwdloss1 三block ilrT1 ConsineSch fbPercent0.3
#python $basic --add_param_groups_method 1 --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE_softmax_T --blocks_amount 3 --infwd_loss 1 --learning_rate_t 1.0 --scheduler_method CosineAnnealingLR --fb_set_percent 0.3 -t 8-02_02 &&


#测试 loss的计算方式
#mean_std_allAE + MSE Tpart1 init0.1 decay_rate0.1 method3 Conv fwdloss1 三block ilrT0.05 ConsineSch fbPercent0.3
#python $basic --add_param_groups_method 1 --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE --blocks_amount 3 --infwd_loss 1 --learning_rate_t 0.05 --scheduler_method CosineAnnealingLR --fb_set_percent 0.3 -t 8-04_01 &&


#测试 fbPercent
#mean_std_allAE + MSEsoftmaxT Tpart1 init0.1 decay_rate0.1 method3 Conv fwdloss1 三block ilrT0.05 ConsineSch fbPercent0.1
#python $basic --add_param_groups_method 1 --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE_softmax_T --blocks_amount 3 --infwd_loss 1 --learning_rate_t 0.05 --scheduler_method CosineAnnealingLR --fb_set_percent 0.1 -t 8-03_01 &&

#mean_std_allAE + MSEsoftmaxT Tpart1 init0.1 decay_rate0.1 method3 Conv fwdloss1 三block ilrT0.05 ConsineSch fbPercent0.2
#python $basic --add_param_groups_method 1 --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE_softmax_T --blocks_amount 3 --infwd_loss 1 --learning_rate_t 0.05 --scheduler_method CosineAnnealingLR --fb_set_percent 0.2 -t 8-03_02 &&

#mean_std_allAE + MSEsoftmaxT Tpart1 init0.1 decay_rate0.1 method3 Conv fwdloss1 三block ilrT0.05 ConsineSch fbPercent0.4
#python $basic --add_param_groups_method 1 --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE_softmax_T --blocks_amount 3 --infwd_loss 1 --learning_rate_t 0.05 --scheduler_method CosineAnnealingLR --fb_set_percent 0.4 -t 8-03_03 &&

#mean_std_allAE + MSEsoftmaxT Tpart1 init0.1 decay_rate0.1 method3 Conv fwdloss1 三block ilrT0.05 ConsineSch fbPercent0.5
#python $basic --add_param_groups_method 1 --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE_softmax_T --blocks_amount 3 --infwd_loss 1 --learning_rate_t 0.05 --scheduler_method CosineAnnealingLR --fb_set_percent 0.5 -t 8-03_04 &&

#mean_std_allAE + MSEsoftmaxT Tpart1 init0.1 decay_rate0.1 method3 Conv fwdloss1 三block ilrT0.05 ConsineSch fbPercent0.6
#python $basic --add_param_groups_method 1 --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE_softmax_T --blocks_amount 3 --infwd_loss 1 --learning_rate_t 0.05 --scheduler_method CosineAnnealingLR --fb_set_percent 0.6 -t 8-03_05 &&


#测试 batch_size
#mean_std_allAE + MSEsoftmaxT Tpart1 init0.1 decay_rate0.1 method3 Conv fwdloss1 三block ilrT0.05 ConsineSch fbPercent0.3 batchsize64
#python $basic --batch_size 64 --add_param_groups_method 1 --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE_softmax_T --blocks_amount 3 --infwd_loss 1 --learning_rate_t 0.05 --scheduler_method CosineAnnealingLR --fb_set_percent 0.3 -t 8-05_01


#测试 fwdloss 参数
#mean_std_allAE + MSEsoftmaxT Tpart1 init0.1 decay_rate0.1 method3 Conv fwdloss1 三block ilrT0.05 ConsineSch fbPercent0.3 batchsize64 将AEloss都乘以10
#python $basic --batch_size 64 --add_param_groups_method 1 --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE_softmax_T --blocks_amount 3 --infwd_loss 1 --learning_rate_t 0.05 --scheduler_method CosineAnnealingLR --fb_set_percent 0.3 --aefwd_t_loss 10 --aefwd_s_loss 10 --aefb_t_loss 10 --aefb_s_loss 10 -t 8-06_01


#测试 T的所有层都打开更新
#mean_std_allAE + MSEsoftmaxT TpartAll init0.1 decay_rate0.1 method3 Conv fwdloss1 三block ilrT1e-4 ConsineSch fbPercent0.3 batchsize64
#python $basic --batch_size 64 --add_param_groups_method all --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE_softmax_T --blocks_amount 3 --infwd_loss 1 --learning_rate_t 1e-4 --scheduler_method CosineAnnealingLR --fb_set_percent 0.3 -t 8-07_01

#mean_std_allAE + MSEsoftmaxT TpartAll init0.1 decay_rate0.1 method3 Conv fwdloss1 三block ilrT1e-1 ConsineSch fbPercent0.3 batchsize64
#python $basic --batch_size 64 --add_param_groups_method all --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE_softmax_T --blocks_amount 3 --infwd_loss 1 --learning_rate_t 1e-1 --scheduler_method CosineAnnealingLR --fb_set_percent 0.3 -t 8-07_02 &&

#mean_std_allAE + MSEsoftmaxT TpartAll init0.1 decay_rate0.1 method3 Conv fwdloss1 三block ilrT1e-2 ConsineSch fbPercent0.3 batchsize64
#python $basic --batch_size 64 --add_param_groups_method all --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE_softmax_T --blocks_amount 3 --infwd_loss 1 --learning_rate_t 1e-2 --scheduler_method CosineAnnealingLR --fb_set_percent 0.3 -t 8-07_03 &&

#mean_std_allAE + MSEsoftmaxT TpartAll init0.1 decay_rate0.1 method3 Conv fwdloss1 三block ilrT1e-3 ConsineSch fbPercent0.3 batchsize64
#python $basic --batch_size 64 --add_param_groups_method all --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE_softmax_T --blocks_amount 3 --infwd_loss 1 --learning_rate_t 1e-3 --scheduler_method CosineAnnealingLR --fb_set_percent 0.3 -t 8-07_04 &&


#mean_std_allAE + MSEsoftmaxT TpartAll init0.1 decay_rate0.1 method3 Conv fwdloss1 三block ilrT1e-5 ConsineSch fbPercent0.3 batchsize64
#python $basic --batch_size 64 --add_param_groups_method all --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE_softmax_T --blocks_amount 3 --infwd_loss 1 --learning_rate_t 1e-5 --scheduler_method CosineAnnealingLR --fb_set_percent 0.3 -t 8-07_05


#mean_std_allAE + MSEsoftmaxT TpartAll init0.1 decay_rate0.1 method3 Conv fwdloss1 三block ilrT5e-3 ConsineSch fbPercent0.3 batchsize64
#python $basic --batch_size 64 --add_param_groups_method all --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE_softmax_T --blocks_amount 3 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method CosineAnnealingLR --fb_set_percent 0.3 -t 8-07_06

#mean_std_allAE + MSEsoftmaxT TpartAll init0.1 decay_rate0.1 method3 Conv fwdloss1 三block ilrT1e-4 ConsineSch fbPercent0.3 batchsize128
#python $basic --batch_size 128 --add_param_groups_method all --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE_softmax_T --blocks_amount 3 --infwd_loss 1 --learning_rate_t 1e-4 --scheduler_method CosineAnnealingLR --fb_set_percent 0.3 --pin_memory -t 8-07_07

#mean_std_allAE + MSEsoftmaxT TpartAll init0.1 decay_rate0.1 method3 Conv fwdloss1 三block ilrT5e-3 ConsineSch fbPercent0.3 batchsize128
#python $basic --batch_size 128 --add_param_groups_method all --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE_softmax_T --blocks_amount 3 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method CosineAnnealingLR --fb_set_percent 0.3 -t 8-07_08

#mean_std_allAE + MSEsoftmaxT TpartAll init0.1 decay_rate0.1 method3 Conv fwdloss1 三block ilrT5e-3 ConsineSch fbPercent0.5 batchsize64
python $basic --batch_size 64 --add_param_groups_method all --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE_softmax_T --blocks_amount 3 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method CosineAnnealingLR --fb_set_percent 0.5 --pin_memory -t 8-07_09

#mean_std_allAE + KLsoftmaxT TpartAll init0.1 decay_rate0.1 method3 Conv fwdloss1 三block ilrT5e-3 ConsineSch fbPercent0.5 batchsize64
python $basic --batch_size 64 --add_param_groups_method all --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion KL_softmax --blocks_amount 3 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method CosineAnnealingLR --fb_set_percent 0.5 --pin_memory -t 8-07_10


