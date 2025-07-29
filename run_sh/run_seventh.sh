#!/bin/bash

basic='train_ours_13.py --model_s ResNet18_cifar_another --model_t ResNet34_cifar_another --path_t ./save/base/cifar100/ResNet34_cifar_another_base/ResNet34_cifar_another_teacher_best.pth.tar --method 1 --in_method MEAN_STD_allAE --fusion_method Conv --feedback_time 0'
#--add_param_groups_method
#--infwd_loss
#--fusion_method
#--in_criterion
#-t

#base
#mean_std_allAE + MSE Tpart2 init0.1 decay_rate0.1 method1 Conv fwdloss1 三block ilrT0.1
#python $basic --add_param_groups_method 2 --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE --blocks_amount 3 --infwd_loss 1 --learning_rate_t 0.1 --scheduler_method MultiStepLR -t 7-01_01_Tpart2_ilr0.1_ilrT0.1_threeBlock_ConsineSch &&

#mean_std_allAE + MSE Tpart2 init0.1 decay_rate0.1 method1 Conv fwdloss1 三block ilrT0.1 ConsineSch
#python $basic --add_param_groups_method 2 --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE --blocks_amount 3 --infwd_loss 1 --learning_rate_t 0.1 --scheduler_method CosineAnnealingLR -t 7-01_02_Tpart2_ilr0.1_ilrT0.1_threeBlock_ConsineSch &&

#mean_std_allAE + MSE Tpart2 init0.05 decay_rate0.1 method1 Conv fwdloss1 三block ilrT0.1 ConsineSch
#python $basic --add_param_groups_method 2 --CosineSch_Tmax 200 --learning_rate 0.05 --lr_decay_rate 0.1 --in_criterion MSE --blocks_amount 3 --infwd_loss 1 --learning_rate_t 0.1 --scheduler_method CosineAnnealingLR -t 7-01_03_Tpart2_ilr0.05_ilrT0.1_threeBlock_ConsineSch &&

#mean_std_allAE + MSE Tpart2 init0.05 decay_rate0.1 method1 Conv fwdloss1 三block ilrT0.05 ConsineSch
#python $basic --add_param_groups_method 2 --CosineSch_Tmax 200 --learning_rate 0.05 --lr_decay_rate 0.1 --in_criterion MSE --blocks_amount 3 --infwd_loss 1 --learning_rate_t 0.05 --scheduler_method CosineAnnealingLR -t 7-01_04_Tpart2_ilr0.05_ilrT0.05_threeBlock_ConsineSch &&

#mean_std_allAE + MSE Tpart1 init0.1 decay_rate0.1 method1 Conv fwdloss1 三block ilrT0.1 ConsineSch
#python $basic --add_param_groups_method 1 --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE --blocks_amount 3 --infwd_loss 1 --learning_rate_t 0.1 --scheduler_method CosineAnnealingLR -t 7-01_05_Tpart1_ilr0.1_ilrT0.1_threeBlock_ConsineSch &&

#mean_std_allAE + MSE Tpart1 init0.1 decay_rate0.1 method1 Conv fwdloss1 四block ilrT0.1 ConsineSch
#python $basic --add_param_groups_method 1 --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE --blocks_amount 4 --infwd_loss 1 --learning_rate_t 0.1 --scheduler_method CosineAnnealingLR -t 7-01_06_Tpart1_ilr0.1_ilrT0.1_fourBlock_ConsineSch &&

#mean_std_allAE + MSE Tpart12 init0.1 decay_rate0.1 method1 Conv fwdloss1 三block ilrT0.1 ConsineSch
#python $basic --add_param_groups_method 12 --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE --blocks_amount 3 --infwd_loss 1 --learning_rate_t 0.1 --scheduler_method CosineAnnealingLR -t 7-01_07_Tpart12_ilr0.1_ilrT0.1_threeBlock_ConsineSch &&

#mean_std_allAE + MSE Tpart1 init0.1 decay_rate0.1 method1 Conv fwdloss1 三block ilrT0.05 ConsineSch
#python $basic --add_param_groups_method 1 --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE --blocks_amount 3 --infwd_loss 1 --learning_rate_t 0.05 --scheduler_method CosineAnnealingLR -t 7-01_08_Tpart1_ilr0.1_ilrT0.05_threeBlock_ConsineSch


#测试 Tpart 部分
#mean_std_allAE + MSE Tpart2 init0.1 decay_rate0.1 method1 Conv fwdloss1 三block ilrT0.05 ConsineSch
#python $basic --add_param_groups_method 2 --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE --blocks_amount 3 --infwd_loss 1 --learning_rate_t 0.05 --scheduler_method CosineAnnealingLR -t 7-01_09_Tpart2_ilr0.1_ilrT0.05_threeBlock_ConsineSch

#mean_std_allAE + MSE Tpart12 init0.1 decay_rate0.1 method1 Conv fwdloss1 三block ilrT0.05 ConsineSch
#python $basic --add_param_groups_method 12 --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE --blocks_amount 3 --infwd_loss 1 --learning_rate_t 0.05 --scheduler_method CosineAnnealingLR -t 7-01_10_Tpart12_ilr0.1_ilrT0.05_threeBlock_ConsineSch

#mean_std_allAE + MSE Tpart2 init0.1 decay_rate0.1 method1 Conv fwdloss1 四block ilrT0.05 ConsineSch
#python $basic --add_param_groups_method 2 --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE --blocks_amount 4 --infwd_loss 1 --learning_rate_t 0.05 --scheduler_method CosineAnnealingLR -t 7-01_11_Tpart2_ilr0.1_ilrT0.05_fourBlock_ConsineSch

#mean_std_allAE + MSE Tpart1 init0.1 decay_rate0.1 method1 Conv fwdloss1 三block ilrT0.05 MultiStep
#python $basic --add_param_groups_method 1 --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE --blocks_amount 3 --infwd_loss 1 --learning_rate_t 0.05 --scheduler_method MultiStepLR -t 7-01_12_Tpart1_ilr0.1_ilrT0.05_threeBlock_MultiStepSch

#mean_std_allAE + MSE Tpart5 init0.1 decay_rate0.1 method1 Conv fwdloss1 三block ilrT0.05 ConsineSch
#python $basic --add_param_groups_method 5 --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE --blocks_amount 3 --infwd_loss 1 --learning_rate_t 0.05 --scheduler_method CosineAnnealingLR -t 7-01_13_Tpart5_ilr0.1_ilrT0.05_threeBlock_ConsineSch

#mean_std_allAE + MSE Tpart6 init0.1 decay_rate0.1 method1 Conv fwdloss1 三block ilrT0.05 ConsineSch
#python $basic --add_param_groups_method 6 --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE --blocks_amount 3 --infwd_loss 1 --learning_rate_t 0.05 --scheduler_method CosineAnnealingLR -t 7-01_14_Tpart6_ilr0.1_ilrT0.05_threeBlock_ConsineSch


#测试 Cosine的T_max
#mean_std_allAE + MSE Tpart1 init0.1 decay_rate0.1 method1 Conv fwdloss1 三block ilrT0.05 ConsineSch T_max100
#python $basic --add_param_groups_method 1 --CosineSch_Tmax 100 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE --blocks_amount 3 --infwd_loss 1 --learning_rate_t 0.05 --scheduler_method CosineAnnealingLR -t 7-03_01_Tpart1_ilr0.1_ilrT0.05_threeBlock_ConsineSch_Tmax100


#测试 loss的计算方式
#mean_std_allAE + KLsoftmax Tpart1 init0.1 decay_rate0.1 method1 Conv fwdloss1 三block ilrT0.05 ConsineSch
#python $basic --add_param_groups_method 1 --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion KL_softmax --blocks_amount 3 --infwd_loss 1 --learning_rate_t 0.05 --scheduler_method CosineAnnealingLR -t 7-04_01_KLsoftmax_Tpart1_ilr0.1_ilrT0.05_threeBlock_ConsineSch

#mean_std_allAE + MSEsoftmax Tpart1 init0.1 decay_rate0.1 method1 Conv fwdloss1 三block ilrT0.05 ConsineSch
#python $basic --add_param_groups_method 1 --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE_softmax --blocks_amount 3 --infwd_loss 1 --learning_rate_t 0.05 --scheduler_method CosineAnnealingLR -t 7-04_02_MSEsoftmax_Tpart1_ilr0.1_ilrT0.05_threeBlock_ConsineSch

#mean_std_allAE + MSEsoftmaxT Tpart1 init0.1 decay_rate0.1 method1 Conv fwdloss1 三block ilrT0.05 ConsineSch
#python $basic --add_param_groups_method 1 --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE_softmax_T --blocks_amount 3 --infwd_loss 1 --learning_rate_t 0.05 --scheduler_method CosineAnnealingLR -t 7-04_03_MSEsoftmaxT_Tpart1_ilr0.1_ilrT0.05_threeBlock_ConsineSch

#mean_std_allAE + MSEsoftmaxT Tpart1 init0.1 decay_rate0.1 method1 Conv fwdloss1 三block ilrT0.05 ConsineSch repeat1
#python $basic --add_param_groups_method 1 --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE_softmax_T --blocks_amount 3 --infwd_loss 1 --learning_rate_t 0.05 --scheduler_method CosineAnnealingLR -t 7-04_04_MSEsoftmaxT_Tpart1_ilr0.1_ilrT0.05_threeBlock_ConsineSch_repeat1

#mean_std_allAE + MSEsoftmaxT Tpart1 init0.1 decay_rate0.1 method1 Conv fwdloss1 三block ilrT0.05 MultiStep
#python $basic --add_param_groups_method 1 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE_softmax_T --blocks_amount 3 --infwd_loss 1 --learning_rate_t 0.05 --scheduler_method MultiStepLR -t 7-04_05_MSEsoftmaxT_Tpart1_ilr0.1_ilrT0.05_threeBlock_MultiStep

#mean_std_allAE + MSEsoftmaxT Tpart2 init0.1 decay_rate0.1 method1 Conv fwdloss1 三block ilrT0.05 ConsineSch
#python $basic --add_param_groups_method 2 --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE_softmax_T --blocks_amount 3 --infwd_loss 1 --learning_rate_t 0.05 --scheduler_method CosineAnnealingLR -t 7-04_06_MSEsoftmaxT_Tpart2_ilr0.1_ilrT0.05_threeBlock_ConsineSch

#mean_std_allAE + MSEsoftmaxT Tpart12 init0.1 decay_rate0.1 method1 Conv fwdloss1 三block ilrT0.05 ConsineSch
#python $basic --add_param_groups_method 12 --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE_softmax_T --blocks_amount 3 --infwd_loss 1 --learning_rate_t 0.05 --scheduler_method CosineAnnealingLR -t 7-04_07_MSEsoftmaxT_Tpart12_ilr0.1_ilrT0.05_threeBlock_ConsineSch

#mean_std_allAE + MSEsoftmaxT Tpart1 init0.1 decay_rate0.1 method1 Conv fwdloss1 四block ilrT0.05 ConsineSch
#python $basic --add_param_groups_method 1 --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE_softmax_T --blocks_amount 4 --infwd_loss 1 --learning_rate_t 0.05 --scheduler_method CosineAnnealingLR -t 7-04_08_MSEsoftmaxT_Tpart1_ilr0.1_ilrT0.05_fourBlock_ConsineSch

#测试 decay_rate
#mean_std_allAE + MSE Tpart1 init0.1 decay_rate0.5 method1 Conv fwdloss1 三block ilrT0.05 ConsineSch
#python $basic --add_param_groups_method 1 --learning_rate 0.1 --lr_decay_rate 0.5 --in_criterion MSE --blocks_amount 3 --infwd_loss 1 --learning_rate_t 0.05 --scheduler_method MultiStepLR -t 7-05_03_Tpart1_ilr0.1_dlr0.5_ilrT0.05_threeBlock

#mean_std_allAE + MSE Tpart1 init0.1 decay_rate0.05 method1 Conv fwdloss1 三block ilrT0.05 ConsineSch
#python $basic --add_param_groups_method 1 --learning_rate 0.1 --lr_decay_rate 0.05 --in_criterion MSE --blocks_amount 3 --infwd_loss 1 --learning_rate_t 0.05 --scheduler_method MultiStepLR -t 7-05_04_Tpart1_ilr0.1_dlr0.05_ilrT0.05_threeBlock


#测试 fwdloss 参数
basic_fwdloss='train_ours_13.py --model_s ResNet18_cifar_another --model_t ResNet34_cifar_another --path_t ./save/base/cifar100/ResNet34_cifar_another_base/ResNet34_cifar_another_teacher_best.pth.tar --feedback_time 0 --in_method MEAN_STD_allAE --method 1 --fusion_method Conv --learning_rate 0.1 --lr_decay_rate 0.1 --learning_rate_t 0.05 --scheduler_method CosineAnnealingLR --CosineSch_Tmax 200'
#[mean_std_allAE, method1, Conv, init0.1 ilrT0.05, ConsineSch, 四block， MSEsoftmaxT， Tpart1] fwdloss5
#python $basic_fwdloss  --blocks_amount 4 --add_param_groups_method 1  --in_criterion MSE_softmax_T --infwd_loss 5 -t 7-06_01_MSEsoftmaxT_Tpart1_fwdloss5

#[mean_std_allAE, method1, Conv, init0.1 ilrT0.05, ConsineSch, 四block， MSEsoftmaxT， Tpart1] fwdloss10
#python $basic_fwdloss  --blocks_amount 4 --add_param_groups_method 1 --in_criterion MSE_softmax_T --infwd_loss 10 -t 7-06_02_MSEsoftmaxT_Tpart1_fwdloss10

#[mean_std_allAE, method1, Conv, init0.1 ilrT0.05, ConsineSch, 四block， MSE， Tpart2] fwdloss5
#python $basic_fwdloss  --blocks_amount 4 --add_param_groups_method 2 --in_criterion MSE --infwd_loss 5 -t 7-06_03_MSE_Tpart2_fwdloss5

#[mean_std_allAE, method1, Conv, init0.1 ilrT0.05, ConsineSch, 四block， MSE， Tpart2] fwdloss10
#python $basic_fwdloss  --blocks_amount 4 --add_param_groups_method 2 --in_criterion MSE --infwd_loss 10 -t 7-06_04_MSE_Tpart2_fwdloss10

#[mean_std_allAE, method1, Conv, init0.1 ilrT0.05, ConsineSch, 三block， MSEsoftmaxT， Tpart1] fwdloss10
#python $basic_fwdloss  --blocks_amount 3 --in_criterion MSE_softmax_T --add_param_groups_method 1 --infwd_loss 10 -t 7-06_05_Tpart1_fwdloss10_threeBlock

#[mean_std_allAE, method1, Conv, init0.1 ilrT0.05, ConsineSch, 四block， MSEsoftmaxT， Tpart5] fwdloss1
#python $basic_fwdloss  --blocks_amount 4 --in_criterion MSE_softmax_T --add_param_groups_method 5 --infwd_loss 1 -t 7-06_06_Tpart5_fwdloss1_fourBlock

#[mean_std_allAE, method1, Conv, init0.1 ilrT0.05, ConsineSch, 四block， MSEsoftmaxT， Tpart6] fwdloss1
#python $basic_fwdloss  --blocks_amount 4 --in_criterion MSE_softmax_T --add_param_groups_method 6 --infwd_loss 1 -t 7-06_07_Tpart6_fwdloss1_fourBlock


#测试 ende是否使用relu

#mean_std_allAE + MSEsoftmaxT Tpart1 init0.1 decay_rate0.1 method1 Conv fwdloss1 三block ilrT0.05 ConsineSch 无bn和relu
#python $basic --add_param_groups_method 1 --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE_softmax_T --blocks_amount 3 --infwd_loss 1 --learning_rate_t 0.05 --scheduler_method CosineAnnealingLR --ende_use_relu -t 7-07_01_MSEsoftmaxT_Tpart1_noBNRELU &&

#mean_std_allAE + MSE Tpart1 init0.1 decay_rate0.1 method1 Conv fwdloss1 三block ilrT0.05 ConsineSch 无bn和relu
#python $basic --add_param_groups_method 1 --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE --blocks_amount 3 --infwd_loss 1 --learning_rate_t 0.05 --scheduler_method CosineAnnealingLR --ende_use_relu -t 7-07_02_MSE_Tpart1_noBNRELU

#mean_std_allAE + MSEsoftmaxT Tpart1 init0.1 decay_rate0.1 method1 Conv fwdloss1 三block ilrT0.05 ConsineSch 无relu
#python $basic --add_param_groups_method 1 --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE_softmax_T --blocks_amount 3 --infwd_loss 1 --learning_rate_t 0.05 --scheduler_method CosineAnnealingLR --ende_use_relu -t 7-07_03_MSEsoftmaxT_Tpart1_noRELU &&

#mean_std_allAE + MSE Tpart1 init0.1 decay_rate0.1 method1 Conv fwdloss1 三block ilrT0.05 ConsineSch 无relu
#python $basic --add_param_groups_method 1 --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE --blocks_amount 3 --infwd_loss 1 --learning_rate_t 0.05 --scheduler_method CosineAnnealingLR --ende_use_relu -t 7-07_04_MSE_Tpart1_noRELU



#测试 T的所有层都打开更新 测试不同ilrT
#mean_std_allAE + MSEsoftmaxT TpartAll init0.1 decay_rate0.1 method1 Conv fwdloss1 三block ilrT1e-4 ConsineSch
#python $basic --add_param_groups_method all --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE_softmax_T --blocks_amount 3 --infwd_loss 1 --learning_rate_t 1e-4 --scheduler_method CosineAnnealingLR --pin_memory -t 7-08_01

#mean_std_allAE + MSEsoftmaxT TpartAll init0.1 decay_rate0.1 method1 Conv fwdloss1 四block ilrT1e-4 ConsineSch
#python $basic --add_param_groups_method all --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE_softmax_T --blocks_amount 4 --infwd_loss 1 --learning_rate_t 1e-4 --scheduler_method CosineAnnealingLR --pin_memory -t 7-08_02

#mean_std_allAE + MSEsoftmaxT TpartAll init0.1 decay_rate0.1 method1 Conv fwdloss1 三block ilrT1e-3 ConsineSch
#python $basic --add_param_groups_method all --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE_softmax_T --blocks_amount 3 --infwd_loss 1 --learning_rate_t 1e-3 --scheduler_method CosineAnnealingLR --pin_memory -t 7-08_03

#mean_std_allAE + MSEsoftmaxT TpartAll init0.1 decay_rate0.1 method1 Conv fwdloss1 三block ilrT1e-5 ConsineSch
#python $basic --add_param_groups_method all --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE_softmax_T --blocks_amount 3 --infwd_loss 1 --learning_rate_t 1e-5 --scheduler_method CosineAnnealingLR --pin_memory -t 7-08_04

#mean_std_allAE + MSEsoftmaxT TpartAll init0.1 decay_rate0.1 method1 Conv fwdloss1 三block ilrT5e-4 ConsineSch
#python $basic --add_param_groups_method all --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE_softmax_T --blocks_amount 3 --infwd_loss 1 --learning_rate_t 5e-4 --scheduler_method CosineAnnealingLR --pin_memory -t 7-08_05

#mean_std_allAE + MSEsoftmaxT TpartAll_fc init0.1 decay_rate0.1 method1 Conv fwdloss1 三block ilrT1e-4 linear*10 ConsineSch
#python $basic --add_param_groups_method all_fc --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE_softmax_T --blocks_amount 3 --infwd_loss 1 --learning_rate_t 1e-4 --scheduler_method CosineAnnealingLR --pin_memory -t 7-08_06

#mean_std_allAE + MSEsoftmaxT TpartAll_fc init0.1 decay_rate0.1 method1 Conv fwdloss1 三block ilrT5e-5 linear*10 ConsineSch
#python $basic --add_param_groups_method all_fc --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE_softmax_T --blocks_amount 3 --infwd_loss 1 --learning_rate_t 5e-5 --scheduler_method CosineAnnealingLR --pin_memory -t 7-08_07

#mean_std_allAE + MSEsoftmaxT TpartAll init0.1 decay_rate0.1 method1 Conv fwdloss1 三block ilrT5e-3 ConsineSch
#python $basic --add_param_groups_method all --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE_softmax_T --blocks_amount 3 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method CosineAnnealingLR --pin_memory -t 7-08_08

#mean_std_allAE + MSEsoftmaxT TpartAll_fc init0.1 decay_rate0.1 method1 Conv fwdloss1 三block ilrT5e-3 linear*10 ConsineSch
#python $basic --add_param_groups_method all_fc --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE_softmax_T --blocks_amount 3 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method CosineAnnealingLR --pin_memory -t 7-08_09

#mean_std_allAE + MSEsoftmaxT TpartAll init0.1 decay_rate0.1 method1 Conv fwdloss1 四block ilrT5e-3 ConsineSch
#python $basic --add_param_groups_method all --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE_softmax_T --blocks_amount 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method CosineAnnealingLR --pin_memory -t 7-08_10

#mean_std_allAE + MSE TpartAll init0.1 decay_rate0.1 method1 Conv fwdloss1 三block ilrT5e-3 ConsineSch
#python $basic --add_param_groups_method all --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE --blocks_amount 3 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method CosineAnnealingLR --pin_memory -t 7-08_11

#mean_std_allAE + MSEsoftmax TpartAll init0.1 decay_rate0.1 method1 Conv fwdloss1 三block ilrT5e-3 ConsineSch
#python $basic --add_param_groups_method all --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE_softmax --blocks_amount 3 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method CosineAnnealingLR --pin_memory -t 7-08_12

#mean_std_allAE + KLsoftmax TpartAll init0.1 decay_rate0.1 method1 Conv fwdloss1 三block ilrT5e-3 ConsineSch
#python $basic --add_param_groups_method all --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion KL_softmax --blocks_amount 3 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method CosineAnnealingLR --pin_memory -t 7-08_13

#mean_std_allAE + MSEsoftmaxT TpartAll init0.1 decay_rate0.1 method1 Conv fwdloss1 三block ilrT5e-3 MultiStep
#python $basic --add_param_groups_method all --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE_softmax_T --blocks_amount 3 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --pin_memory -t 7-08_14


#测试 kd_T

#mean_std_allAE + MSEsoftmaxT TpartAll init0.1 decay_rate0.1 method1 Conv fwdloss1 三block ilrT5e-3 ConsineSch kdT 2
#python $basic --add_param_groups_method all --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE_softmax_T --blocks_amount 3 --infwd_loss 1 --kd_T 2 --learning_rate_t 5e-3 --scheduler_method CosineAnnealingLR --pin_memory -t 7-09_01

#mean_std_allAE + MSEsoftmaxT TpartAll init0.1 decay_rate0.1 method1 Conv fwdloss1 三block ilrT5e-3 ConsineSch kdT 3
#python $basic --add_param_groups_method all --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE_softmax_T --blocks_amount 3 --infwd_loss 1 --kd_T 3 --learning_rate_t 5e-3 --scheduler_method CosineAnnealingLR --pin_memory -t 7-09_02

#mean_std_allAE + MSEsoftmaxT TpartAll init0.1 decay_rate0.1 method1 Conv fwdloss1 三block ilrT5e-3 ConsineSch kdT 5
#python $basic --add_param_groups_method all --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE_softmax_T --blocks_amount 3 --infwd_loss 1 --kd_T 5 --learning_rate_t 5e-3 --scheduler_method CosineAnnealingLR --pin_memory -t 7-09_03


#测试 不同的中间层特征提取方法
basic_inMethod='train_ours_13.py --model_s ResNet18_cifar_another --model_t ResNet34_cifar_another --path_t ./save/base/cifar100/ResNet34_cifar_another_base/ResNet34_cifar_another_teacher_best.pth.tar --method 1 --fusion_method Conv --feedback_time 0'

#AT_allAE + KLsoftmax TpartAll init0.1 decay_rate0.1 method1 Conv fwdloss1 三block ilrT5e-3 ConsineSch
python $basic_inMethod --in_method AT_allAE --add_param_groups_method all --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion KL_softmax --blocks_amount 3 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method CosineAnnealingLR --pin_memory -t 7-10_01

#CHMEAN_allAE + KLsoftmax TpartAll init0.1 decay_rate0.1 method1 Conv fwdloss1 三block ilrT5e-3 ConsineSch
python $basic_inMethod --in_method CHMEAN_allAE --add_param_groups_method all --CosineSch_Tmax 200 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion KL_softmax --blocks_amount 3 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method CosineAnnealingLR --pin_memory -t 7-10_02



