#!/bin/bash

basic='train_ours_12.py --model_s ResNet18_cifar_another --model_t ResNet34_cifar_another --path_t ./save/base/cifar100/ResNet34_cifar_another_base/ResNet34_cifar_another_teacher_best.pth.tar --method 1 --in_method MEAN_STD --fusion_method Conv'
#--add_param_groups_method
#--infwd_loss
#--fusion_method
#--in_criterion
#-t

#测试 不同forward loss超参数值
##mean_std + KL(softmax) Tpart12 init0.1 decay_rate0.1 method1 Conv fwdloss10 三block
#python $basic --feedback_time 0 --add_param_groups_method 12 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion KL_softmax --blocks_amount 3 --infwd_loss 10 -t 5-03_fifth_MEANSTD_KLsoftmax_feedback_Tpart12_ilr0.1_dlr0.1_method1_Conv_threeBlock &&
#
##mean_std + KL(softmax) Tpart12 init0.1 decay_rate0.1 method1 Conv fwdloss2 三block
#python $basic --feedback_time 0 --add_param_groups_method 12 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion KL_softmax --blocks_amount 3 --infwd_loss 2 -t 5-04_fifth_MEANSTD_KLsoftmax_feedback_Tpart12_ilr0.1_dlr0.1_method1_Conv_threeBlock &&

##mean_std + KL(softmax) Tpart12 init0.1 decay_rate0.1 method1 Conv fwdloss3 三block
#python $basic --feedback_time 0 --add_param_groups_method 12 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion KL_softmax --blocks_amount 3 --infwd_loss 3 -t 5-13_fifth_MEANSTD_KLsoftmax_feedback_Tpart12_ilr0.1_dlr0.1_method1_Conv_fwdloss3_threeBlock &&
#
##mean_std + KL(softmax) Tpart12 init0.1 decay_rate0.1 method1 Conv fwdloss7 三block
#python $basic --feedback_time 0 --add_param_groups_method 12 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion KL_softmax --blocks_amount 3 --infwd_loss 7 -t 5-14_fifth_MEANSTD_KLsoftmax_feedback_Tpart12_ilr0.1_dlr0.1_method1_Conv_fwdloss7_threeBlock


#测试 loss的计算方式
##mean_std + MSE Tpart12 init0.1 decay_rate0.1 method1 Conv 三block
#python $basic --feedback_time 0 --add_param_groups_method 12 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE --blocks_amount 3 --infwd_loss 1 -t 5-05_fifth_MEANSTD_MSE_feedback_Tpart12_ilr0.1_dlr0.1_method1_Conv_threeBlock &&
#
##mean_std + MSE(normalize) Tpart12 init0.1 decay_rate0.1 method1 Conv 三block
#python $basic --feedback_time 0 --add_param_groups_method 12 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE_normalize --blocks_amount 3 --infwd_loss 1 -t 5-06_fifth_MEANSTD_MSEnormalize_feedback_Tpart12_ilr0.1_dlr0.1_method1_Conv_threeBlock &&
#
##mean_std + MSE(softmax) Tpart12 init0.1 decay_rate0.1 method1 Conv 三block
#python $basic --feedback_time 0 --add_param_groups_method 12 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE_softmax --blocks_amount 3 --infwd_loss 1 -t 5-07_fifth_MEANSTD_MSEsoftmax_feedback_Tpart12_ilr0.1_dlr0.1_method1_Conv_threeBlock

#测试 初始学习率
##mean_std + KL(softmax) Tpart12 init0.05 decay_rate0.1 method1 Conv 三block
#python $basic --feedback_time 0 --add_param_groups_method 12 --learning_rate 0.05 --lr_decay_rate 0.1 --in_criterion KL_softmax --blocks_amount 3 --infwd_loss 1 -t 5-08_fifth_MEANSTD_KLsoftmax_feedback_Tpart12_ilr0.05_dlr0.1_method1_Conv_threeBlock


#测试 Tpart 部分
##mean_std + KL(softmax) Tpart1 init0.1 decay_rate0.1 method1 Conv 三block
#python $basic --feedback_time 0 --add_param_groups_method 1 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion KL_softmax --blocks_amount 3 --infwd_loss 1 -t 5-09_fifth_MEANSTD_KLsoftmax_feedback_Tpart1_ilr0.1_dlr0.1_method1_Conv_threeBlock &&
#
##mean_std + KL(softmax) Tpart2 init0.1 decay_rate0.1 method1 Conv 三block
#python $basic --feedback_time 0 --add_param_groups_method 2 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion KL_softmax --blocks_amount 3 --infwd_loss 1 -t 5-10_fifth_MEANSTD_KLsoftmax_feedback_Tpart2_ilr0.1_dlr0.1_method1_Conv_threeBlock &&

##mean_std + KL(softmax) Tpart3 init0.1 decay_rate0.1 method1 Conv 三block
#python $basic --feedback_time 0 --add_param_groups_method 3 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion KL_softmax --blocks_amount 3 --infwd_loss 1 -t 5-15_fifth_MEANSTD_KLsoftmax_feedback_Tpart3_ilr0.1_dlr0.1_method1_Conv_threeBlock &&
#
##mean_std + KL(softmax) Tpart4 init0.1 decay_rate0.1 method1 Conv 三block
#python $basic --feedback_time 0 --add_param_groups_method 4 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion KL_softmax --blocks_amount 3 --infwd_loss 1 -t 5-15_fifth_MEANSTD_KLsoftmax_feedback_Tpart3_ilr0.1_dlr0.1_method1_Conv_threeBlock

#mean_std + KL(softmax) Tpart1 init0.1 decay_rate0.1 method1 Conv 四block
#python $basic --feedback_time 0 --add_param_groups_method 1 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion KL_softmax --blocks_amount 4 --infwd_loss 1 -t 5-22_fifth_MEANSTD_KLsoftmax_Tpart1_fourBlock &&

#mean_std + KL(softmax) Tpart5 init0.1 decay_rate0.1 method1 Conv 四block
#python $basic --feedback_time 0 --add_param_groups_method 5 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion KL_softmax --blocks_amount 4 --infwd_loss 1 -t 5-23_fifth_MEANSTD_KLsoftmax_Tpart5_fourBlock &&


#测试 不同数量的block
##mean_std + KL(softmax) Tpart12 init0.1 decay_rate0.1 method1 Conv 四block
#python $basic --feedback_time 0 --add_param_groups_method 12 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion KL_softmax --blocks_amount 4 --infwd_loss 1 -t 5-11_fifth_MEANSTD_KLsoftmax_feedback_Tpart12_ilr0.1_dlr0.1_method1_Conv_fourBlock &&
#
##mean_std + KL(softmax) Tpart12 init0.1 decay_rate0.1 method1 Conv 二block
#python $basic --feedback_time 0 --add_param_groups_method 12 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion KL_softmax --blocks_amount 2 --infwd_loss 1 -t 5-12_fifth_MEANSTD_KLsoftmax_feedback_Tpart12_ilr0.1_dlr0.1_method1_Conv_twoBlock


#测试 处理块时用不用normalize --in_useOther
#mean_std + KL(softmax) Tpart2 init0.1 decay_rate0.1 method1 Conv 三block interNormalize
#python $basic --feedback_time 0 --add_param_groups_method 2 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion KL_softmax --blocks_amount 3 --infwd_loss 1 --in_useOther -t 5-17_fifth_MEANSTD_KLsoftmax_feedback_Tpart2_ilr0.1_dlr0.1_method1_Conv_threeBlock_inNormalize

#测试 feedback time
##mean_std + KL(softmax) Tpart2 init0.1 decay_rate0.1 method1 Conv 三block feedback60
#python $basic --feedback_time 60 --add_param_groups_method 2 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion KL_softmax --blocks_amount 3 --infwd_loss 1 -t 5-18_fifth_MEANSTD_KLsoftmax_feedback60_Tpart2_ilr0.1_dlr0.1_method1_Conv_threeBlock &&
#
##mean_std + KL(softmax) Tpart2 init0.1 decay_rate0.1 method1 Conv 三block feedback120
#python $basic --feedback_time 120 --add_param_groups_method 2 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion KL_softmax --blocks_amount 3 --infwd_loss 1 -t 5-19_fifth_MEANSTD_KLsoftmax_feedback120_Tpart2_ilr0.1_dlr0.1_method1_Conv_threeBlock &&
#
##mean_std + KL(softmax) Tpart2 init0.1 decay_rate0.1 method1 Conv 三block feedback160
#python $basic --feedback_time 160 --add_param_groups_method 2 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion KL_softmax --blocks_amount 3 --infwd_loss 1 -t 5-20_fifth_MEANSTD_KLsoftmax_feedback160_Tpart2_ilr0.1_dlr0.1_method1_Conv_threeBlock &&
#
##mean_std + KL(softmax) Tpart2 init0.1 decay_rate0.1 method1 Conv 三block feedback300
#python $basic --feedback_time 300 --add_param_groups_method 2 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion KL_softmax --blocks_amount 3 --infwd_loss 1 -t 5-21_fifth_MEANSTD_KLsoftmax_feedback300_Tpart2_ilr0.1_dlr0.1_method1_Conv_threeBlock

#多种结果融合最终测试
#mean_std + MSE Tpart2 init0.1 decay_rate0.1 method1 Conv fwdloss1 二block
#python $basic --feedback_time 0 --add_param_groups_method 2 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE --blocks_amount 2 --infwd_loss 1 -t 5-09_01_fifth_MEANSTD_MSE_fwdloss1_twoBlock &&

#mean_std + MSE Tpart2 init0.1 decay_rate0.1 method1 Conv fwdloss5 二block
#python $basic --feedback_time 0 --add_param_groups_method 2 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE --blocks_amount 2 --infwd_loss 5 -t 5-09_02_fifth_MEANSTD_MSE_fwdloss5_twoBlock &&

#mean_std + MSE Tpart2 init0.1 decay_rate0.1 method1 Conv fwdloss5 二block interNormalize
#python $basic --feedback_time 0 --add_param_groups_method 2 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE --blocks_amount 2 --infwd_loss 5 --in_useOther -t 5-09_03_fifth_MEANSTD_MSE_fwdloss5_twoBlock_inNormalize &&

#mean_std + MSE Tpart2 init0.1 decay_rate0.1 method1 Conv fwdloss5 三block
#python $basic --feedback_time 0 --add_param_groups_method 2 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE --blocks_amount 3 --infwd_loss 5 -t 5-09_04_fifth_MEANSTD_MSE_fwdloss5_threeBlock &&

#mean_std + MSE Tpart2 init0.1 decay_rate0.1 method1 Conv fwdloss5 三block interNormalize
#python $basic --feedback_time 0 --add_param_groups_method 2 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE --blocks_amount 3 --infwd_loss 5 --in_useOther -t 5-09_05_fifth_MEANSTD_MSE_fwdloss5_threeBlock_inNormalize

#mean_std + MSE Tpart2 init0.1 decay_rate0.1 method1 Conv fwdloss5 四block
#python $basic --feedback_time 0 --add_param_groups_method 2 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE --blocks_amount 4 --infwd_loss 5 -t 5-09_06_fifth_MEANSTD_MSE_fwdloss5_fourBlock

#mean_std + KL(softmax) Tpart2 init0.1 decay_rate0.1 method1 Conv fwdloss5 三block
#python $basic --feedback_time 0 --add_param_groups_method 2 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion KL_softmax --blocks_amount 3 --infwd_loss 5 -t 5-09_07_fifth_MEANSTD_KLsoftmax_fwdloss5_threeBlock

#mean_std + MSE Tpart2 init0.1 decay_rate0.1 method1 Conv fwdloss1 三block
#python $basic --feedback_time 0 --add_param_groups_method 2 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE --blocks_amount 3 --infwd_loss 1 -t 5-09_08_fifth_MEANSTD_MSE_Tpart2_fwdloss1_threeBlock &&

#mean_std + MSE Tpart2 init0.1 decay_rate0.1 method1 Conv fwdloss3 三block
#python $basic --feedback_time 0 --add_param_groups_method 2 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE --blocks_amount 3 --infwd_loss 3 -t 5-09_09_fifth_MEANSTD_MSE_Tpart2_fwdloss3_threeBlock &&

#mean_std + MSE Tpart2 init0.1 decay_rate0.1 method1 Conv fwdloss7 三block
#python $basic --feedback_time 0 --add_param_groups_method 2 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE --blocks_amount 3 --infwd_loss 7 -t 5-09_10_fifth_MEANSTD_MSE_Tpart2_fwdloss7_threeBlock &&

#mean_std + MSE Tpart2 init0.1 decay_rate0.1 method1 Conv fwdloss10 三block
#python $basic --feedback_time 0 --add_param_groups_method 2 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE --blocks_amount 3 --infwd_loss 10 -t 5-09_11_fifth_MEANSTD_MSE_Tpart2_fwdloss10_threeBlock &&

#mean_std + MSE Tpart1 init0.1 decay_rate0.1 method1 Conv fwdloss5 三block
#python $basic --feedback_time 0 --add_param_groups_method 1 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE --blocks_amount 3 --infwd_loss 5 -t 5-09_12_fifth_MEANSTD_MSE_Tpart1_fwdloss5_threeBlock &&

#另类测试 使用method 2
#mean_std + KL(softmax) Tpart2 init0.1 decay_rate0.1 method2  Conv 三block
#python $basic --feedback_time 0 --add_param_groups_method 2 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion KL_softmax --blocks_amount 3 --infwd_loss 1 --fb_set_percent 0.4 -t 5-10_01_fifth_MEANSTD_KLsoftmax_method2_Conv_fbsetP0.4


#测试 教师模型的初始学习率
#mean_std + MSE Tpart1 init0.1 decay_rate0.1 method1 Conv 三block ilrT1e-2
#python $basic --feedback_time 0 --add_param_groups_method 1 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE --blocks_amount 3 --infwd_loss 1 --learning_rate_t 1e-2 -t 5-11_01_MEANSTD_MSE_Tpart1_threeBlock_ilrT1e-2 &&

#mean_std + MSE Tpart1 init0.1 decay_rate0.1 method1 Conv 三block ilrT1e-3
#python $basic --feedback_time 0 --add_param_groups_method 1 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE --blocks_amount 3 --infwd_loss 1 --learning_rate_t 1e-3 -t 5-11_02_MEANSTD_MSE_Tpart1_threeBlock_ilrT1e-3 &&

#mean_std + MSE Tpart1 init0.1 decay_rate0.1 method1 Conv 三block ilrT1e-4
#python $basic --feedback_time 0 --add_param_groups_method 1 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE --blocks_amount 3 --infwd_loss 1 --learning_rate_t 1e-4 -t 5-11_03_MEANSTD_MSE_Tpart1_threeBlock_ilrT1e-4 &&

#mean_std + MSE Tpart1 init0.1 decay_rate0.1 method1 Conv 三block ilrT1e-5
#python $basic --feedback_time 0 --add_param_groups_method 1 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE --blocks_amount 3 --infwd_loss 1 --learning_rate_t 1e-5 -t 5-11_04_MEANSTD_MSE_Tpart1_threeBlock_ilrT1e-5 &&

#mean_std + MSE Tpart1 init0.1 decay_rate0.1 method1 Conv 三block ilrT1e-6
#python $basic --feedback_time 0 --add_param_groups_method 1 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE --blocks_amount 3 --infwd_loss 1 --learning_rate_t 1e-6 -t 5-11_05_MEANSTD_MSE_Tpart1_threeBlock_ilrT1e-6

#mean_std + MSE Tpart2 init0.1 decay_rate0.1 method1 Conv 三block ilrT1e-2
#python $basic --feedback_time 0 --add_param_groups_method 2 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE --blocks_amount 3 --infwd_loss 1 --learning_rate_t 1e-2 -t 5-11_06_MEANSTD_MSE_Tpart2_threeBlock_ilrT1e-2 &&

#mean_std + MSE Tpart12 init0.1 decay_rate0.1 method1 Conv 三block ilrT1e-2
#python $basic --feedback_time 0 --add_param_groups_method 12 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE --blocks_amount 3 --infwd_loss 1 --learning_rate_t 1e-2 -t 5-11_07_MEANSTD_MSE_Tpart12_threeBlock_ilrT1e-2 &&

#mean_std + MSE Tpart1 init0.1 decay_rate0.1 method1 Conv 三block ilrT5e-2
#python $basic --feedback_time 0 --add_param_groups_method 1 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE --blocks_amount 3 --infwd_loss 1 --learning_rate_t 5e-2 -t 5-11_08_MEANSTD_MSE_Tpart1_threeBlock_ilrT5e-2 &&



#测试 COSINE Scheduler
#mean_std + MSE Tpart2 init0.1 decay_rate0.1 method1 Conv fwdloss5 三block ilrT1e-1 CosineSch
#python $basic --feedback_time 0 --add_param_groups_method 2 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE --learning_rate_t 1e-1 --scheduler_method CosineAnnealingLR --blocks_amount 3 --infwd_loss 5 -t 5-12_01_Tpart2_fwdloss5_threeBlock_CosineSch &&

#mean_std + MSE Tpart2 init0.1 decay_rate0.1 method1 Conv fwdloss1 三block ilrT1e-1 CosineSch
#python $basic --feedback_time 0 --add_param_groups_method 2 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE --learning_rate_t 1e-1 --scheduler_method CosineAnnealingLR --blocks_amount 3 --infwd_loss 1 -t 5-12_02_Tpart2_fwdloss1_threeBlock_CosineSch &&

#mean_std + MSE Tpart1 init0.1 decay_rate0.1 method1 Conv fwdloss1 三block ilrT1e-1 CosineSch
#python $basic --feedback_time 0 --add_param_groups_method 1 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE --learning_rate_t 1e-1 --scheduler_method CosineAnnealingLR --blocks_amount 3 --infwd_loss 1 -t 5-12_03_Tpart1_fwdloss1_threeBlock_CosineSch &&

#mean_std + MSE Tpart1 init0.05 decay_rate0.1 method1 Conv fwdloss1 三block ilrT1e-1 CosineSch
#python $basic --feedback_time 0 --add_param_groups_method 1 --learning_rate 0.05 --lr_decay_rate 0.1 --in_criterion MSE --learning_rate_t 1e-1 --scheduler_method CosineAnnealingLR --blocks_amount 3 --infwd_loss 1 -t 5-12_04_Tpart1_fwdloss1_threeBlock_CosineSch &&

#mean_std + MSE Tpart2 init0.1 decay_rate0.1 method1 Conv fwdloss5 三block ilrT1e-2 CosineSch
#python $basic --feedback_time 0 --add_param_groups_method 2 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE --learning_rate_t 1e-2 --scheduler_method CosineAnnealingLR --blocks_amount 3 --infwd_loss 5 -t 5-12_05_Tpart2_fwdloss5_ilrT1e-2_threeBlock_CosineSch

#mean_std + MSE Tpart2 init0.1 decay_rate0.1 method1 Conv fwdloss5 四block ilrT1e-2 CosineSch
python $basic --feedback_time 0 --add_param_groups_method 2 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE --learning_rate_t 1e-2 --scheduler_method CosineAnnealingLR --blocks_amount 4 --infwd_loss 5 -t 5-12_06_Tpart2_fwdloss5_ilrT1e-2_fourBlock_CosineSch &&

#mean_std + MSE Tpart1 init0.05 decay_rate0.1 method1 Conv fwdloss1 三block ilrT5e-2 CosineSch
python $basic --feedback_time 0 --add_param_groups_method 1 --learning_rate 0.05 --lr_decay_rate 0.1 --in_criterion MSE --learning_rate_t 5e-2 --scheduler_method CosineAnnealingLR --blocks_amount 3 --infwd_loss 1 -t 5-12_07_Tpart1_fwdloss1_ilrT5e-2_threeBlock_CosineSch &&

#mean_std + MSE Tpart1 init0.1 decay_rate0.1 method1 Conv fwdloss1 三block ilrT5e-2 CosineSch
python $basic --feedback_time 0 --add_param_groups_method 1 --learning_rate 0.1 --lr_decay_rate 0.1 --in_criterion MSE --learning_rate_t 5e-2 --scheduler_method CosineAnnealingLR --blocks_amount 3 --infwd_loss 1 -t 5-12_08_Tpart1_fwdloss1_ilrT5e-2_threeBlock_CosineSch
