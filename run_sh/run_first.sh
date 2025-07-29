#!/bin/bash

# AT + MSE
python train_ours.py --model_s ResNet18_cifar_another --model_t ResNet34_cifar_another --path_t ./save/base/cifar100/ResNet34_cifar_another_base/ResNet34_cifar_another_teacher_best.pth.tar --feedback_time 500 --in_method Attention --in_criterion MSE -t first_AT_MSE &&
# AT + MSE_SOFTMAX
python train_ours.py --model_s ResNet18_cifar_another --model_t ResNet34_cifar_another --path_t ./save/base/cifar100/ResNet34_cifar_another_base/ResNet34_cifar_another_teacher_best.pth.tar --feedback_time 500 --in_method Attention --in_criterion MSE_softmax -t first_AT_MSEsoftmax &&
# AT + KL_SOFTMAX
python train_ours.py --model_s ResNet18_cifar_another --model_t ResNet34_cifar_another --path_t ./save/base/cifar100/ResNet34_cifar_another_base/ResNet34_cifar_another_teacher_best.pth.tar --feedback_time 500 --in_method Attention --in_criterion KL_softmax -t first_AT_KLsoftmax &&

# CH_MEAN + MSE
python train_ours.py --model_s ResNet18_cifar_another --model_t ResNet34_cifar_another --path_t ./save/base/cifar100/ResNet34_cifar_another_base/ResNet34_cifar_another_teacher_best.pth.tar --feedback_time 500 --in_method Channel_MEAN --in_criterion MSE -t first_CHMEAN_MSE &&
# CH_MEAN + MSE_SOFTMAX
python train_ours.py --model_s ResNet18_cifar_another --model_t ResNet34_cifar_another --path_t ./save/base/cifar100/ResNet34_cifar_another_base/ResNet34_cifar_another_teacher_best.pth.tar --feedback_time 500 --in_method Channel_MEAN --in_criterion MSE_softmax -t first_CHMEAN_MSEsoftmax &&
# CH_MEAN + KL_SOFTMAX
python train_ours.py --model_s ResNet18_cifar_another --model_t ResNet34_cifar_another --path_t ./save/base/cifar100/ResNet34_cifar_another_base/ResNet34_cifar_another_teacher_best.pth.tar --feedback_time 500 --in_method Channel_MEAN --in_criterion KL_softmax -t first_CHMEAN_KLsoftmax &&

# MEAN_STD + MSE
python train_ours.py --model_s ResNet18_cifar_another --model_t ResNet34_cifar_another --path_t ./save/base/cifar100/ResNet34_cifar_another_base/ResNet34_cifar_another_teacher_best.pth.tar --feedback_time 500 --in_method MEAN_STD --in_criterion MSE -t first_MEANSTD_MSE &&
# MEAN_STD + MSE_SOFTMAX
python train_ours.py --model_s ResNet18_cifar_another --model_t ResNet34_cifar_another --path_t ./save/base/cifar100/ResNet34_cifar_another_base/ResNet34_cifar_another_teacher_best.pth.tar --feedback_time 500 --in_method MEAN_STD --in_criterion MSE_softmax -t first_MEANSTD_MSEsoftmax &&
# MEAN_STD + KL_SOFTMAX
python train_ours.py --model_s ResNet18_cifar_another --model_t ResNet34_cifar_another --path_t ./save/base/cifar100/ResNet34_cifar_another_base/ResNet34_cifar_another_teacher_best.pth.tar --feedback_time 500 --in_method MEAN_STD --in_criterion KL_softmax -t first_MEANSTD_KLsoftmax &&

# SELF_AT_HW + MSE
python train_ours.py --model_s ResNet18_cifar_another --model_t ResNet34_cifar_another --path_t ./save/base/cifar100/ResNet34_cifar_another_base/ResNet34_cifar_another_teacher_best.pth.tar --feedback_time 500 --in_method SELF_AT_HW --in_criterion MSE -t first_SELFATHW_MSE &&
# SELF_AT_HW + MSE_SOFTMAX
python train_ours.py --model_s ResNet18_cifar_another --model_t ResNet34_cifar_another --path_t ./save/base/cifar100/ResNet34_cifar_another_base/ResNet34_cifar_another_teacher_best.pth.tar --feedback_time 500 --in_method SELF_AT_HW --in_criterion MSE_softmax -t first_SELFATHW_MSEsoftmax &&
# SELF_AT_HW + KL_SOFTMAX
python train_ours.py --model_s ResNet18_cifar_another --model_t ResNet34_cifar_another --path_t ./save/base/cifar100/ResNet34_cifar_another_base/ResNet34_cifar_another_teacher_best.pth.tar --feedback_time 500 --in_method SELF_AT_HW --in_criterion KL_softmax -t first_SELFATHW_KLsoftmax &&

# SELF_AT_C + MSE
python train_ours.py --model_s ResNet18_cifar_another --model_t ResNet34_cifar_another --path_t ./save/base/cifar100/ResNet34_cifar_another_base/ResNet34_cifar_another_teacher_best.pth.tar --feedback_time 500 --in_method SELF_AT_C --in_criterion MSE -t first_SELFATC_MSE &&
# SELF_AT_C + MSE_SOFTMAX
python train_ours.py --model_s ResNet18_cifar_another --model_t ResNet34_cifar_another --path_t ./save/base/cifar100/ResNet34_cifar_another_base/ResNet34_cifar_another_teacher_best.pth.tar --feedback_time 500 --in_method SELF_AT_C --in_criterion MSE_softmax -t first_SELFATC_MSEsoftmax &&
# SELF_AT_C + KL_SOFTMAX
python train_ours.py --model_s ResNet18_cifar_another --model_t ResNet34_cifar_another --path_t ./save/base/cifar100/ResNet34_cifar_another_base/ResNet34_cifar_another_teacher_best.pth.tar --feedback_time 500 --in_method SELF_AT_C --in_criterion KL_softmax -t first_SELFATC_KLsoftmax &&