#!/bin/bash

#TfKD需要跑他自己的代码，在TfKD文件夹中

# CIFAR100 SELF TfKD
#200 [60,120,160] lr0.1 decay0.1 batchsize128

# ResNet18_cifar_another - ResNet18_cifar_another
python main_tfkd_self.py --self_training --model_dir ./experiments/kd_experiments/ResNet18_cifar_another_distill/ResNet18_cifar_another_self_teacher/

#WRN_40_2 - WRN_40_2
python main_tfkd_self.py --self_training --model_dir ./experiments/kd_experiments/WRN_40_2_distill/WRN_40_2_self_teacher/

#VGG11_BN - VGG11_BN
python main_tfkd_self.py --self_training --model_dir ./experiments/kd_experiments/VGG11_BN_distill/VGG11_BN_self_teacher/
