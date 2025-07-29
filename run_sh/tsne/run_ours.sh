#!/bin/bash


#basic='plot_t-sne.py --distill test --model ResNet18_cifar_another'
#
## 当 seed 为 None 且 已经完成step1，想单独执行step2时。将step1的代码注释掉，使用 --classesList
## 如：
## python $basic --model_pth ./save/tsne/Our/19-01_34-T79.77-S80.10/ResNet18_cifar_another_best.pth.tar --seed None --classesList 5_27_2_16_71_91_58_97_24_80
#
## 随机
#python $basic --model_pth ./save/tsne/Our/19-01_29-T79.93-S79.83/ResNet18_cifar_another_best.pth.tar --seed None --classesList None &&
#
#python $basic --model_pth ./save/tsne/Our/19-01_30-T79.85-S79.50/ResNet18_cifar_another_best.pth.tar --seed None --classesList None &&
#
#python $basic --model_pth ./save/tsne/Our/19-01_30_repeat1-T79.81-S79.70/ResNet18_cifar_another_best.pth.tar --seed None --classesList None &&
#
#python $basic --model_pth ./save/tsne/Our/19-01_32-T80.10-S79.83/ResNet18_cifar_another_best.pth.tar --seed None --classesList None &&
#
#python $basic --model_pth ./save/tsne/Our/19-01_34-T79.77-S80.10/ResNet18_cifar_another_best.pth.tar --seed None --classesList None &&
#
#python $basic --model_pth ./save/tsne/Our/19-01_41-T80.07-S79.87/ResNet18_cifar_another_best.pth.tar --seed None --classesList None &&
#
#python $basic --model_pth ./save/tsne/Our/19-01_43_dist-T80.08-S80.08/ResNet18_cifar_another_best.pth.tar --seed None --classesList None



basic2='plot_t-sne.py --model ResNet18_cifar_another'

python $basic2 --distill test_best --model_pth ./save/tsne/Our/19-01_34-T79.77-S80.10/ResNet18_cifar_another_best.pth.tar --seed None --classesList None



