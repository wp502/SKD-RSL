#!/bin/bash


basic='plot_t-sne.py --distill test --model ResNet18_cifar_another'

# 随机
python $basic --model_pth ./save/tsne/ReviewKD/ResNet18_cifar_another_best.pth.tar --seed None --classesList None








