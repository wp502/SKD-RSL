#!/bin/bash


basic='plot_t-sne.py --model ResNet18_cifar_another'

# 随机
python $basic --distill test_best --model_pth ./save/tsne/SOKD/ResNet18_cifar_another_best.pth.tar --seed None --classesList None &&
python $basic --distill test_last --model_pth ./save/tsne/SOKD/ResNet18_cifar_another_last.pth.tar --seed None --classesList None








