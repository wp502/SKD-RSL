from .shufflenetv2 import shufflev2
from .resnet_another import resnet18 as resnet18_another
from .resnet_another import resnet34 as resnet34_another
from .resnet_another import resnet50 as resnet50_another
from .resnet_another import resnet101 as resnet101_another
from .resnet_cifar_another import resnet18 as resnet18_cifar_another
from .resnet_cifar_another import resnet34 as resnet34_cifar_another
from .resnet_cifar_another import resnet50 as resnet50_cifar_another
from .resnet_cifar_another import resnet101 as resnet101_cifar_another
import numpy as np
from .shufflenetv2_img import shufflev2 as shufflev2_img
from .wrn import WRN_16_2, WRN_28_10, WRN_40_2, WRN_28_4, WRN_28_2, WRN_40_1
from .wrn_img import WRN_28_4 as WRN_28_4_img
from .wrn_img import WRN_40_1 as WRN_40_1_img
from .wrn_img import WRN_40_2 as WRN_40_2_img
from .wrn_img import WRN_28_10 as WRN_28_10_img
from .mobilenetv2 import mobile_half as MobileNetV2
from .mobilenetv2_img import mobile_half as MobileNetV2_img
from .vgg import vgg13, vgg13_bn, vgg8, vgg8_bn, vgg11_bn
from .vgg_img import vgg11_bn as vgg11_bn_img
from .vgg_img import vgg8_bn as vgg8_bn_img
from .vgg_img import vgg13_bn as vgg13_bn_img

"""
    mobilenetv2_img.py: 训练输入图像大小为224*224的网络, 主要是tiny-imagenet、caltech256、food101数据集
    mobilenetv2.py: 训练输入图像大小为32*32的网络, 主要是cifar10、cifar100数据集
    resnet_another.py: 训练输入图像大小为224*224的网络, 主要是tiny-imagenet、caltech256、food101数据集
    resnet_cifar_another.py: 训练输入图像大小为32*32的网络, 主要是cifar10、cifar100数据集
    shufflenetv2_img.py: 训练输入图像大小为224*224的网络, 主要是tiny-imagenet、caltech256、food101数据集
    shufflenetv2.py: 训练输入图像大小为32*32的网络, 主要是cifar10、cifar100数据集
    vgg_img.py: 训练输入图像大小为224*224的网络, 主要是tiny-imagenet、caltech256、food101数据集
    vgg.py: 训练输入图像大小为32*32的网络, 主要是cifar10、cifar100数据集
    wrn_img.py: 训练输入图像大小为224*224的网络, 主要是tiny-imagenet、caltech256、food101数据集
    wrn.py: 训练输入图像大小为32*32的网络, 主要是cifar10、cifar100数据集
"""

model_dict = {
    'ResNet18_cifar_another': resnet18_cifar_another,
    'ResNet34_cifar_another': resnet34_cifar_another,
    'ResNet50_cifar_another': resnet50_cifar_another,
    'ResNet101_cifar_another': resnet101_cifar_another,
    'ResNet18_another': resnet18_another,
    'ResNet34_another': resnet34_another,
    'ResNet50_another': resnet50_another,
    'ResNet101_another': resnet101_another,
    'ShuffleV2': shufflev2,
    'ShuffleV2_img': shufflev2_img,
    'WRN_16_2': WRN_16_2,
    'WRN_28_10': WRN_28_10,
    'WRN_40_2': WRN_40_2,
    'WRN_40_1': WRN_40_1,
    'WRN_28_4': WRN_28_4,
    'WRN_28_2': WRN_28_2,
    'WRN_28_4_img': WRN_28_4_img,
    'WRN_40_1_img': WRN_40_1_img,
    'WRN_40_2_img': WRN_40_2_img,
    'WRN_28_10_img': WRN_28_10_img,
    'MobileNetV2': MobileNetV2,
    'MobileNetV2_img': MobileNetV2_img,
    'VGG8': vgg8,
    'VGG8_BN': vgg8_bn,
    'VGG11_BN': vgg11_bn,
    'VGG13': vgg13,
    'VGG13_BN': vgg13_bn,
    'VGG11_BN_IMG': vgg11_bn_img,
    'VGG8_BN_IMG': vgg8_bn_img,
    'VGG13_BN_IMG': vgg13_bn_img,
}
