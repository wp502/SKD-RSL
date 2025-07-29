# -*- coding: utf-8 -*-
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
from utils_ddp import is_main_process
import os
import numpy as np
import time
import logging
import copy

__all__ = ["Our_FWD_Conv_111", "Our_FB_Conv_111", "initAUXCFAndAE111"]

# Conv 的方法
class aeFusionConv2(nn.Module):
    """ 经过 inAuxCF 后的特征，经过 Conv2d AE """

    def __init__(self, in_channel, out_channel, fusion_dim, use_relu=False):
        super(aeFusionConv2, self).__init__()
        self.use_relu = use_relu
        # 提取特征
        self.en = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1,
                            padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.en_fc = nn.Linear(out_channel, fusion_dim)
        self.en_bn_fc = nn.BatchNorm1d(fusion_dim)
        self.relu = nn.ReLU(inplace=False)
        self.de_fc = nn.Linear(fusion_dim, out_channel)
        self.de_bn_fc = nn.BatchNorm1d(out_channel)
        # 改变通道大小
        self.de = nn.Conv2d(in_channels=out_channel, out_channels=in_channel, kernel_size=1, stride=1,
                            padding=0, bias=True)
        self.bn2 = nn.BatchNorm2d(in_channel)

    def forward(self, x):
        out = self.relu(self.bn1(self.en(x)))
        out = out.view(out.shape[0], -1)
        rep = self.relu(self.en_bn_fc(self.en_fc(out)))

        out = self.relu(self.de_bn_fc(self.de_fc(rep)))
        out = out.view(out.shape[0], out.shape[1], 1, 1)
        out = self.bn2(self.de(out))
        if self.use_relu:
            recon = self.relu(out)
        else:
            recon = out

        return rep, recon

class aeFusionConv23x3Single(nn.Module):
    """ 经过 inAuxCF 后的特征，经过 Single Conv2d AE """

    def __init__(self, in_channel, out_channel, fusion_dim, use_relu=False):
        super(aeFusionConv23x3Single, self).__init__()
        self.use_relu = use_relu
        self.en1 = nn.Conv2d(in_channels=in_channel, out_channels=fusion_dim, kernel_size=3, stride=1,
                            padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(fusion_dim)
        self.relu = nn.ReLU(inplace=False)
        self.de2 = nn.Conv2d(in_channels=fusion_dim, out_channels=in_channel, kernel_size=1, stride=1,
                             padding=0, bias=True)
        self.bn4 = nn.BatchNorm2d(in_channel)

    def forward(self, x):
        rep = self.relu(self.bn1(self.en1(x)))
        out = self.bn4(self.de2(rep))
        if self.use_relu:
            recon = self.relu(out)
        else:
            recon = out

        return rep, recon

class aeFusionConv2Single(nn.Module):
    """ 经过 inAuxCF 后的特征，经过 Single Conv2d AE """

    def __init__(self, in_channel, out_channel, fusion_dim, use_relu=False):
        super(aeFusionConv2Single, self).__init__()
        self.use_relu = use_relu
        self.en1 = nn.Conv2d(in_channels=in_channel, out_channels=fusion_dim, kernel_size=1, stride=1,
                            padding=0, bias=True)
        self.bn1 = nn.BatchNorm2d(fusion_dim)
        self.relu = nn.ReLU(inplace=False)
        self.de2 = nn.Conv2d(in_channels=fusion_dim, out_channels=in_channel, kernel_size=1, stride=1,
                             padding=0, bias=True)
        self.bn4 = nn.BatchNorm2d(in_channel)

    def forward(self, x):
        rep = self.relu(self.bn1(self.en1(x)))
        out = self.bn4(self.de2(rep))
        if self.use_relu:
            recon = self.relu(out)
        else:
            recon = out

        return rep, recon

class aeFusionLinear(nn.Module):
    """ 经过 inAuxCF 后的特征，经过 Linear AE """

    def __init__(self, in_channel, out_channel, fusion_dim, use_relu=False):
        super(aeFusionLinear, self).__init__()
        self.use_relu = use_relu
        self.en1 = nn.Linear(in_channel, fusion_dim, bias=True)
        self.bn1 = nn.BatchNorm1d(fusion_dim)
        self.relu = nn.ReLU(inplace=False)
        self.de2 = nn.Linear(fusion_dim, in_channel, bias=True)
        self.bn4 = nn.BatchNorm1d(in_channel)

    def forward(self, x):
        rep = self.relu(self.bn1(self.en1(x)))
        out = self.bn4(self.de2(rep))
        if self.use_relu:
            recon = self.relu(out)
        else:
            recon = out

        return rep, recon


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, use_relu=True):
        super(LinearBlock, self).__init__()
        self.use_relu = use_relu
        self.fc = nn.Linear(input_dim, output_dim, bias=True)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.fc(x)
        out = self.bn(out)
        if self.use_relu:
            return self.relu(out)
        else:
            return out

class Conv3x3Block(nn.Module):
    def __init__(self, input_dim, output_dim, use_relu=True):
        super(Conv3x3Block, self).__init__()
        self.use_relu = use_relu
        self.conv = nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1,
                            padding=1, bias=True)
        self.bn = nn.BatchNorm2d(output_dim)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        if self.use_relu:
            return self.relu(out)
        else:
            return out

class Conv1x1Block(nn.Module):
    def __init__(self, input_dim, output_dim, use_relu=True):
        super(Conv1x1Block, self).__init__()
        self.use_relu = use_relu
        self.conv = nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=1, stride=1,
                            padding=0, bias=True)
        self.bn = nn.BatchNorm2d(output_dim)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        if self.use_relu:
            return self.relu(out)
        else:
            return out

class adpAEFusionLinear(nn.Module):
    """ 经过 inAuxCF 后的特征，经过 自适应的 Linear AE """

    def __init__(self, in_channel, fusion_dim, input_list, use_relu=False):
        super(adpAEFusionLinear, self).__init__()
        self.use_relu = use_relu
        self.en = self.make_en(LinearBlock, fusion_dim, input_list, self.use_relu)
        self.de = self.make_de(LinearBlock, fusion_dim, input_list, self.use_relu)

    def make_en(self, block, fusion_dim, layers_list, use_relu):
        layers = []

        for i in range(1, len(layers_list)):
            layers.append(block(layers_list[i-1], layers_list[i], use_relu=True))
        layers.append(block(layers_list[-1], fusion_dim, use_relu=True))

        return nn.Sequential(*layers)

    def make_de(self, block, fusion_dim, layers_list, use_relu):
        layers_list = list(reversed(layers_list))
        layers = []

        layers.append(block(fusion_dim, layers_list[0], use_relu=True))

        for i in range(1, len(layers_list)):

            if i == len(layers_list) - 1:
                # 只有最后一层才考虑是否不需要relu
                layers.append(block(layers_list[i - 1], layers_list[i], use_relu))
            else:
                layers.append(block(layers_list[i - 1], layers_list[i], use_relu=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        rep = self.en(x)
        recon = self.de(rep)

        return rep, recon


class adpAEFusionConv3x3Linear(nn.Module):
    """ 经过 inAuxCF 后的特征，经过 自适应的 Linear AE """

    def __init__(self, in_channel, fusion_dim, input_list, use_relu=False):
        super(adpAEFusionConv3x3Linear, self).__init__()
        self.use_relu = use_relu
        self.en_first_conv3x3 = Conv3x3Block(in_channel, input_list[1], True)
        self.en = self.make_en(LinearBlock, fusion_dim, input_list, self.use_relu)
        self.de = self.make_de(LinearBlock, fusion_dim, input_list, self.use_relu)
        self.de_last_conv1x1 = Conv1x1Block(input_list[1], in_channel, self.use_relu)

    def make_en(self, block, fusion_dim, layers_list, use_relu):
        layers = []

        for i in range(2, len(layers_list)):
            # 当心 越界
            layers.append(block(layers_list[i-1], layers_list[i], use_relu=True))
        layers.append(block(layers_list[-1], fusion_dim, use_relu=True))

        return nn.Sequential(*layers)

    def make_de(self, block, fusion_dim, layers_list, use_relu):
        layers_list = list(reversed(layers_list))
        layers = []

        layers.append(block(fusion_dim, layers_list[0], use_relu=True))

        for i in range(1, len(layers_list)-1):
            # 当心 越界
            layers.append(block(layers_list[i - 1], layers_list[i], use_relu=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.en_first_conv3x3(x)
        out = out.view(out.shape[0], -1)
        rep = self.en(out)
        rep_de = self.de(rep)
        rep_de = rep_de.view(rep_de.shape[0], rep_de.shape[1], 1, 1)
        recon = self.de_last_conv1x1(rep_de)
        recon = recon.view(recon.shape[0], -1)

        return rep, recon

class adpAEFusionConv1x1Linear(nn.Module):
    """ 经过 inAuxCF 后的特征，经过 自适应的 Linear AE """

    def __init__(self, in_channel, fusion_dim, input_list, use_relu=False):
        super(adpAEFusionConv1x1Linear, self).__init__()
        self.use_relu = use_relu
        self.en_first_conv3x3 = Conv1x1Block(in_channel, input_list[1], True)
        self.en = self.make_en(LinearBlock, fusion_dim, input_list, self.use_relu)
        self.de = self.make_de(LinearBlock, fusion_dim, input_list, self.use_relu)
        self.de_last_conv1x1 = Conv1x1Block(input_list[1], in_channel, self.use_relu)

    def make_en(self, block, fusion_dim, layers_list, use_relu):
        layers = []

        for i in range(2, len(layers_list)):
            # 当心 越界
            layers.append(block(layers_list[i-1], layers_list[i], use_relu=True))
        layers.append(block(layers_list[-1], fusion_dim, use_relu=True))

        return nn.Sequential(*layers)

    def make_de(self, block, fusion_dim, layers_list, use_relu):
        layers_list = list(reversed(layers_list))
        layers = []

        layers.append(block(fusion_dim, layers_list[0], use_relu=True))

        for i in range(1, len(layers_list)-1):
            # 当心 越界
            layers.append(block(layers_list[i - 1], layers_list[i], use_relu=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.en_first_conv3x3(x)
        out = out.view(out.shape[0], -1)
        rep = self.en(out)
        rep_de = self.de(rep)
        rep_de = rep_de.view(rep_de.shape[0], rep_de.shape[1], 1, 1)
        recon = self.de_last_conv1x1(rep_de)
        recon = recon.view(recon.shape[0], -1)

        return rep, recon


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlockMy_133(nn.Module):
    def __init__(self, channel_in, channel_out, stride=2, downsample=None):
        super(BasicBlockMy_133, self).__init__()
        self.conv1 = conv1x1(channel_in, channel_out)
        self.bn1 = nn.BatchNorm2d(channel_out)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(channel_out, channel_out)
        self.bn2 = nn.BatchNorm2d(channel_out)
        self.conv3 = conv3x3(channel_out, channel_out, stride)
        self.bn3 = nn.BatchNorm2d(channel_out)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class BasicBlockMy_313_another(nn.Module):
    def __init__(self, channel_in, channel_out, stride=2, downsample=None):
        super(BasicBlockMy_313_another, self).__init__()
        self.conv1 = conv3x3(channel_in, channel_in)
        self.bn1 = nn.BatchNorm2d(channel_in)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv1x1(channel_in, channel_out)
        self.bn2 = nn.BatchNorm2d(channel_out)
        self.conv3 = conv3x3(channel_out, channel_out, stride)
        self.bn3 = nn.BatchNorm2d(channel_out)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class BasicBlockMy_313(nn.Module):
    def __init__(self, channel_in, channel_out, stride=2, downsample=None):
        super(BasicBlockMy_313, self).__init__()
        self.conv1 = conv3x3(channel_in, channel_out, stride)
        self.bn1 = nn.BatchNorm2d(channel_out)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv1x1(channel_out, channel_out)
        self.bn2 = nn.BatchNorm2d(channel_out)
        self.conv3 = conv3x3(channel_out, channel_out)
        self.bn3 = nn.BatchNorm2d(channel_out)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class BasicBlockMy(nn.Module):
    def __init__(self, channel_in, channel_out, stride=2, downsample=None):
        super(BasicBlockMy, self).__init__()
        self.conv1 = conv3x3(channel_in, channel_out, stride)
        self.bn1 = nn.BatchNorm2d(channel_out)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(channel_out, channel_out)
        self.bn2 = nn.BatchNorm2d(channel_out)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class BottleneckMy(nn.Module):
    def __init__(self, channel_in, channel_out, stride=2, downsample=None):
        super(BottleneckMy, self).__init__()
        self.conv1 = conv1x1(channel_in, channel_in)
        self.bn1 = nn.BatchNorm2d(channel_in)
        self.conv2 = conv3x3(channel_in, channel_in, stride)
        self.bn2 = nn.BatchNorm2d(channel_in)
        self.conv3 = conv1x1(channel_in, channel_out)
        self.bn3 = nn.BatchNorm2d(channel_out)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class inAuxCF(nn.Module):
    """ 用于每个中间层的辅助分类器 """

    def __init__(self, layers_list, poolSize=[1, 1], block_method='Bottle', num_classes=100):
        super(inAuxCF, self).__init__()
        assert len(layers_list) > 1
        self.block_method = block_method
        if 'big' in self.block_method:
            self.stride = 1
        else:
            self.stride = 2
        if 'Bottle' in self.block_method:
            block = BottleneckMy
        elif 'Basic' in self.block_method:
            if '133' in self.block_method:
                block = BasicBlockMy_133
            elif '313' in self.block_method:
                if 'another' in self.block_method:
                    block = BasicBlockMy_313_another
                else:
                    block = BasicBlockMy_313
            else:
                block = BasicBlockMy
        self.block_my = self.make_seq(block, self.stride, layers_list)
        self.avgpool = nn.AdaptiveAvgPool2d((poolSize[0], poolSize[1]))

        self.linear = nn.Linear(layers_list[-1] * poolSize[0] * poolSize[1], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_seq(self, block, stride, layers_list):
        layers = []

        for i in range(1, len(layers_list)):
            downsample_temp = nn.Sequential(
                conv1x1(layers_list[i-1], layers_list[i], stride=stride),
                nn.BatchNorm2d(layers_list[i]),
            )
            layers.append(block(layers_list[i-1], layers_list[i], stride=stride, downsample=downsample_temp))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.block_my(x)

        out = self.avgpool(x)
        feat = out
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return feat, out

# class inAuxCF_Stride122(nn.Module):
#     """ 用于每个中间层的辅助分类器 Stride 分别为 1 2 2 ，不区分 big 或 small"""
#
#     def __init__(self, layers_list, poolSize=[1, 1], block_method='Basic', num_classes=100):
#         super(inAuxCF_Stride122, self).__init__()
#         assert len(layers_list) > 1
#         self.block_method = block_method
#         self.stride = [1, 2, 2, 2]
#         if 'Bottle' in self.block_method:
#             block = BottleneckMy
#         elif 'Basic' in self.block_method:
#             block = BasicBlockMy
#         self.block_my = self.make_seq(block, self.stride, layers_list)
#         self.avgpool = nn.AdaptiveAvgPool2d((poolSize[0], poolSize[1]))
#
#         self.linear = nn.Linear(layers_list[-1] * poolSize[0] * poolSize[1], num_classes)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def make_seq(self, block, stride, layers_list):
#         layers = []
#
#         for i in range(1, len(layers_list)):
#             downsample_temp = nn.Sequential(
#                 conv1x1(layers_list[i-1], layers_list[i], stride=stride),
#                 nn.BatchNorm2d(layers_list[i]),
#             )
#             layers.append(block(layers_list[i-1], layers_list[i], stride=stride[i], downsample=downsample_temp))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.block_my(x)
#
#         out = self.avgpool(x)
#         feat = out
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#
#         return feat, out


class forLast(nn.Module):
    def __init__(self, in_channel, out_channel, poolSize, block_method='Basic'):
        super(forLast, self).__init__()
        self.block_method = block_method
        if 'big' in self.block_method:
            self.stride = 1
        else:
            self.stride = 2
        if 'Bottle' in self.block_method:
            block = BottleneckMy
        elif 'Basic' in self.block_method:
            block = BasicBlockMy
        self.layer1 = block(in_channel, out_channel, self.stride)
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d((poolSize[0], poolSize[1]))

    def forward(self, x):
        out = self.layer1(x)
        # out = self.relu(x)
        out = self.avgpool(out)

        return out

class forLastJustRelu(nn.Module):
    def __init__(self, poolSize):
        super(forLastJustRelu, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d((poolSize[0], poolSize[1]))

    def forward(self, x):
        out = self.relu(x)
        out = self.avgpool(out)

        return out

def initAUXCFAndAE111(args):
    if is_main_process():
        logger = logging.getLogger()
        logger.parent = None
        logger.info("T SELF method: " + str(args.self_method_t))
        logger.info("S SELF method: " + str(args.self_method_s))
    poolSize = [1, 1]
    args.auxCF_t = {}
    args.auxCF_s = {}
    block_method_t = args.aux_method_t
    block_method_s = args.aux_method_s
    if is_main_process():
        logger = logging.getLogger()
        logger.parent = None
        logger.info("T block method: " + str(block_method_t))
        logger.info("S block method: " + str(block_method_s))

    for i in range(args.blocks_amount_t - 1):
        layers_list_t = [args.t_dim[j][1] for j in range(i, args.blocks_amount_t)]
        args.auxCF_t.update({str(i + 1): inAuxCF(layers_list_t, poolSize, block_method_t, args.num_class).to(args.device)})
    args.auxCF_t.update({str(args.blocks_amount_t): nn.AdaptiveAvgPool2d((1, 1)).to(args.device)})
    for i in range(args.blocks_amount_s - 1):
        layers_list_s = [args.s_dim[j][1] for j in range(i, args.blocks_amount_s)]
        args.auxCF_s.update({str(i + 1): inAuxCF(layers_list_s, poolSize, block_method_s, args.num_class).to(args.device)})
    args.auxCF_s.update({str(args.blocks_amount_s): nn.AdaptiveAvgPool2d((1, 1)).to(args.device)})

    in_channel_t = args.t_dim[-1][1] * args.blocks_amount_t
    in_channel_s = args.s_dim[-1][1] * args.blocks_amount_s
    out_channel_t = in_channel_t // 2
    out_channel_s = in_channel_s // 2
    fusion_dim = sum([args.t_dim[-1][1], args.s_dim[-1][1]]) // 2
    if args.fusion_size == 'Mean':
        fusion_dim = int(fusion_dim)
    elif args.fusion_size == 'Big':
        fusion_dim = int(fusion_dim * 2)
    elif args.fusion_size == 'Small':
        fusion_dim = int(fusion_dim // 2)
    elif args.fusion_size == 'littleSmall':
        fusion_dim = int(fusion_dim // 1.5)
    elif args.fusion_size == 'largeSmall':
        fusion_dim = int(fusion_dim // 3)
    elif args.fusion_size == 'hugeSmall':
        fusion_dim = int(fusion_dim // 4)
    elif args.fusion_size == 'numClass':
        fusion_dim = int(args.num_class)
    elif args.fusion_size == 'ADP':
        min_dim = int(min(args.t_dim[-1][1], args.s_dim[-1][1]))
        fusion_dim = int(min_dim // 1.5)
    # # fusion size 最小 就是 num_classes
    # if fusion_dim < args.num_class:
    #     fusion_dim = int(args.num_class)

    fusion_method_t = args.fusion_method_AE_t
    fusion_method_s = args.fusion_method_AE_s
    use_relu = args.ende_use_relu
    if is_main_process():
        logger = logging.getLogger()
        logger.parent = None
        logger.info("Fusion Feature Method: " + str(args.fusion_size) + "\tFusion Feature Size: " + str(fusion_dim))
        logger.info("T fusion AE method: " + str(fusion_method_t))
        logger.info("S fusion AE method: " + str(fusion_method_s))
        logger.info("USE RELU: " + str(use_relu))

    ae_t = selectFusionMethod(in_channel_t, out_channel_t, fusion_dim, use_relu, fusion_method_t, args, is_T=True)
    ae_s = selectFusionMethod(in_channel_s, out_channel_s, fusion_dim, use_relu, fusion_method_s, args, is_T=False)
    args.ae_t = ae_t.to(args.device)
    args.ae_s = ae_s.to(args.device)


def selectFusionMethod(in_channel, out_channel, fusion_dim, use_relu, fusion_method, args, is_T=True):
    if fusion_method == 'AEConv':
        # Conv2d 1x1 Single
        ae = aeFusionConv2Single(in_channel, out_channel, fusion_dim, use_relu=use_relu).to(args.device)
    elif fusion_method == 'AEConv3x3Single':
        # Conv2d 3x3 Single
        ae = aeFusionConv23x3Single(in_channel, out_channel, fusion_dim, use_relu=use_relu).to(args.device)
    elif fusion_method == 'AEConv3x3Linear':
        # Conv2d 3x3 + Linear
        ae = aeFusionConv2(in_channel, out_channel, fusion_dim, use_relu=use_relu).to(args.device)
    elif fusion_method == 'AELinear':
        # Linear
        ae = aeFusionLinear(in_channel, out_channel, fusion_dim, use_relu=use_relu).to(args.device)
    elif 'ADP' in fusion_method:
        # Adaptive AE Linear
        input_dim = in_channel
        target_dim = fusion_dim
        input_list = []
        while input_dim > target_dim:
            input_list.append(input_dim)
            input_dim = input_dim // 2

        if is_T:
            str1 = "T Model: " + str(args.model_t) + ", AE_T Encoder Layers Number: " + str(len(input_list))
        else:
            str1 = "S Model: " + str(args.model_s) + ", AE_S Encoder Layers Number: " + str(len(input_list))

        if is_main_process():
            logger = logging.getLogger()
            logger.parent = None
            logger.info(str1)
        if fusion_method == 'ADPAELinear':
            ae = adpAEFusionLinear(in_channel, fusion_dim, input_list, use_relu=use_relu).to(args.device)
        elif fusion_method == 'ADPAEConv3x3Linear':
            ae = adpAEFusionConv3x3Linear(in_channel, fusion_dim, input_list, use_relu=use_relu).to(args.device)
        elif fusion_method == 'ADPAEConv1x1Linear':
            ae = adpAEFusionConv1x1Linear(in_channel, fusion_dim, input_list, use_relu=use_relu).to(args.device)

    return ae

class Our_FWD_Conv_111(nn.Module):
    """ 用于计算 forward 过程中的 中间特征层 的 loss """

    def __init__(self, T, args):
        super(Our_FWD_Conv_111, self).__init__()
        self.T = 1.0
        if 'KL' in args.in_criterion or args.in_criterion == 'MSE_softmax_T':
            self.T = T


    def forward(self, feat_t_fwd_list, feat_s_fwd_list, output_s_fwd, fwd_labels, criterion_cls, criterion_div, model_auxcfae_t, model_auxcfae_s, args):
        # model_auxcfae_t   # 0: ae, 1-last: auxcf
        # model_auxcfae_s   # 0: ae, 1-last: auxcf
        N, C, H, W = feat_t_fwd_list[0].shape
        t_feat_list = []
        # t_out_list = []
        s_feat_list = []
        s_out_list = []
        ae_t = model_auxcfae_t[0]
        ae_s = model_auxcfae_s[0]
        auxCF_t = model_auxcfae_t[1:]
        auxCF_s = model_auxcfae_s[1:]

        with torch.no_grad():
            for i in range(args.blocks_amount_t - 1):
                feat_t_temp, out_t_temp = auxCF_t[i](feat_t_fwd_list[i])
                t_feat_list.append(feat_t_temp)
                # t_out_list.append(out_t_temp)

        for i in range(args.blocks_amount_s - 1):
            feat_s_temp, out_s_temp = auxCF_s[i](feat_s_fwd_list[i])
            s_feat_list.append(feat_s_temp)
            s_out_list.append(out_s_temp)

        # t_out_list.append(output_t_fwd)
        s_out_list.append(output_s_fwd)
        t_feat_list.append(auxCF_t[-1](feat_t_fwd_list[-1]))
        s_feat_list.append(auxCF_s[-1](feat_s_fwd_list[-1]))
        t_ae_list = t_feat_list
        s_ae_list = s_feat_list

        s_reshape_list = [each.float() for each in s_out_list]

        loss_ce_all, loss_kl_all, loss_self_supervised_fwd = self_supervised_loss(fwd_labels, s_reshape_list,
                                                                                args.blocks_amount_s, criterion_cls,
                                                                                criterion_div, args.self_method_s, args)

        if args.fusion_method_AE_t == 'AELinear' or args.fusion_method_AE_t == 'ADPAELinear':
            t_ae_list = [each.reshape(each.shape[0], -1) for each in t_ae_list]
        if args.fusion_method_AE_s == 'AELinear' or args.fusion_method_AE_s == 'ADPAELinear':
            s_ae_list = [each.reshape(each.shape[0], -1) for each in s_ae_list]

        t_cat = torch.cat(t_ae_list, 1)
        s_cat = torch.cat(s_ae_list, 1)

        with torch.no_grad():
            t_fusion, t_recon = ae_t(t_cat)

        s_fusion, s_recon = ae_s(s_cat)

        s_recon_split = torch.split(s_recon, [args.s_dim[-1][1]]*args.blocks_amount_s, dim=1)
        loss_recon_fwd_s = 0.0

        loss_recon_fwd_ = [F.mse_loss(each_s, each_recon) for each_s, each_recon in zip(s_ae_list, s_recon_split)]
        loss_recon_fwd_s = sum(loss_recon_fwd_)

        t_fusion = t_fusion.reshape(t_fusion.shape[0], -1)
        s_fusion = s_fusion.reshape(s_fusion.shape[0], -1)

        # 教师和学生的 Fusion loss
        if args.NO_FUSION is True:
            loss_fusion_fwd = torch.Tensor([0.0]).to(args.device)
        else:
            if args.in_criterion == 'MSE':
                loss_fusion_fwd = (t_fusion - s_fusion).pow(2).mean()
                # loss_fusion_fwd = F.mse_loss(t_fusion, s_fusion)
            elif args.in_criterion == 'L1':
                # (t_fusion - s_fusion).abs().mean()
                loss_fusion_fwd = F.l1_loss(t_fusion, s_fusion)
            elif args.in_criterion == 'MSE_normalize':
                t_rep = F.normalize(t_fusion, dim=-1)
                s_fusion = F.normalize(s_fusion, dim=-1)
                loss_fusion_fwd = F.mse_loss(t_rep, s_fusion)
            elif args.in_criterion == 'MSE_softmax':
                loss_fusion_fwd = F.mse_loss(F.softmax(t_fusion, dim=-1), F.softmax(s_fusion, dim=-1))
            elif args.in_criterion == 'MSE_softmax_T':
                loss_fusion_fwd = F.mse_loss(F.softmax(t_fusion / self.T, dim=-1), F.softmax(s_fusion / self.T, dim=-1)) * (self.T * self.T)
            elif args.in_criterion == 'KL_softmax':
                # 维度结构不同，所用的KL loss的计算方式有所不同。
                assert len(t_fusion.shape) == 2
                loss_fusion_fwd = criterion_div(s_fusion, t_fusion)

        loss_chfwd_dict = dict(recon_S_fwd=loss_recon_fwd_s, fusion_fwd=loss_fusion_fwd, self_supervised_fwd=loss_self_supervised_fwd)
        return loss_chfwd_dict


class Our_FB_Conv_111(nn.Module):
    """ 用于计算 feedback 过程中的 中间特征层 的 loss """

    def __init__(self, T, args):
        super(Our_FB_Conv_111, self).__init__()
        self.T = 1.0
        if 'KL' in args.in_criterion or args.in_criterion == 'MSE_softmax_T':
            self.T = T

    def forward(self, feat_t_fb_list, feat_s_fb_list, output_t_fb, fb_labels, criterion_cls, criterion_div, loss_div_fb, model_t_weights, model_auxcfae_t, model_auxcfae_s, args):
        # model_auxcfae_t   # 0: ae, 1-last: auxcf
        # model_auxcfae_s   # 0: ae, 1-last: auxcf
        N, C, H, W = feat_t_fb_list[0].shape
        t_feat_list = []
        t_out_list = []
        s_feat_list = []

        ae_t = model_auxcfae_t[0]
        ae_s = model_auxcfae_s[0]
        auxCF_t = model_auxcfae_t[1:]
        auxCF_s = model_auxcfae_s[1:]

        for i in range(args.blocks_amount_t - 1):
            feat_t_temp, out_t_temp = auxCF_t[i](feat_t_fb_list[i])
            t_feat_list.append(feat_t_temp)
            t_out_list.append(out_t_temp)

        with torch.no_grad():
            for i in range(args.blocks_amount_s - 1):
                feat_s_temp, out_s_temp = auxCF_s[i](feat_s_fb_list[i])
                s_feat_list.append(feat_s_temp)
                # s_out_list.append(out_s_temp)

        t_out_list.append(output_t_fb)
        # s_out_list.append(output_s_fb)
        t_feat_list.append(auxCF_t[-1](feat_t_fb_list[-1]))
        s_feat_list.append(auxCF_s[-1](feat_s_fb_list[-1]))
        t_ae_list = t_feat_list
        s_ae_list = s_feat_list

        t_reshape_list = [each.float() for each in t_out_list]

        loss_ce_all, loss_kl_all, loss_self_supervised_fb = self_supervised_loss(fb_labels, t_reshape_list,
                                                                                args.blocks_amount_t, criterion_cls,
                                                                                criterion_div, args.self_method_t, args)

        if args.fusion_method_AE_t == 'AELinear' or args.fusion_method_AE_t == 'ADPAELinear':
            t_ae_list = [each.reshape(each.shape[0], -1) for each in t_ae_list]
        if args.fusion_method_AE_s == 'AELinear' or args.fusion_method_AE_s == 'ADPAELinear':
            s_ae_list = [each.reshape(each.shape[0], -1) for each in s_ae_list]

        t_cat = torch.cat(t_ae_list, 1)
        s_cat = torch.cat(s_ae_list, 1)

        t_fusion, t_recon = ae_t(t_cat)

        with torch.no_grad():
            s_fusion, s_recon = ae_s(s_cat)

        t_recon_split = torch.split(t_recon, [args.t_dim[-1][1]]*args.blocks_amount_t, dim=1)

        loss_recon_fb_t = 0.0
        loss_recon_fb_ = [F.mse_loss(each_t, each_recon) for each_t, each_recon in zip(t_ae_list, t_recon_split)]
        loss_recon_fb_t = sum(loss_recon_fb_)

        t_fusion = t_fusion.reshape(t_fusion.shape[0], -1)
        s_fusion = s_fusion.reshape(s_fusion.shape[0], -1)

        # 教师和学生的 Fusion loss
        if args.NO_FUSION is True:
            loss_fusion_fb = torch.Tensor([0.0]).to(args.device)
        else:
            if args.in_criterion == 'MSE':
                loss_fusion_fb = (t_fusion - s_fusion).pow(2).mean()
                # loss_fusion_fb = F.mse_loss(t_fusion, s_fusion)
            elif args.in_criterion == 'L1':
                # (t_fusion - s_fusion).abs().mean()
                loss_fusion_fb = F.l1_loss(t_fusion, s_fusion)
            elif args.in_criterion == 'MSE_normalize':
                t_rep = F.normalize(t_fusion, dim=-1)
                s_fusion = F.normalize(s_fusion, dim=-1)
                loss_fusion_fb = F.mse_loss(t_rep, s_fusion)
            elif args.in_criterion == 'MSE_softmax':
                loss_fusion_fb = F.mse_loss(F.softmax(t_fusion, dim=-1), F.softmax(s_fusion, dim=-1))
            elif args.in_criterion == 'MSE_softmax_T':
                loss_fusion_fb = F.mse_loss(F.softmax(t_fusion / self.T, dim=-1), F.softmax(s_fusion / self.T, dim=-1)) * (
                            self.T * self.T)
            elif args.in_criterion == 'KL_softmax':
                # 维度结构不同，所用的KL loss的计算方式有所不同。
                assert len(t_fusion.shape) == 2
                loss_fusion_fb = criterion_div(t_fusion, s_fusion)


        if args.fbUseGradSim is True:
            # 将 损失 做梯度相似性比较。
            loss_fb_kl = loss_fusion_fb + loss_div_fb
            # cls_grad = torch.autograd.grad([loss_ce_all, loss_kl_all], model_t_weights, allow_unused=True, retain_graph=True)

            cls_grad = torch.autograd.grad(loss_ce_all, model_t_weights, allow_unused=True, retain_graph=True)
            div_kl_grad = torch.autograd.grad(loss_fb_kl, model_t_weights, allow_unused=True, retain_graph=True)
            cos_similarity_feedback = cal_each_grad_sim3(cls_grad, div_kl_grad, args)
            feedback_coe = 1.0
            if cos_similarity_feedback < 0:
                feedback_coe = 0.0

            if args.NO_SELF is False:
                kl_all_grad = torch.autograd.grad(loss_kl_all, model_t_weights, allow_unused=True, retain_graph=True)
                cos_similarity_self = cal_each_grad_sim3(cls_grad, kl_all_grad, args)
                self_kl_all_coe = 1.0
                if cos_similarity_self < 0:
                    self_kl_all_coe = 0.0
            else:
                self_kl_all_coe = 0.0

            # loss_fb_other = loss_ce_all + self_kl_all_coe * loss_kl_all + fusion_coe * loss_fusion_fb + div_coe * loss_div_fb
            loss_fb_other = loss_ce_all + self_kl_all_coe * loss_kl_all + feedback_coe * loss_fb_kl
            # if cos_similarity_temp < 0:
            #     loss_self_supervised_fb = loss_ce_all
            # else:
            #     loss_self_supervised_fb = loss_ce_all + loss_kl_all
        else:
            loss_fb_kl = loss_fusion_fb + loss_div_fb
            # loss_fb_other = loss_ce_all + loss_kl_all + loss_fusion_fb + loss_div_fb
            loss_fb_other = loss_ce_all + loss_kl_all + loss_fb_kl

        # loss_chfb = loss_recon_fb + loss_fusion_fb
        loss_chfb_dict = dict(recon_T_fb=loss_recon_fb_t, loss_other=loss_fb_other)
        return loss_chfb_dict

def self_supervised_loss(labels, reshape_list, block_amounts, criterion_cls, criterion_div, self_method, args):
    '''以三个为例
            CE 1
            KL 2 1
            KL 3 1
            CE 2
            KL 1 2
            KL 3 2
            CE 3
            KL 1 3
            KL 2 3'''

    loss_ce_ = []
    loss_deep_shallow_ = []
    loss_shallow_deep_ = []
    if args.NO_SELF is True:
        # 如果不考虑 self distillation的情况下，只保留了CE loss，以保证AUXCF的更新

        for s_1 in reshape_list:
            temp_ce = criterion_cls(s_1, labels)
            loss_ce_.append(temp_ce)

        for i in range(block_amounts - 1):
            # deep - shallow
            deep_sender_idx = block_amounts - 1 - i
            # deep_sender = s_reshape_list[deep_sender_idx]
            deep_receiver = reshape_list[:deep_sender_idx]
            for each in deep_receiver:
                temp_kl_deep = torch.Tensor([0.0]).to(args.device)
                loss_deep_shallow_.append(temp_kl_deep)
            # shallow - deep
            shallow_sender_idx = i
            # shallow_sender = s_reshape_list[shallow_sender_idx]
            shallow_receiver = reshape_list[shallow_sender_idx + 1:]
            for each in shallow_receiver:
                temp_kl_shallow = torch.Tensor([0.0]).to(args.device)
                loss_shallow_deep_.append(temp_kl_shallow)
    else:

        for s_1 in reshape_list:
            temp_ce = criterion_cls(s_1, labels)
            loss_ce_.append(temp_ce)

        for i in range(block_amounts - 1):
            # deep - shallow
            deep_sender_idx = block_amounts - 1 - i
            deep_sender = reshape_list[deep_sender_idx]
            deep_receiver = reshape_list[:deep_sender_idx]
            for each in deep_receiver:
                temp_kl_deep = criterion_div(each, deep_sender)
                loss_deep_shallow_.append(temp_kl_deep)
            # shallow - deep
            shallow_sender_idx = i
            shallow_sender = reshape_list[shallow_sender_idx]
            shallow_receiver = reshape_list[shallow_sender_idx + 1:]
            for each in shallow_receiver:
                temp_kl_shallow = criterion_div(each, shallow_sender)
                loss_shallow_deep_.append(temp_kl_shallow)

    loss_ce_all = sum(loss_ce_)

    if 'deep_shallow' in self_method:
        if 'single' in self_method:
            loss_kl_all = sum(loss_deep_shallow_[:(block_amounts-1)])
        else:
            loss_kl_all = sum(loss_deep_shallow_)
    elif 'shallow_deep' in self_method:
        if 'single' in self_method:
            loss_kl_all = sum(loss_shallow_deep_[:(block_amounts-1)])
        else:
            loss_kl_all = sum(loss_shallow_deep_)
    elif 'bi_directional' in self_method:
        loss_kl_all = sum(loss_deep_shallow_) + sum(loss_shallow_deep_)

    loss_self_supervised = loss_ce_all + loss_kl_all

    return loss_ce_all, loss_kl_all, loss_self_supervised

def cal_each_grad_sim(loss_cls, loss_other, model_t_weights, args):
    cls_grad = torch.autograd.grad(loss_cls, model_t_weights, allow_unused=True, retain_graph=True)
    other_grad = torch.autograd.grad(loss_other, model_t_weights, allow_unused=True, retain_graph=True)

    nontype_idx_cls = len(cls_grad)
    nontype_idx_oth = len(other_grad)
    for idx, each in enumerate(cls_grad):
        if each is None:
            nontype_idx_cls = idx
            break

    for idx, each in enumerate(other_grad):
        if each is None:
            nontype_idx_oth = idx
            break

    split_idx = min(nontype_idx_cls, nontype_idx_oth)

    cosin_simility_all = torch.mean(torch.Tensor(
        [F.cosine_similarity(each_cls.reshape(-1), each_other.reshape(-1), dim=0) for each_cls, each_other in
         zip(cls_grad[:split_idx], other_grad[:split_idx])]).to(args.device))


    return cosin_simility_all

def cal_each_grad_sim2(cls_grad, other_grad, args):


    nontype_idx_cls = len(cls_grad)
    nontype_idx_oth = len(other_grad)
    for idx, each in enumerate(cls_grad):
        if each is None:
            nontype_idx_cls = idx
            break

    for idx, each in enumerate(other_grad):
        if each is None:
            nontype_idx_oth = idx
            break

    split_idx = min(nontype_idx_cls, nontype_idx_oth)

    cosin_simility_all = torch.mean(torch.Tensor(
        [F.cosine_similarity(each_cls.reshape(-1), each_other.reshape(-1), dim=0) for each_cls, each_other in
         zip(cls_grad[:split_idx], other_grad[:split_idx])]).to(args.device))

    return cosin_simility_all

def cal_each_grad_sim3(cls_grad, other_grad, args):

    nontype_idx_cls = len(cls_grad)
    nontype_idx_oth = len(other_grad)
    for idx, each in enumerate(cls_grad):
        if each is None:
            nontype_idx_cls = idx
            break

    for idx, each in enumerate(other_grad):
        if each is None:
            nontype_idx_oth = idx
            break

    split_idx = min(nontype_idx_cls, nontype_idx_oth)

    cls_grad = torch.cat([each.reshape(-1) for each in cls_grad[:split_idx]], -1)
    other_grad = torch.cat([each.reshape(-1) for each in other_grad[:split_idx]], -1)
    # cls_grad = torch.cat([each.reshape(-1) for each in cls_grad], -1).to(args.device)
    # other_grad = torch.cat([each.reshape(-1) for each in other_grad], -1).to(args.device)

    cosin_simility_all = F.cosine_similarity(cls_grad, other_grad, dim=0)


    return cosin_simility_all

def main():

    # 测试Adaptive AE Fusion Conv1
    # adpaec1 = adpAEFusionConv1x1Linear(in_channel=2048, fusion_dim=320, input_list=[2048, 1024, 512],
    #                                    use_relu=False).cuda()
    # total_aec1 = sum([param.nelement() for param in adpaec1.parameters()])
    # print("Parmeter: %.2fM" % (total_aec1 / 1e6))
    # torchinfo.summary(adpaec1, (1, 2048, 1, 1))
    # ac1 = torch.randn(10, 2048, 1, 1).cuda()
    # repc1, outc1 = adpaec1(ac1)
    # print(repc1.shape, outc1.shape)
    #
    # # 测试Adaptive AE Fusion Conv3
    # adpaec3 = adpAEFusionConv3x3Linear(in_channel=2048, fusion_dim=320, input_list=[2048, 1024, 512], use_relu=False).cuda()
    # total_aec3 = sum([param.nelement() for param in adpaec3.parameters()])
    # print("Parmeter: %.2fM" % (total_aec3 / 1e6))
    # torchinfo.summary(adpaec3, (1, 2048, 1, 1))
    # ac3 = torch.randn(10, 2048, 1, 1).cuda()
    # repc3, outc3 = adpaec3(ac3)
    # print(repc3.shape, outc3.shape)
    #
    # # 测试Adaptive AE Fusion
    # adpae = adpAEFusionLinear(in_channel=2048, fusion_dim=320, input_list=[2048, 1024, 512], use_relu=False).cuda()
    # total_ae1 = sum([param.nelement() for param in adpae.parameters()])
    # print("Parmeter: %.2fM" % (total_ae1 / 1e6))
    # torchinfo.summary(adpae, (1, 2048))
    # a = torch.randn(10, 2048).cuda()
    # rep, out = adpae(a)
    # print(rep.shape, out.shape)
    # print("123")

    # 测试 AUXCF
    layers_list_t = [64, 128, 256, 512]
    bottle_net = inAuxCF(layers_list_t, [1, 1], 'Bottle', 100)
    basic_net = inAuxCF(layers_list_t, [1, 1], 'Basic', 100)

    bottle_single = BottleneckMy(64, 128)
    basic_single = BasicBlockMy(64, 128)

    import utils
    print("Bottle")
    utils.statis_params_amount(bottle_net)
    print("Basic")
    utils.statis_params_amount(basic_net)


if __name__ == '__main__':
    main()