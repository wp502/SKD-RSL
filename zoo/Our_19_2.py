# -*- coding: utf-8 -*-
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

__all__ = ["Our_FWD_Conv_19_2", "Our_FB_Conv_19_2", "initAUXCFAndAE19_2"]

# Conv 的方法

class ConvReg(nn.Module):
    """Convolutional regression for FitNet"""

    def __init__(self, s_shape, t_shape, use_relu=True):
        super(ConvReg, self).__init__()
        self.use_relu = use_relu
        s_N, s_C, s_H, s_W = s_shape
        t_N, t_C, t_H, t_W = t_shape
        if s_H == 2 * t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=3, stride=2, padding=1)
        elif s_H * 2 == t_H:
            self.conv = nn.ConvTranspose2d(s_C, t_C, kernel_size=4, stride=2, padding=1)
        elif s_H >= t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=(1 + s_H - t_H, 1 + s_W - t_W))
        else:
            raise NotImplemented('student size {}, teacher size {}'.format(s_H, t_H))
        self.bn = nn.BatchNorm2d(t_C)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        if self.use_relu:
            return self.relu(self.bn(x))
        else:
            return self.bn(x)

class AEFusionSender(nn.Module):
    """ 发送知识的模型，经过卷积自编码器融合知识。在forward中，教师为发送者，在feedback中，学生为发送者。适用于 MEAN_STD和 Channel_MEAN """

    def __init__(self, sender_channel, receiver_channel, use_relu=True):
        super(AEFusionSender, self).__init__()
        self.use_relu = use_relu
        self.en = nn.Conv1d(in_channels=sender_channel, out_channels=receiver_channel, kernel_size=1, stride=1, padding=0,
                            bias=True)
        self.bn1 = nn.BatchNorm1d(receiver_channel)
        self.relu = nn.ReLU(inplace=False)
        self.de = nn.Conv1d(in_channels=receiver_channel, out_channels=sender_channel, kernel_size=1, stride=1, padding=0,
                            bias=True)
        self.bn2 = nn.BatchNorm1d(sender_channel)

    def forward(self, x):
        x_en = self.bn1(self.en(x))
        if self.use_relu:
            rep = self.relu(x_en)
        else:
            rep = x_en

        x_de = self.bn2(self.de(rep))
        if self.use_relu:
            recon = self.relu(x_de)
        else:
            recon = x_de
        return rep, recon

class AEFusionReceiver(nn.Module):
    """ 接收知识的模型，经过卷积自编码器融合知识。在forward中，学生为接收者，在feedback中，教师为接收者。适用于 MEAN_STD和 Channel_MEAN """

    def __init__(self, receiver_channel, use_relu=True):
        super(AEFusionReceiver, self).__init__()
        self.use_relu = use_relu
        self.en = nn.Conv1d(in_channels=receiver_channel, out_channels=receiver_channel, kernel_size=1, stride=1, padding=0,
                            bias=True)
        self.bn1 = nn.BatchNorm1d(receiver_channel)
        self.relu = nn.ReLU(inplace=False)
        self.de = nn.Conv1d(in_channels=receiver_channel, out_channels=receiver_channel, kernel_size=1, stride=1, padding=0,
                            bias=True)
        self.bn2 = nn.BatchNorm1d(receiver_channel)

    def forward(self, x):
        x_en = self.bn1(self.en(x))
        if self.use_relu:
            rep = self.relu(x_en)
        else:
            rep = x_en

        x_de = self.bn2(self.de(rep))
        if self.use_relu:
            recon = self.relu(x_de)
        else:
            recon = x_de
        return rep, recon


# class aeFusionLinear(nn.Module):
#     """ 经过 inAuxCF 后的特征，经过 Linear AE """
#
#     def __init__(self, in_dim, fusion_dim, use_relu=False):
#         super(aeFusionLinear, self).__init__()
#         self.use_relu = use_relu
#         self.en = nn.Linear(in_dim, fusion_dim)
#         self.bn1 = nn.BatchNorm1d(fusion_dim)
#         self.de = nn.Linear(fusion_dim, in_dim)
#         self.bn2 = nn.BatchNorm1d(in_dim)
#         self.relu = nn.ReLU(inplace=False)
#
#     def forward(self, x):
#         fusion = self.relu(self.bn1(self.en(x)))
#         recon = self.bn2(self.de(fusion))
#         if self.use_relu:
#             recon = self.relu(recon)
#
#         return fusion, recon
#
# class aeFusionConv(nn.Module):
#     """ 经过 inAuxCF 后的特征，经过 Conv AE """
#
#     def __init__(self, in_dim, fusion_dim, use_relu=False):
#         super(aeFusionConv, self).__init__()
#         self.use_relu = use_relu
#         self.en = nn.Conv1d(in_channels=in_dim, out_channels=fusion_dim, kernel_size=1, stride=1,
#                             padding=0, bias=True)
#         self.bn1 = nn.BatchNorm1d(fusion_dim)
#         self.relu = nn.ReLU(inplace=False)
#         self.de = nn.Conv1d(in_channels=fusion_dim, out_channels=in_dim, kernel_size=1, stride=1,
#                             padding=0, bias=True)
#         self.bn2 = nn.BatchNorm1d(in_dim)
#
#     def forward(self, x):
#         rep = self.relu(self.bn1(self.en(x)))
#
#         x_de = self.bn2(self.de(rep))
#         if self.use_relu:
#             recon = self.relu(x_de)
#         else:
#             recon = x_de
#         return rep, recon


class inAuxCF(nn.Module):
    """ 用于每个中间层的辅助分类器 """

    def __init__(self, planes, block_amount=1, num_classes=100):
        super(inAuxCF, self).__init__()
        self.block_amount = block_amount
        self.conv1 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1,
                               stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(planes, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        feat = out
        out = self.linear(out)

        return feat, out


class BasicBlockMy(nn.Module):
    def __init__(self, planes):
        super(BasicBlockMy, self).__init__()
        self.conv1 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)

        return out

class inAuxCF2(nn.Module):
    """ 用于每个中间层的辅助分类器 """

    def __init__(self, planes, block_amount=1, num_classes=100):
        super(inAuxCF2, self).__init__()
        self.block_amount = block_amount
        self.basic_block_my = self.make_seq(BasicBlockMy, planes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(planes, num_classes)

    def make_seq(self, block, planes):
        layers = []
        for _ in range(self.block_amount):
            layers.append(block(planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.basic_block_my(x)

        out = self.avgpool(x)
        out = out.view(out.size(0), -1)
        feat = out
        out = self.linear(out)

        return feat, out

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


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BottleneckMy(nn.Module):
    def __init__(self, channel_in, channel_out, downsample=None):
        super(BottleneckMy, self).__init__()
        self.conv1 = conv1x1(channel_in, channel_in)
        self.bn1 = nn.BatchNorm2d(channel_in)
        self.conv2 = conv3x3(channel_in, channel_in)
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

class inAuxCF3(nn.Module):
    """ 用于每个中间层的辅助分类器 """

    def __init__(self, layers_list, poolSize=[1, 1], num_classes=100):
        super(inAuxCF3, self).__init__()
        assert len(layers_list) > 1
        self.block_my = self.make_seq(BottleneckMy, layers_list)
        self.avgpool = nn.AdaptiveAvgPool2d((poolSize[0], poolSize[1]))

        self.linear = nn.Linear(layers_list[-1] * poolSize[0] * poolSize[1], num_classes)



    def make_seq(self, block, layers_list):
        layers = []

        for i in range(1, len(layers_list)):
            downsample_temp = nn.Sequential(
                conv1x1(layers_list[i-1], layers_list[i]),
                nn.BatchNorm2d(layers_list[i]),
            )
            layers.append(block(layers_list[i-1], layers_list[i], downsample_temp))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.block_my(x)

        out = self.avgpool(x)
        feat = out
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return feat, out

def mean_std(x, args):
    B, C, H, W = x.shape
    if args.in_useOther:
        mean_x = F.normalize(x.reshape(B, C, -1).mean(2)).reshape(B, C, 1)
        std_x = F.normalize(x.reshape(B, C, -1).std(2)).reshape(B, C, 1)
    else:
        mean_x = x.reshape(B, C, -1).mean(2).reshape(B, C, 1)
        std_x = x.reshape(B, C, -1).std(2).reshape(B, C, 1)
    mean_std_x = torch.cat([mean_x, std_x], -1)
    return mean_std_x

def channel_mean(x, args):
    return x.reshape(x.shape[0], x.shape[1], -1).mean(2).reshape(x.shape[0], x.shape[1], 1)

def at(x, args):
    B, C, H, W = x.shape
    if args.in_useOther:
        # return F.normalize(x.pow(2).mean(1).view(x.shape[0], -1))
        return F.normalize(x.pow(2).mean(1).reshape(B, H*W, 1))
    else:
        # return x.pow(2).mean(1).view(x.shape[0], -1)
        return x.pow(2).mean(1).reshape(B, H*W, 1)
    # return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

class forLast(nn.Module):
    def __init__(self, in_channel, out_channel, poolSize):
        super(forLast, self).__init__()
        self.layer1 = BottleneckMy(in_channel, out_channel)
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d((poolSize[0], poolSize[1]))

    def forward(self, x):
        out = self.layer1(x)
        # out = self.relu(x)
        out = self.avgpool(out)

        return out

def initAUXCFAndAE19_2(args):
    poolSize = [1, 1]
    args.auxCF_t = {}
    args.auxCF_s = {}
    for i in range(args.blocks_amount - 1):
        layers_list_t = [args.t_dim[j][1] for j in range(i, args.blocks_amount)]
        args.auxCF_t.update({str(i + 1): inAuxCF3(layers_list_t, poolSize, args.num_class).cuda()})
    args.auxCF_t.update({str(args.blocks_amount): nn.AdaptiveAvgPool2d((1, 1)).cuda()})
    # args.auxCF_t.update({str(args.blocks_amount): forLast(args.t_dim[-1][1], args.t_dim[-1][1], poolSize).cuda()})
    for i in range(args.blocks_amount - 1):
        layers_list_s = [args.s_dim[j][1] for j in range(i, args.blocks_amount)]
        args.auxCF_s.update({str(i + 1): inAuxCF3(layers_list_s, poolSize, args.num_class).cuda()})
    args.auxCF_s.update({str(args.blocks_amount): nn.AdaptiveAvgPool2d((1, 1)).cuda()})
    # args.auxCF_s.update({str(args.blocks_amount): forLast(args.t_dim[-1][1], args.t_dim[-1][1], poolSize).cuda()})

    in_channel_t = args.t_dim[-1][1] * args.blocks_amount
    in_channel_s = args.s_dim[-1][1] * args.blocks_amount
    out_channel_t = args.t_dim[-1][1]
    out_channel_s = args.s_dim[-1][1]
    fusion_dim = sum([args.t_dim[-1][1], args.s_dim[-1][1]]) // 2
    if args.fusion_method_AUXCF == 'AEConv':
        # Conv2d 1x1 Single
        args.ae_t = aeFusionConv2Single(in_channel_t, out_channel_t, fusion_dim, use_relu=args.ende_use_relu).cuda()
        args.ae_s = aeFusionConv2Single(in_channel_s, out_channel_s, fusion_dim, use_relu=args.ende_use_relu).cuda()
    elif args.fusion_method_AUXCF == 'AEConv3x3Single':
        # Conv2d 3x3 Single
        args.ae_t = aeFusionConv23x3Single(in_channel_t, out_channel_t, fusion_dim, use_relu=args.ende_use_relu).cuda()
        args.ae_s = aeFusionConv23x3Single(in_channel_s, out_channel_s, fusion_dim, use_relu=args.ende_use_relu).cuda()
    elif args.fusion_method_AUXCF == 'AEConv3x3Linear':
        # Conv2d 3x3 + Linear
        args.ae_t = aeFusionConv2(in_channel_t, out_channel_t, fusion_dim, use_relu=args.ende_use_relu).cuda()
        args.ae_s = aeFusionConv2(in_channel_s, out_channel_s, fusion_dim, use_relu=args.ende_use_relu).cuda()
    elif args.fusion_method_AUXCF == 'AELinear':
        # Linear
        args.ae_t = aeFusionLinear(in_channel_t, out_channel_t, fusion_dim, use_relu=args.ende_use_relu).cuda()
        args.ae_s = aeFusionLinear(in_channel_s, out_channel_s, fusion_dim, use_relu=args.ende_use_relu).cuda()


class Our_FWD_Conv_19_2(nn.Module):
    """ 用于计算 forward 过程中的 中间特征层 的 loss """

    def __init__(self, T, args):
        super(Our_FWD_Conv_19_2, self).__init__()
        self.T = 1.0
        # self.fwd_ae_t = AEFusionSender(args.t_shape, args.s_shape, args.ende_use_relu)
        # self.fwd_ae_s = AEFusionReceiver(args.s_shape, args.ende_use_relu)
        # self.in_dim = args.t_shape
        # self.rep_dim = args.s_shape
        # self.use_relu = args.ende_use_relu
        # self.poolSize = [1, 1]
        # if args.fusion_method_AUXCF == 'AELinear':
        #     self.fwd_ae_t = aeFusionLinear(self.in_dim, self.rep_dim, use_relu=self.use_relu)
        #     self.fwd_ae_s = aeFusionLinear(self.rep_dim, self.rep_dim, use_relu=self.use_relu)
        # elif args.fusion_method_AUXCF == 'AEConv':
        #     self.fwd_ae_t = aeFusionConv(self.in_dim, self.rep_dim, use_relu=self.use_relu)
        #     self.fwd_ae_s = aeFusionConv(self.rep_dim, self.rep_dim, use_relu=self.use_relu)
        # if args.fusion_method_AUXCF == 'AEConv':
        #     self.fwd_ae_t = aeFusionConv2Single(args.t_dim[-1][1] * args.blocks_amount, args.t_dim[-1][1], args.s_dim[-1][1],
        #                                   use_relu=self.use_relu)
        #     self.fwd_ae_s = aeFusionConv2Single(args.s_dim[-1][1] * args.blocks_amount, args.s_dim[-1][1], args.s_dim[-1][1],
        #                                   use_relu=self.use_relu)
        # elif args.fusion_method_AUXCF == 'AELinear':
        #     self.fwd_ae_t = aeFusionLinear(args.t_dim[-1][1] * args.blocks_amount, args.t_dim[-1][1], args.s_dim[-1][1],
        #                                   use_relu=self.use_relu)
        #     self.fwd_ae_s = aeFusionLinear(args.s_dim[-1][1] * args.blocks_amount, args.s_dim[-1][1], args.s_dim[-1][1],
        #                                   use_relu=self.use_relu)
        # self.fwd_auxCF_t = {}
        # self.fwd_auxCF_s = {}
        # for i in range(args.blocks_amount-1):
        #     layers_list_t = [args.t_dim[j][1] for j in range(i, args.blocks_amount)]
        #     self.fwd_auxCF_t.update({str(i+1): inAuxCF3(layers_list_t, self.poolSize, args.num_class)})
        # self.fwd_auxCF_t.update({str(args.blocks_amount): nn.AdaptiveAvgPool2d((1, 1))})
        # for i in range(args.blocks_amount-1):
        #     layers_list_s = [args.s_dim[j][1] for j in range(i, args.blocks_amount)]
        #     self.fwd_auxCF_s.update({str(i+1): inAuxCF3(layers_list_s, self.poolSize,  args.num_class)})
        # self.fwd_auxCF_s.update({str(args.blocks_amount): nn.AdaptiveAvgPool2d((1, 1))})

        # if 'CHMEAN' in args.in_method:
        #     self.method_in = channel_mean
        # elif 'AT' in args.in_method:
        #     self.method_in = at
        # elif 'MEAN_STD' in args.in_method:
        #     self.method_in = mean_std

        if 'KL' in args.in_criterion or args.in_criterion == 'MSE_softmax_T':
            self.T = T


    def forward(self, feat_t_fwd_list, feat_s_fwd_list, output_t_fwd, output_s_fwd, fwd_labels, criterion_cls, criterion_div,  args):
        N, C, H, W = feat_t_fwd_list[0].shape
        t_feat_list = []
        t_out_list = []
        s_feat_list = []
        s_out_list = []

        for i in range(args.blocks_amount-1):
            feat_t_temp, out_t_temp = args.auxCF_t[str(i+1)](feat_t_fwd_list[i])
            feat_s_temp, out_s_temp = args.auxCF_s[str(i+1)](feat_s_fwd_list[i])
            t_feat_list.append(feat_t_temp)
            t_out_list.append(out_t_temp)
            s_feat_list.append(feat_s_temp)
            s_out_list.append(out_s_temp)

        t_out_list.append(output_t_fwd)
        s_out_list.append(output_s_fwd)
        t_feat_list.append(args.auxCF_t[str(args.blocks_amount)](feat_t_fwd_list[-1]))
        s_feat_list.append(args.auxCF_s[str(args.blocks_amount)](feat_s_fwd_list[-1]))
        t_ae_list = t_feat_list
        s_ae_list = s_feat_list

        s_reshape_list = [each.float() for each in s_out_list]

        loss_ = []
        for s_1 in s_reshape_list:
            temp_ce = criterion_cls(s_1, fwd_labels)
            loss_.append(temp_ce)
            args.loss_csv_fwd.append(temp_ce.item())
            for s_2 in s_reshape_list:
                if s_1 is not s_2:
                    temp_kl = criterion_div(s_2, s_1.detach())
                    loss_.append(temp_kl)
                    args.loss_csv_fwd.append(temp_kl.item())

        loss_self_supervised_fwd = sum(loss_)
        args.loss_csv_fwd.append(loss_self_supervised_fwd.item())

        # if args.fusion_method_AUXCF == 'AEConv':
        #     for idx, each in enumerate(t_ae_list):
        #         B_temp, C_temp = each.shape
        #         t_ae_list[idx] = t_ae_list[idx].reshape(B_temp, C_temp, 1)
        #     for idx, each in enumerate(s_ae_list):
        #         B_temp, C_temp = each.shape
        #         s_ae_list[idx] = s_ae_list[idx].reshape(B_temp, C_temp, 1)
        if args.fusion_method_AUXCF == 'AELinear':
            for idx, each in enumerate(t_ae_list):
                t_ae_list[idx] = t_ae_list[idx].reshape(each.shape[0], -1)
            for idx, each in enumerate(s_ae_list):
                s_ae_list[idx] = s_ae_list[idx].reshape(each.shape[0], -1)

        t_cat = torch.cat(t_ae_list, 1)
        s_cat = torch.cat(s_ae_list, 1)

        t_fusion, t_recon = args.ae_t(t_cat)
        s_fusion, s_recon = args.ae_s(s_cat)

        # t_recon_split = torch.split(t_recon, [args.t_dim[-1][1]]*args.blocks_amount, dim=1)
        s_recon_split = torch.split(s_recon, [args.s_dim[-1][1]]*args.blocks_amount, dim=1)
        # t_recon_split = torch.split(t_recon, [args.num_class] * args.blocks_amount, dim=1)
        # s_recon_split = torch.split(s_recon, [args.num_class] * args.blocks_amount, dim=1)

        # loss_recon_fwd_t = 0.0
        loss_recon_fwd_s = 0.0
        # 重构损失
        # FWD过程中不计算T AE的相关损失
        # for each_t, each_recon in zip(t_ae_list, t_recon_split):
        #     loss_recon_fwd_t += F.mse_loss(each_t, each_recon)
        for each_s, each_recon in zip(s_ae_list, s_recon_split):
            loss_recon_fwd_s += F.mse_loss(each_s, each_recon)

        # args.loss_csv_fwd.append(loss_recon_fwd_t.item())
        args.loss_csv_fwd.append(loss_recon_fwd_s.item())

        t_fusion = t_fusion.reshape(t_fusion.shape[0], -1)
        s_fusion = s_fusion.reshape(s_fusion.shape[0], -1)

        # 教师和学生的loss
        if args.in_criterion == 'MSE':
            # loss_intfwd = (t_rep - s_fusion).pow(2).mean()
            loss_fusion_fwd = F.mse_loss(t_fusion, s_fusion)
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
            if len(t_fusion.shape) is 2:
                loss_fusion_fwd = criterion_div(s_fusion, t_fusion)
            # elif len(t_fusion.shape) > 2:
            #     # 因为展开后算上 batchsize 是三维，需要用 sum 后再平均
            #     t_fusion = t_fusion.view(t_fusion.shape[0], t_fusion.shape[1], -1)
            #     s_fusion = s_fusion.view(s_fusion.shape[0], s_fusion.shape[1], -1)
            #     criterion_int_kl = nn.KLDivLoss(reduction='sum')
            #     loss_fusion_fwd = criterion_int_kl(F.log_softmax(s_fusion / self.T, dim=-1),
            #                                    F.softmax(t_fusion / self.T, dim=-1)) * (self.T * self.T) / (N * C)

        # with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
        #     f.write("CH_Recon_T_fwd:" + str(loss_recon_fwd_t.item()) + '\t')
        #     f.write("CH_Recon_S_fwd:" + str(loss_recon_fwd_s.item()) + '\t')
        #     f.write("CH_Fusion_fwd:" + str(loss_fusion_fwd.item()) + '\t')
        #     f.write("Self-Supervised_fwd:" + str(loss_self_supervised_fwd.item()) + '\t')

        args.loss_csv_fwd.append(loss_fusion_fwd.item())
        # loss_chfwd = loss_recon_fwd + loss_fusion_fwd
        loss_chfwd_dict = dict(recon_S_fwd=loss_recon_fwd_s, fusion_fwd=loss_fusion_fwd, self_supervised_fwd=loss_self_supervised_fwd)
        return loss_chfwd_dict


class Our_FB_Conv_19_2(nn.Module):
    """ 用于计算 feedback 过程中的 中间特征层 的 loss """

    def __init__(self, T, args):
        super(Our_FB_Conv_19_2, self).__init__()
        self.T = 1.0
        # self.fb_ae_t = AEFusionSender(args.t_shape, args.s_shape, args.ende_use_relu)
        # self.fb_ae_s = AEFusionReceiver(args.s_shape, args.ende_use_relu)
        # self.in_dim = args.s_shape
        # self.rep_dim = args.t_shape
        # self.poolSize = [1, 1]
        # self.use_relu = args.ende_use_relu
        # if args.fusion_method_AUXCF == 'AELinear':
        #     self.fb_ae_t = aeFusionLinear(self.rep_dim, self.rep_dim, use_relu=self.use_relu)
        #     self.fb_ae_s = aeFusionLinear(self.in_dim, self.rep_dim, use_relu=self.use_relu)
        # elif args.fusion_method_AUXCF == 'AEConv':
        #     self.fb_ae_t = aeFusionConv(self.rep_dim, self.rep_dim, use_relu=self.use_relu)
        #     self.fb_ae_s = aeFusionConv(self.in_dim, self.rep_dim, use_relu=self.use_relu)
        # self.fb_ae_t = aeFusionLinear(, args.num_class, use_relu=args.ende_use_relu)
        # self.fb_ae_s = aeFusionLinear(args.num_class * (args.blocks_amount+1), args.num_class, use_relu=args.ende_use_relu)
        # if args.fusion_method_AUXCF == 'AEConv':
        #     self.fb_ae_t = aeFusionConv2Single(args.t_dim[-1][1] * args.blocks_amount, args.t_dim[-1][1], args.t_dim[-1][1],
        #                   use_relu=self.use_relu)
        #     self.fb_ae_s = aeFusionConv2Single(args.s_dim[-1][1] * args.blocks_amount, args.s_dim[-1][1], args.t_dim[-1][1],
        #                   use_relu=self.use_relu)
        # elif args.fusion_method_AUXCF == 'AELinear':
        #     self.fb_ae_t = aeFusionLinear(args.t_dim[-1][1] * args.blocks_amount, args.t_dim[-1][1], args.t_dim[-1][1],
        #                   use_relu=self.use_relu)
        #     self.fb_ae_s = aeFusionLinear(args.s_dim[-1][1] * args.blocks_amount, args.s_dim[-1][1], args.t_dim[-1][1],
        #                   use_relu=self.use_relu)

        # self.fb_auxCF_t = {}
        # self.fb_auxCF_s = {}
        # for i in range(args.blocks_amount-1):
        #     layers_list_t = [args.t_dim[j][1] for j in range(i, args.blocks_amount)]
        #     self.fb_auxCF_t.update({str(i + 1): inAuxCF3(layers_list_t, self.poolSize, args.num_class)})
        # self.fb_auxCF_t.update({str(args.blocks_amount): nn.AdaptiveAvgPool2d((1, 1))})
        # for i in range(args.blocks_amount-1):
        #     layers_list_s = [args.s_dim[j][1] for j in range(i, args.blocks_amount)]
        #     self.fb_auxCF_s.update({str(i + 1): inAuxCF3(layers_list_s, self.poolSize, args.num_class)})
        # self.fb_auxCF_s.update({str(args.blocks_amount): nn.AdaptiveAvgPool2d((1, 1))})

        # if 'CHMEAN' in args.in_method:
        #     self.method_in = channel_mean
        # elif 'AT' in args.in_method:
        #     self.method_in = at
        # elif 'MEAN_STD' in args.in_method:
        #     self.method_in = mean_std

        if 'KL' in args.in_criterion or args.in_criterion == 'MSE_softmax_T':
            self.T = T

    def forward(self, feat_t_fb_list, feat_s_fb_list, output_t_fb, output_s_fb, fb_labels, criterion_cls, criterion_div, loss_div_fb, model_t_weights, args):
        N, C, H, W = feat_t_fb_list[0].shape
        t_feat_list = []
        t_out_list = []
        s_feat_list = []
        s_out_list = []
        # for each_t in feat_t_fb_list:
        #     t_reshape_list.append(self.method_in(each_t, args))
        #
        # for each_s in feat_s_fb_list:
        #     s_reshape_list.append(self.method_in(each_s, args))

        for i in range(args.blocks_amount-1):
            feat_t_temp, out_t_temp = args.auxCF_t[str(i + 1)](feat_t_fb_list[i])
            feat_s_temp, out_s_temp = args.auxCF_s[str(i + 1)](feat_s_fb_list[i])
            t_feat_list.append(feat_t_temp)
            t_out_list.append(out_t_temp)
            s_feat_list.append(feat_s_temp)
            s_out_list.append(out_s_temp)

        t_out_list.append(output_t_fb)
        s_out_list.append(output_s_fb)
        t_feat_list.append(args.auxCF_t[str(args.blocks_amount)](feat_t_fb_list[-1]))
        s_feat_list.append(args.auxCF_s[str(args.blocks_amount)](feat_s_fb_list[-1]))
        t_ae_list = t_feat_list
        s_ae_list = s_feat_list

        t_reshape_list = [each.float() for each in t_out_list]

        loss_ce_ = []
        loss_kl_ = []
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
        for t_1 in t_reshape_list:
            temp_ce = criterion_cls(t_1, fb_labels)
            loss_ce_.append(temp_ce)
            args.loss_csv_fb.append(temp_ce.item())
            for t_2 in t_reshape_list:
                if t_1 is not t_2:
                    # 发送者是t_1, 接收者是t_2
                    temp_kl = criterion_div(t_2, t_1.detach())
                    loss_kl_.append(temp_kl)
                    args.loss_csv_fb.append(temp_kl.item())
                    # if args.fbUseGradSim is True:
                    #     # 将 t_1 迁移到 t_2 的知识（KL loss），与 t_1 与 label的知识（CE loss）进行梯度的相似度比较。
                    #     cos_similarity_temp = cal_each_grad_sim(temp_ce, temp_kl, model_t_weights, args)
                    #     if cos_similarity_temp < 0:
                    #         pass
                    #     else:
                    #         loss_kl_.append(temp_kl)
                    # else:
                    #     loss_kl_.append(temp_kl)

        # loss_ce_all = sum(loss_ce_)
        # loss_kl_all = sum(loss_kl_)
        # loss_self_supervised_fb = loss_ce_all + loss_kl_all
        # loss_cls = loss_ce_[-1]

        # loss_cls = loss_ce_[-1]

        # if args.fusion_method_AUXCF == 'AEConv':
        #     for idx, each in enumerate(t_ae_list):
        #         B_temp, C_temp = each.shape
        #         t_ae_list[idx] = t_ae_list[idx].reshape(B_temp, C_temp, 1)
        #     for idx, each in enumerate(s_ae_list):
        #         B_temp, C_temp = each.shape
        #         s_ae_list[idx] = s_ae_list[idx].reshape(B_temp, C_temp, 1)
        if args.fusion_method_AUXCF == 'AELinear':
            for idx, each in enumerate(t_ae_list):
                t_ae_list[idx] = t_ae_list[idx].reshape(each.shape[0], -1)
            for idx, each in enumerate(s_ae_list):
                s_ae_list[idx] = s_ae_list[idx].reshape(each.shape[0], -1)

        t_cat = torch.cat(t_ae_list, 1)
        s_cat = torch.cat(s_ae_list, 1)

        t_fusion, t_recon = args.ae_t(t_cat)
        s_fusion, s_recon = args.ae_s(s_cat)

        t_recon_split = torch.split(t_recon, [args.t_dim[-1][1]]*args.blocks_amount, dim=1)
        # s_recon_split = torch.split(s_recon, [args.s_dim[-1][1]]*args.blocks_amount, dim=1)
        # t_recon_split = torch.split(t_recon, [args.num_class] * args.blocks_amount, dim=1)
        # s_recon_split = torch.split(s_recon, [args.num_class] * args.blocks_amount, dim=1)

        # if 'CHMEAN' in args.in_method or 'MEAN_STD' in args.in_method:
        #     t_recon_split = torch.split(t_recon, [each[1] for each in args.t_dim], dim=1)
        #     s_recon_split = torch.split(s_recon, [each[1] for each in args.s_dim], dim=1)
        # elif 'AT' in args.in_method:
        #     t_recon_split = torch.split(t_recon, [each[2]*each[3] for each in args.t_dim], dim=1)
        #     s_recon_split = torch.split(s_recon, [each[2]*each[3] for each in args.s_dim], dim=1)
        loss_recon_fb_t = 0.0
        # loss_recon_fb_s = 0.0
        # 重构损失
        for each_t, each_recon in zip(t_ae_list, t_recon_split):
            loss_recon_fb_t += F.mse_loss(each_t, each_recon)
        # for each_s, each_recon in zip(s_ae_list, s_recon_split):
        #     loss_recon_fb_s += F.mse_loss(each_s, each_recon)

        args.loss_csv_fb.append(loss_recon_fb_t.item())
        # args.loss_csv_fb.append(loss_recon_fb_s.item())

        t_fusion = t_fusion.reshape(t_fusion.shape[0], -1)
        s_fusion = s_fusion.reshape(s_fusion.shape[0], -1)

        # 教师和学生的loss
        if args.in_criterion == 'MSE':
            # loss_intfb = (t_rep - s_fusion).pow(2).mean()
            loss_fusion_fb = F.mse_loss(t_fusion, s_fusion)
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
            if len(t_fusion.shape) is 2:
                loss_fusion_fb = criterion_div(t_fusion, s_fusion)
            # elif len(t_fusion.shape) > 2:
            #     # 因为展开后算上 batchsize 是三维，需要用 sum 后再平均
            #     t_fusion = t_fusion.view(t_fusion.shape[0], t_fusion.shape[1], -1)
            #     s_fusion = s_fusion.view(s_fusion.shape[0], s_fusion.shape[1], -1)
            #     criterion_int_kl = nn.KLDivLoss(reduction='sum')
            #     loss_fusion_fb = criterion_int_kl(F.log_softmax(t_fusion / self.T, dim=-1),
            #                                        F.softmax(s_fusion / self.T, dim=-1)) * (self.T * self.T) / (N * C)

        # loss_kl_.append(loss_fusion_fb)
        loss_ce_all = sum(loss_ce_)
        loss_kl_all = sum(loss_kl_)
        args.loss_csv_fb.append(loss_fusion_fb.item())
        args.loss_csv_fb.append(loss_ce_all.item())
        args.loss_csv_fb.append(loss_kl_all.item())
        # loss_self_supervised_fb = 0.0


        if args.fbUseGradSim is True:
            # 将 损失 做梯度相似性比较。
            loss_fb_kl = loss_fusion_fb + loss_div_fb
            # cls_grad = torch.autograd.grad([loss_ce_all, loss_kl_all], model_t_weights, allow_unused=True, retain_graph=True)
            cls_grad = torch.autograd.grad(loss_ce_all, model_t_weights, allow_unused=True, retain_graph=True)
            kl_all_grad = torch.autograd.grad(loss_kl_all, model_t_weights, allow_unused=True, retain_graph=True)
            div_kl_grad = torch.autograd.grad(loss_fb_kl, model_t_weights, allow_unused=True, retain_graph=True)
            cos_similarity_temp = cal_each_grad_sim2(cls_grad, kl_all_grad, args)
            # cos_similarity_fusion = cal_each_grad_sim(loss_ce_all, loss_fusion_fb, model_t_weights, args)
            # cos_similarity_div = cal_each_grad_sim(loss_ce_all, loss_div_fb, model_t_weights, args)
            # args.loss_csv_fb.append(cos_similarity_temp.item())
            # args.loss_csv_fb.append(cos_similarity_fusion.item())
            # args.loss_csv_fb.append(cos_similarity_div.item())
            cos_similarity_div = cal_each_grad_sim2(cls_grad, div_kl_grad, args)
            args.loss_csv_fb.append(cos_similarity_temp.item())
            args.loss_csv_fb.append(cos_similarity_div.item())
            args.loss_csv_fb.append('None')
            self_kl_all_coe = 1.0
            fusion_coe = 1.0
            div_coe = 1.0
            if cos_similarity_temp < 0:
                self_kl_all_coe = 0
            # if cos_similarity_fusion < 0:
            #     fusion_coe = 0
            if cos_similarity_div < 0:
                div_coe = 0
            # loss_fb_other = loss_ce_all + self_kl_all_coe * loss_kl_all + fusion_coe * loss_fusion_fb + div_coe * loss_div_fb
            loss_fb_other = loss_ce_all + self_kl_all_coe * loss_kl_all + div_coe * loss_fb_kl
            # if cos_similarity_temp < 0:
            #     loss_self_supervised_fb = loss_ce_all
            # else:
            #     loss_self_supervised_fb = loss_ce_all + loss_kl_all
        else:
            loss_fb_kl = loss_fusion_fb + loss_div_fb
            args.loss_csv_fb.append('None')
            args.loss_csv_fb.append('None')
            args.loss_csv_fb.append('None')
            # loss_fb_other = loss_ce_all + loss_kl_all + loss_fusion_fb + loss_div_fb
            loss_fb_other = loss_ce_all + loss_kl_all + loss_fb_kl

        # with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
        #     f.write("CH_Recon_T_fb:" + str(loss_recon_fb_t.item()) + '\t')
        #     f.write("CH_Recon_S_fb:" + str(loss_recon_fb_s.item()) + '\t')
        #     f.write("CH_Fusion_fb:" + str(loss_fusion_fb.item()) + '\t')
        #     f.write("Self-Supervised_fb:" + str(loss_self_supervised_fb.item()) + '\t')

        # loss_chfb = loss_recon_fb + loss_fusion_fb
        loss_chfb_dict = dict(recon_T_fb=loss_recon_fb_t, loss_other=loss_fb_other)
        return loss_chfb_dict

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

    # print("cls")
    # print(nontype_cls)
    # print('other')
    # print(nontype_oth)

    cosin_simility_all = torch.mean(torch.Tensor(
        [F.cosine_similarity(each_cls.reshape(-1), each_other.reshape(-1), dim=0) for each_cls, each_other in
         zip(cls_grad[:split_idx], other_grad[:split_idx])]).cuda())

    # cosin_simility = []
    # for each_cls, each_other in zip(cls_grad[:split_idx], other_grad[:split_idx]):
    #     cosin_simility.append(F.cosine_similarity(each_cls.reshape(-1), each_other.reshape(-1), dim=0))
    #
    # cosin_simility_all = torch.tensor(0.0).cuda()
    # for each_sim in cosin_simility:
    #     cosin_simility_all += torch.mean(each_sim)

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

    # print("cls")
    # print(nontype_cls)
    # print('other')
    # print(nontype_oth)

    cosin_simility_all = torch.mean(torch.Tensor(
        [F.cosine_similarity(each_cls.reshape(-1), each_other.reshape(-1), dim=0) for each_cls, each_other in
         zip(cls_grad[:split_idx], other_grad[:split_idx])]).cuda())

    # cosin_simility = []
    # for each_cls, each_other in zip(cls_grad[:split_idx], other_grad[:split_idx]):
    #     cosin_simility.append(F.cosine_similarity(each_cls.reshape(-1), each_other.reshape(-1), dim=0))
    #
    # cosin_simility_all = torch.tensor(0.0).cuda()
    # for each_sim in cosin_simility:
    #     cosin_simility_all += torch.mean(each_sim)

    return cosin_simility_all

def main():
    # a = torch.randn(2, 3, 32, 32)
    # auxcf = inAuxCF(3, 3, 100)
    # # print(torchinfo.summary(auxcf))
    # a_auxcf = auxcf(a)
    # print(a_auxcf.shape)

    a = torch.randn(2, 64, 32, 32)

    auxnet1 = inAuxCF3([64, 128, 256, 512])
    auxnet2 = inAuxCF3([128, 256, 512])
    auxnet3 = inAuxCF3([256, 512])

    # feat, out = auxnet(a)


    # print(feat.shape, out.shape)

    total1 = sum([param.nelement() for param in auxnet1.parameters()])
    total2 = sum([param.nelement() for param in auxnet2.parameters()])
    total3 = sum([param.nelement() for param in auxnet3.parameters()])

    print("Parmeter: %.2fM" % (total1 / 1e6))
    print("Parmeter: %.2fM" % (total2 / 1e6))
    print("Parmeter: %.2fM" % (total3 / 1e6))
    print("Parmeter: %.2fM" % ((total1 + total2 + total3) / 1e6))

    # print(auxnet)

    aenet1 = aeFusionConv2(512*4, 512, 512)
    aenet2 = aeFusionConv2(512*4, 512, 512)
    # aenet3 = aeFusionConv2(512 * 4, 512, 512)
    # aenet4 = aeFusionConv2(512 * 4, 512, 512)
    # ae_input = torch.randn(2, 512*3, 1, 1)
    # rep, recon = aenet(ae_input)

    # print(rep.shape, recon.shape)

    total_ae1 = sum([param.nelement() for param in aenet1.parameters()])
    total_ae2 = sum([param.nelement() for param in aenet2.parameters()])
    # total_ae3 = sum([param.nelement() for param in aenet3.parameters()])
    # total_ae4 = sum([param.nelement() for param in aenet4.parameters()])

    print("Parmeter: %.2fM" % (total_ae1 / 1e6))
    print("Parmeter: %.2fM" % (total_ae2 / 1e6))
    print("Parmeter: %.2fM" % ((total_ae1 + total_ae2) / 1e6))

    aenetsmall1 = aeFusionConv2Single(512 * 4, 512, 512)
    aenetsmall2 = aeFusionConv2Single(512 * 4, 512, 512)

    total_aesmall1 = sum([param.nelement() for param in aenetsmall1.parameters()])
    total_aesmall2 = sum([param.nelement() for param in aenetsmall2.parameters()])

    print("Parmeter: %.2fM" % (total_aesmall1 / 1e6))
    print("Parmeter: %.2fM" % (total_aesmall2 / 1e6))
    print("Parmeter: %.2fM" % ((total_aesmall1 + total_aesmall2) / 1e6))

    aenetlinear1 = aeFusionLinear(512 * 4, 512, 512)
    aenetlinear2 = aeFusionLinear(512 * 4, 512, 512)

    total_aelinear1 = sum([param.nelement() for param in aenetlinear1.parameters()])
    total_aelinear2 = sum([param.nelement() for param in aenetlinear2.parameters()])

    print("Parmeter: %.2fM" % (total_aelinear1 / 1e6))
    print("Parmeter: %.2fM" % (total_aelinear2 / 1e6))
    print("Parmeter: %.2fM" % ((total_aelinear1 + total_aelinear2) / 1e6))

    print(123)


if __name__ == '__main__':
    main()