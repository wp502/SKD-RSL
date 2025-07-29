# -*- coding: utf-8 -*-
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
import os
import numpy as np
import time
import logging
import copy

__all__ = ["Our_FWD_Conv_111", "Our_FB_Conv_111", "initAUXCFAndAE111"]

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


class inAuxCF3(nn.Module):
    """ 用于每个中间层的辅助分类器 """

    def __init__(self, planes, block_amount=1, num_classes=100):
        super(inAuxCF3, self).__init__()
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
    poolSize = [1, 1]
    args.auxCF_t = {}
    args.auxCF_s = {}
    block_method_t = args.aux_method_t
    block_method_s = args.aux_method_s
    for i in range(args.blocks_amount_t - 1):
        layers_list_t = [args.t_dim[j][1] for j in range(i, args.blocks_amount_t)]
        args.auxCF_t.update({str(i + 1): inAuxCF(layers_list_t, poolSize, block_method_t, args.num_class).cuda()})
    args.auxCF_t.update({str(args.blocks_amount_t): nn.AdaptiveAvgPool2d((1, 1)).cuda()})
    # args.auxCF_t.update({str(args.blocks_amount): forLastJustRelu(poolSize).cuda()})
    # args.auxCF_t.update({str(args.blocks_amount): forLast(args.t_dim[-1][1], args.t_dim[-1][1], poolSize).cuda()})
    for i in range(args.blocks_amount_s - 1):
        layers_list_s = [args.s_dim[j][1] for j in range(i, args.blocks_amount_s)]
        args.auxCF_s.update({str(i + 1): inAuxCF(layers_list_s, poolSize, block_method_s, args.num_class).cuda()})
    args.auxCF_s.update({str(args.blocks_amount_s): nn.AdaptiveAvgPool2d((1, 1)).cuda()})
    # args.auxCF_s.update({str(args.blocks_amount): forLastJustRelu(poolSize).cuda()})
    # args.auxCF_s.update({str(args.blocks_amount): forLast(args.t_dim[-1][1], args.t_dim[-1][1], poolSize).cuda()})

    in_channel_t = args.t_dim[-1][1] * args.blocks_amount_t
    in_channel_s = args.s_dim[-1][1] * args.blocks_amount_s
    out_channel_t = in_channel_t // 2
    out_channel_s = in_channel_s // 2
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


class Our_FWD_Conv_111(nn.Module):
    """ 用于计算 forward 过程中的 中间特征层 的 loss """

    def __init__(self, T, args):
        super(Our_FWD_Conv_111, self).__init__()
        self.T = 1.0
        if 'KL' in args.in_criterion or args.in_criterion == 'MSE_softmax_T':
            self.T = T


    def forward(self, feat_t_fwd_list, feat_s_fwd_list, output_s_fwd, fwd_labels, criterion_cls, criterion_div,  args):
        N, C, H, W = feat_t_fwd_list[0].shape
        t_feat_list = []
        # t_out_list = []
        s_feat_list = []
        s_out_list = []

        with torch.no_grad():
            for i in range(args.blocks_amount_t-1):
                feat_t_temp, out_t_temp = args.auxCF_t[str(i+1)](feat_t_fwd_list[i])
                t_feat_list.append(feat_t_temp)
                # t_out_list.append(out_t_temp)

        for i in range(args.blocks_amount_s-1):
            feat_s_temp, out_s_temp = args.auxCF_s[str(i+1)](feat_s_fwd_list[i])
            s_feat_list.append(feat_s_temp)
            s_out_list.append(out_s_temp)

        # t_out_list.append(output_t_fwd)
        s_out_list.append(output_s_fwd)
        t_feat_list.append(args.auxCF_t[str(args.blocks_amount_t)](feat_t_fwd_list[-1]))
        s_feat_list.append(args.auxCF_s[str(args.blocks_amount_s)](feat_s_fwd_list[-1]))
        t_ae_list = t_feat_list
        s_ae_list = s_feat_list

        s_reshape_list = [each.float() for each in s_out_list]

        # loss_ce_ = []
        # loss_deep_shallow_ = []
        # loss_shallow_deep_ = []
        # if args.NO_SELF is True:
        #     # 如果不考虑 self distillation的情况下，只保留了CE loss，以保证AUXCF的更新
        #     # for s_1 in s_reshape_list:
        #     #     temp_ce = criterion_cls(s_1, fwd_labels)
        #     #     loss_.append(temp_ce)
        #     #     args.loss_csv_fwd.append(temp_ce.item())
        #     #     for s_2 in s_reshape_list:
        #     #         if s_1 is not s_2:
        #     #             temp_kl = torch.Tensor([0.0]).cuda()
        #     #             loss_.append(temp_kl)
        #     #             args.loss_csv_fwd.append(temp_kl.item())
        #
        #     for s_1 in s_reshape_list:
        #         temp_ce = criterion_cls(s_1, fwd_labels)
        #         loss_ce_.append(temp_ce)
        #         args.loss_csv_fwd.append(temp_ce.item())
        #
        #
        #     for i in range(args.blocks_amount_s - 1):
        #         # deep - shallow
        #         deep_sender_idx = args.blocks_amount_s-1-i
        #         # deep_sender = s_reshape_list[deep_sender_idx]
        #         deep_receiver = s_reshape_list[:deep_sender_idx]
        #         for each in deep_receiver:
        #             temp_kl = torch.Tensor([0.0]).cuda()
        #             loss_deep_shallow_.append(temp_kl)
        #         # shallow - deep
        #         shallow_sender_idx = i
        #         # shallow_sender = s_reshape_list[shallow_sender_idx]
        #         shallow_receiver = s_reshape_list[shallow_sender_idx+1:]
        #         for each in shallow_receiver:
        #             temp_kl = torch.Tensor([0.0]).cuda()
        #             loss_shallow_deep_.append(temp_kl)
        #
        #
        #
        # else:
        #     # for s_1 in s_reshape_list:
        #     #     temp_ce = criterion_cls(s_1, fwd_labels)
        #     #     loss_.append(temp_ce)
        #     #     args.loss_csv_fwd.append(temp_ce.item())
        #     #     for s_2 in s_reshape_list:
        #     #         if s_1 is not s_2:
        #     #             temp_kl = criterion_div(s_2, s_1)
        #     #             loss_.append(temp_kl)
        #     #             args.loss_csv_fwd.append(temp_kl.item())
        #
        #     for s_1 in s_reshape_list:
        #         temp_ce = criterion_cls(s_1, fwd_labels)
        #         loss_ce_.append(temp_ce)
        #         args.loss_csv_fwd.append(temp_ce.item())
        #
        #
        #     for i in range(args.blocks_amount_s - 1):
        #         # deep - shallow
        #         deep_sender_idx = args.blocks_amount_s-1-i
        #         deep_sender = s_reshape_list[deep_sender_idx]
        #         deep_receiver = s_reshape_list[:deep_sender_idx]
        #         for each in deep_receiver:
        #             temp_kl = criterion_div(each, deep_sender)
        #             loss_deep_shallow_.append(temp_kl)
        #         # shallow - deep
        #         shallow_sender_idx = i
        #         shallow_sender = s_reshape_list[shallow_sender_idx]
        #         shallow_receiver = s_reshape_list[shallow_sender_idx+1:]
        #         for each in shallow_receiver:
        #             temp_kl = criterion_div(each, shallow_sender)
        #             loss_shallow_deep_.append(temp_kl)
        loss_ce_all, loss_kl_all, loss_self_supervised_fwd = self_supervised_loss(fwd_labels, s_reshape_list,
                                                                                args.blocks_amount_s, criterion_cls,
                                                                                criterion_div, args.loss_csv_fwd,
                                                                                 args.self_method_s, args)

        # loss_ce_all = sum(loss_ce_)
        # args.loss_csv_fwd.append(loss_ce_all.item())
        #
        # if args.self_method == 'deep_shallow':
        #     loss_kl_all = sum(loss_deep_shallow_)
        # elif args.self_method == 'shallow_deep':
        #     loss_kl_all = sum(loss_shallow_deep_)
        # elif args.self_method == 'bi_directional':
        #     loss_kl_all = sum(loss_deep_shallow_) + sum(loss_shallow_deep_)
        #
        # args.loss_csv_fwd.append(loss_kl_all.item())
        #
        # loss_self_supervised_fwd = loss_ce_all + loss_kl_all
        # args.loss_csv_fwd.append(loss_self_supervised_fwd.item())

        if args.fusion_method_AUXCF == 'AELinear':
            t_ae_list = [each.reshape(each.shape[0], -1) for each in t_ae_list]
            s_ae_list = [each.reshape(each.shape[0], -1) for each in s_ae_list]

            # for idx, each in enumerate(t_ae_list):
            #     t_ae_list[idx] = t_ae_list[idx].reshape(each.shape[0], -1)
            # for idx, each in enumerate(s_ae_list):
            #     s_ae_list[idx] = s_ae_list[idx].reshape(each.shape[0], -1)

        t_cat = torch.cat(t_ae_list, 1)
        s_cat = torch.cat(s_ae_list, 1)

        with torch.no_grad():
            t_fusion, t_recon = args.ae_t(t_cat)

        s_fusion, s_recon = args.ae_s(s_cat)

        # t_recon_split = torch.split(t_recon, [args.t_dim[-1][1]]*args.blocks_amount, dim=1)
        s_recon_split = torch.split(s_recon, [args.s_dim[-1][1]]*args.blocks_amount_s, dim=1)
        # t_recon_split = torch.split(t_recon, [args.num_class] * args.blocks_amount, dim=1)
        # s_recon_split = torch.split(s_recon, [args.num_class] * args.blocks_amount, dim=1)

        # loss_recon_fwd_t = 0.0
        loss_recon_fwd_s = 0.0
        # 重构损失
        # FWD过程中不计算T AE的相关损失
        # for each_t, each_recon in zip(t_ae_list, t_recon_split):
        #     loss_recon_fwd_t += F.mse_loss(each_t, each_recon)
        # for each_s, each_recon in zip(s_ae_list, s_recon_split):
        #     loss_recon_fwd_s += F.mse_loss(each_s, each_recon)

        loss_recon_fwd_ = [F.mse_loss(each_s, each_recon) for each_s, each_recon in zip(s_ae_list, s_recon_split)]
        loss_recon_fwd_s = sum(loss_recon_fwd_)

        # args.loss_csv_fwd.append(loss_recon_fwd_t.item())
        args.loss_csv_fwd.append(loss_recon_fwd_s.item())

        t_fusion = t_fusion.reshape(t_fusion.shape[0], -1)
        s_fusion = s_fusion.reshape(s_fusion.shape[0], -1)

        # 教师和学生的 Fusion loss
        if args.NO_FUSION is True:
            loss_fusion_fwd = torch.Tensor([0.0]).cuda()
            args.loss_csv_fwd.append(loss_fusion_fwd.item())
        else:
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
                assert len(t_fusion.shape) == 2
                loss_fusion_fwd = criterion_div(s_fusion, t_fusion)
                # elif len(t_fusion.shape) > 2:
                #     # 因为展开后算上 batchsize 是三维，需要用 sum 后再平均
                #     t_fusion = t_fusion.view(t_fusion.shape[0], t_fusion.shape[1], -1)
                #     s_fusion = s_fusion.view(s_fusion.shape[0], s_fusion.shape[1], -1)
                #     criterion_int_kl = nn.KLDivLoss(reduction='sum')
                #     loss_fusion_fwd = criterion_int_kl(F.log_softmax(s_fusion / self.T, dim=-1),
                #                                    F.softmax(t_fusion / self.T, dim=-1)) * (self.T * self.T) / (N * C)

            args.loss_csv_fwd.append(loss_fusion_fwd.item())
        # loss_chfwd = loss_recon_fwd + loss_fusion_fwd
        loss_chfwd_dict = dict(recon_S_fwd=loss_recon_fwd_s, fusion_fwd=loss_fusion_fwd, self_supervised_fwd=loss_self_supervised_fwd)
        return loss_chfwd_dict


class Our_FB_Conv_111(nn.Module):
    """ 用于计算 feedback 过程中的 中间特征层 的 loss """

    def __init__(self, T, args):
        super(Our_FB_Conv_111, self).__init__()
        self.T = 1.0
        if 'KL' in args.in_criterion or args.in_criterion == 'MSE_softmax_T':
            self.T = T

    def forward(self, feat_t_fb_list, feat_s_fb_list, output_t_fb, fb_labels, criterion_cls, criterion_div, loss_div_fb, model_t_weights, args):
        N, C, H, W = feat_t_fb_list[0].shape
        t_feat_list = []
        t_out_list = []
        s_feat_list = []
        # s_out_list = []
        # for each_t in feat_t_fb_list:
        #     t_reshape_list.append(self.method_in(each_t, args))
        #
        # for each_s in feat_s_fb_list:
        #     s_reshape_list.append(self.method_in(each_s, args))
        for i in range(args.blocks_amount_t-1):
            feat_t_temp, out_t_temp = args.auxCF_t[str(i + 1)](feat_t_fb_list[i])
            t_feat_list.append(feat_t_temp)
            t_out_list.append(out_t_temp)

        with torch.no_grad():
            for i in range(args.blocks_amount_s-1):
                feat_s_temp, out_s_temp = args.auxCF_s[str(i + 1)](feat_s_fb_list[i])
                s_feat_list.append(feat_s_temp)
                # s_out_list.append(out_s_temp)

        t_out_list.append(output_t_fb)
        # s_out_list.append(output_s_fb)
        t_feat_list.append(args.auxCF_t[str(args.blocks_amount_t)](feat_t_fb_list[-1]))
        s_feat_list.append(args.auxCF_s[str(args.blocks_amount_s)](feat_s_fb_list[-1]))
        t_ae_list = t_feat_list
        s_ae_list = s_feat_list

        t_reshape_list = [each.float() for each in t_out_list]

        # loss_ce_ = []
        # loss_kl_ = []
        # if args.NO_SELF is True:
        #     for t_1 in t_reshape_list:
        #         temp_ce = criterion_cls(t_1, fb_labels)
        #         loss_ce_.append(temp_ce)
        #         args.loss_csv_fb.append(temp_ce.item())
        #         for t_2 in t_reshape_list:
        #             if t_1 is not t_2:
        #                 # 发送者是t_1, 接收者是t_2
        #                 temp_kl = torch.Tensor([0.0]).cuda()
        #                 loss_kl_.append(temp_kl)
        #                 args.loss_csv_fb.append(temp_kl.item())
        # else:
        #     for t_1 in t_reshape_list:
        #         temp_ce = criterion_cls(t_1, fb_labels)
        #         loss_ce_.append(temp_ce)
        #         args.loss_csv_fb.append(temp_ce.item())
        #         for t_2 in t_reshape_list:
        #             if t_1 is not t_2:
        #                 # 发送者是t_1, 接收者是t_2
        #                 temp_kl = criterion_div(t_2, t_1)
        #                 loss_kl_.append(temp_kl)
        #                 args.loss_csv_fb.append(temp_kl.item())
        #             # if args.fbUseGradSim is True:
        #             #     # 将 t_1 迁移到 t_2 的知识（KL loss），与 t_1 与 label的知识（CE loss）进行梯度的相似度比较。
        #             #     cos_similarity_temp = cal_each_grad_sim(temp_ce, temp_kl, model_t_weights, args)
        #             #     if cos_similarity_temp < 0:
        #             #         pass
        #             #     else:
        #             #         loss_kl_.append(temp_kl)
        #             # else:
        #             #     loss_kl_.append(temp_kl)
        # loss_ce_all = sum(loss_ce_)
        # loss_kl_all = sum(loss_kl_)
        # loss_self_supervised_fb = loss_ce_all + loss_kl_all
        # loss_cls = loss_ce_[-1]

        # loss_cls = loss_ce_[-1]

        loss_ce_all, loss_kl_all, loss_self_supervised_fb = self_supervised_loss(fb_labels, t_reshape_list,
                                                                                args.blocks_amount_t, criterion_cls,
                                                                                criterion_div, args.loss_csv_fb,
                                                                                 args.self_method_t, args)

        # loss_ce_all = sum(loss_ce_)
        # args.loss_csv_fb.append(loss_ce_all.item())
        #
        # if args.self_method == 'deep_shallow':
        #     loss_kl_all = sum(loss_deep_shallow_)
        # elif args.self_method == 'shallow_deep':
        #     loss_kl_all = sum(loss_shallow_deep_)
        # elif args.self_method == 'bi_directional':
        #     loss_kl_all = sum(loss_deep_shallow_) + sum(loss_shallow_deep_)
        #
        # args.loss_csv_fb.append(loss_kl_all.item())
        #
        # loss_self_supervised_fb = loss_ce_all + loss_kl_all
        # args.loss_csv_fb.append(loss_self_supervised_fb.item())

        if args.fusion_method_AUXCF == 'AELinear':
            t_ae_list = [each.reshape(each.shape[0], -1) for each in t_ae_list]
            s_ae_list = [each.reshape(each.shape[0], -1) for each in s_ae_list]
            # for idx, each in enumerate(t_ae_list):
            #     t_ae_list[idx] = t_ae_list[idx].reshape(each.shape[0], -1)
            # for idx, each in enumerate(s_ae_list):
            #     s_ae_list[idx] = s_ae_list[idx].reshape(each.shape[0], -1)

        t_cat = torch.cat(t_ae_list, 1)
        s_cat = torch.cat(s_ae_list, 1)

        t_fusion, t_recon = args.ae_t(t_cat)

        with torch.no_grad():
            s_fusion, s_recon = args.ae_s(s_cat)

        t_recon_split = torch.split(t_recon, [args.t_dim[-1][1]]*args.blocks_amount_t, dim=1)

        loss_recon_fb_t = 0.0
        # loss_recon_fb_s = 0.0
        # 重构损失
        # for each_t, each_recon in zip(t_ae_list, t_recon_split):
        #     loss_recon_fb_t += F.mse_loss(each_t, each_recon)
        # for each_s, each_recon in zip(s_ae_list, s_recon_split):
        #     loss_recon_fb_s += F.mse_loss(each_s, each_recon)
        loss_recon_fb_ = [F.mse_loss(each_t, each_recon) for each_t, each_recon in zip(t_ae_list, t_recon_split)]
        loss_recon_fb_t = sum(loss_recon_fb_)

        args.loss_csv_fb.append(loss_recon_fb_t.item())
        # args.loss_csv_fb.append(loss_recon_fb_s.item())

        t_fusion = t_fusion.reshape(t_fusion.shape[0], -1)
        s_fusion = s_fusion.reshape(s_fusion.shape[0], -1)

        # 教师和学生的 Fusion loss
        if args.NO_FUSION is True:
            loss_fusion_fb = torch.Tensor([0.0]).cuda()
        else:
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
                assert len(t_fusion.shape) == 2
                loss_fusion_fb = criterion_div(t_fusion, s_fusion)
            # elif len(t_fusion.shape) > 2:
            #     # 因为展开后算上 batchsize 是三维，需要用 sum 后再平均
            #     t_fusion = t_fusion.view(t_fusion.shape[0], t_fusion.shape[1], -1)
            #     s_fusion = s_fusion.view(s_fusion.shape[0], s_fusion.shape[1], -1)
            #     criterion_int_kl = nn.KLDivLoss(reduction='sum')
            #     loss_fusion_fb = criterion_int_kl(F.log_softmax(t_fusion / self.T, dim=-1),
            #                                        F.softmax(s_fusion / self.T, dim=-1)) * (self.T * self.T) / (N * C)

        # loss_kl_.append(loss_fusion_fb)
        # loss_ce_all = sum(loss_ce_)
        # loss_kl_all = sum(loss_kl_)
        args.loss_csv_fb.append(loss_fusion_fb.item())
        # args.loss_csv_fb.append(loss_ce_all.item())
        # args.loss_csv_fb.append(loss_kl_all.item())
        # loss_self_supervised_fb = 0.0


        if args.fbUseGradSim is True:
            # 将 损失 做梯度相似性比较。
            loss_fb_kl = loss_fusion_fb + loss_div_fb
            # cls_grad = torch.autograd.grad([loss_ce_all, loss_kl_all], model_t_weights, allow_unused=True, retain_graph=True)
            cls_grad = torch.autograd.grad(loss_ce_all, model_t_weights, allow_unused=True, retain_graph=True)
            div_kl_grad = torch.autograd.grad(loss_fb_kl, model_t_weights, allow_unused=True, retain_graph=True)
            cos_similarity_feedback = cal_each_grad_sim2(cls_grad, div_kl_grad, args)
            args.loss_csv_fb.append(cos_similarity_feedback.item())
            feedback_coe = 1.0
            if cos_similarity_feedback < 0:
                feedback_coe = 0.0

            if args.NO_SELF is False:
                kl_all_grad = torch.autograd.grad(loss_kl_all, model_t_weights, allow_unused=True, retain_graph=True)
                cos_similarity_self = cal_each_grad_sim2(cls_grad, kl_all_grad, args)
                args.loss_csv_fb.append(cos_similarity_self.item())
                self_kl_all_coe = 1.0
                if cos_similarity_self < 0:
                    self_kl_all_coe = 0.0
            else:
                args.loss_csv_fb.append('0')
                self_kl_all_coe = 0.0

            args.loss_csv_fb.append('None')
            # loss_fb_other = loss_ce_all + self_kl_all_coe * loss_kl_all + fusion_coe * loss_fusion_fb + div_coe * loss_div_fb
            loss_fb_other = loss_ce_all + self_kl_all_coe * loss_kl_all + feedback_coe * loss_fb_kl
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

        # loss_chfb = loss_recon_fb + loss_fusion_fb
        loss_chfb_dict = dict(recon_T_fb=loss_recon_fb_t, loss_other=loss_fb_other)
        return loss_chfb_dict

def self_supervised_loss(labels, reshape_list, block_amounts, criterion_cls, criterion_div, loss_csv, self_method, args):
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
        # for s_1 in s_reshape_list:
        #     temp_ce = criterion_cls(s_1, fwd_labels)
        #     loss_.append(temp_ce)
        #     args.loss_csv_fwd.append(temp_ce.item())
        #     for s_2 in s_reshape_list:
        #         if s_1 is not s_2:
        #             temp_kl = torch.Tensor([0.0]).cuda()
        #             loss_.append(temp_kl)
        #             args.loss_csv_fwd.append(temp_kl.item())

        for s_1 in reshape_list:
            temp_ce = criterion_cls(s_1, labels)
            loss_ce_.append(temp_ce)
            loss_csv.append(temp_ce.item())

        for i in range(block_amounts - 1):
            # deep - shallow
            deep_sender_idx = block_amounts - 1 - i
            # deep_sender = s_reshape_list[deep_sender_idx]
            deep_receiver = reshape_list[:deep_sender_idx]
            for each in deep_receiver:
                temp_kl_deep = torch.Tensor([0.0]).cuda()
                loss_deep_shallow_.append(temp_kl_deep)
                loss_csv.append(temp_kl_deep.item())
            # shallow - deep
            shallow_sender_idx = i
            # shallow_sender = s_reshape_list[shallow_sender_idx]
            shallow_receiver = reshape_list[shallow_sender_idx + 1:]
            for each in shallow_receiver:
                temp_kl_shallow = torch.Tensor([0.0]).cuda()
                loss_shallow_deep_.append(temp_kl_shallow)
                loss_csv.append(temp_kl_shallow.item())
    else:
        # for s_1 in s_reshape_list:
        #     temp_ce = criterion_cls(s_1, fwd_labels)
        #     loss_.append(temp_ce)
        #     args.loss_csv_fwd.append(temp_ce.item())
        #     for s_2 in s_reshape_list:
        #         if s_1 is not s_2:
        #             temp_kl = criterion_div(s_2, s_1)
        #             loss_.append(temp_kl)
        #             args.loss_csv_fwd.append(temp_kl.item())

        for s_1 in reshape_list:
            temp_ce = criterion_cls(s_1, labels)
            loss_ce_.append(temp_ce)
            loss_csv.append(temp_ce.item())

        for i in range(block_amounts - 1):
            # deep - shallow
            deep_sender_idx = block_amounts - 1 - i
            deep_sender = reshape_list[deep_sender_idx]
            deep_receiver = reshape_list[:deep_sender_idx]
            for each in deep_receiver:
                temp_kl_deep = criterion_div(each, deep_sender)
                loss_deep_shallow_.append(temp_kl_deep)
                loss_csv.append(temp_kl_deep.item())
            # shallow - deep
            shallow_sender_idx = i
            shallow_sender = reshape_list[shallow_sender_idx]
            shallow_receiver = reshape_list[shallow_sender_idx + 1:]
            for each in shallow_receiver:
                temp_kl_shallow = criterion_div(each, shallow_sender)
                loss_shallow_deep_.append(temp_kl_shallow)
                loss_csv.append(temp_kl_shallow.item())

    loss_ce_all = sum(loss_ce_)
    loss_csv.append(loss_ce_all.item())

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

    loss_csv.append(loss_kl_all.item())

    loss_self_supervised = loss_ce_all + loss_kl_all
    loss_csv.append(loss_self_supervised.item())

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
    # # a = torch.randn(2, 3, 32, 32)
    # # auxcf = inAuxCF(3, 3, 100)
    # # # print(torchinfo.summary(auxcf))
    # # a_auxcf = auxcf(a)
    # # print(a_auxcf.shape)
    #
    # a = torch.randn(2, 64, 32, 32)
    #
    # auxnet1 = inAuxCF3([64, 128, 256, 512])
    #
    # # feat, out = auxnet(a)
    # # print(feat.shape, out.shape)
    #
    # total1 = sum([param.nelement() for param in auxnet1.parameters()])
    #
    # print("Parmeter: %.2fM" % (total1 / 1e6))
    #
    # # print(auxnet)
    #
    # aenet1 = aeFusionConv2(512*4, 512, 512)
    # aenet2 = aeFusionConv2(512*4, 512, 512)
    # # ae_input = torch.randn(2, 512*3, 1, 1)
    # # rep, recon = aenet(ae_input)
    #
    # # print(rep.shape, recon.shape)
    #
    # total_ae1 = sum([param.nelement() for param in aenet1.parameters()])
    # total_ae2 = sum([param.nelement() for param in aenet2.parameters()])
    #
    # print("Parmeter: %.2fM" % (total_ae1 / 1e6))
    # print("Parmeter: %.2fM" % (total_ae2 / 1e6))
    # print("Parmeter: %.2fM" % ((total_ae1 + total_ae2) / 1e6))
    #
    # print(123)
    downsample_temp = nn.Sequential(
        conv1x1(64, 128, 2),
        nn.BatchNorm2d(128),
    )
    block = BasicBlockMy(64, 128, 2, downsample_temp)
    a = torch.randn(2, 64, 32, 32)
    block(a)


    s_reshape_list = [0, 1, 2, 3]
    deep_shallow = []
    shallow_deep = []

    for i in range(4 - 1):
        print('time ' + str(i))
        # deep - shallow
        deep_sender_idx = 4 - 1 - i
        deep_sender = s_reshape_list[deep_sender_idx]
        deep_receiver = s_reshape_list[:deep_sender_idx]
        for each in deep_receiver:
            str1 = 'sender:' + str(deep_sender) + ' -> receiver: ' + str(each)
            deep_shallow.append(str1)
            print(str1)
        # shallow - deep
        shallow_sender_idx = i
        shallow_sender = s_reshape_list[shallow_sender_idx]
        shallow_receiver = s_reshape_list[shallow_sender_idx+1:]
        for each in shallow_receiver:
            str2 = 'sender:' + str(shallow_sender) + ' -> receiver: ' + str(each)
            shallow_deep.append(str2)
            print(str2)

    print(deep_shallow[:3])


if __name__ == '__main__':
    main()