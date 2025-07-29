# -*- coding: utf-8 -*-
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from .CKA import linear_CKA_GPU, linear_CKA
import os
import numpy as np

__all__ = ["Our_FWD_Conv_122", "Our_FB_Conv_122"]

# Conv 的方法，用KDCL的思想，将 rep 和 fusion 用 均值 处理，再做蒸馏。

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
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_relu:
            return self.relu(self.bn(x))
        else:
            return self.bn(x)


class FWD_ConvT(nn.Module):
    """ forward 过程中教师经过卷积自编码器融合知识，适用于 MEAN_STD和 Channel_MEAN """

    def __init__(self, t_channel, s_channel):
        super(FWD_ConvT, self).__init__()
        self.en = nn.Conv1d(in_channels=t_channel, out_channels=s_channel, kernel_size=1, stride=1, padding=0,
                            bias=True)
        self.bn1 = nn.BatchNorm1d(s_channel)
        self.relu = nn.ReLU(inplace=True)
        self.de = nn.Conv1d(in_channels=s_channel, out_channels=t_channel, kernel_size=1, stride=1, padding=0,
                            bias=True)
        self.bn2 = nn.BatchNorm1d(t_channel)

    def forward(self, x):
        rep = self.relu(self.bn1(self.en(x)))
        recon = self.relu(self.bn2(self.de(rep)))
        return rep, recon

class FWD_ConvS(nn.Module):
    """ forward 过程中学生经过单层卷积融合知识，适用于 MEAN_STD和 Channel_MEAN """

    def __init__(self, s_channel):
        super(FWD_ConvS, self).__init__()
        self.fu = nn.Conv1d(in_channels=s_channel, out_channels=s_channel, kernel_size=1, stride=1, padding=0,
                            bias=True)
        self.bn1 = nn.BatchNorm1d(s_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        fusion = self.relu(self.bn1(self.fu(x)))
        return fusion

class FB_ConvT(nn.Module):
    """ feedback 过程中教师经过单层卷积层融合知识，适用于 MEAN_STD和 Channel_MEAN """

    def __init__(self, t_channel):
        super(FB_ConvT, self).__init__()
        self.fu = nn.Conv1d(in_channels=t_channel, out_channels=t_channel, kernel_size=1, stride=1, padding=0,
                            bias=True)
        self.bn1 = nn.BatchNorm1d(t_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        fusion = self.relu(self.bn1(self.fu(x)))
        return fusion

class FB_ConvS(nn.Module):
    """ feedback 过程中学生经过卷积自编码器融合知识，适用于 MEAN_STD和 Channel_MEAN """

    def __init__(self, t_channel, s_channel):
        super(FB_ConvS, self).__init__()
        self.en = nn.Conv1d(in_channels=s_channel, out_channels=t_channel, kernel_size=1, stride=1, padding=0,
                            bias=True)
        self.bn1 = nn.BatchNorm1d(t_channel)
        self.relu = nn.ReLU(inplace=True)
        self.de = nn.Conv1d(in_channels=t_channel, out_channels=s_channel, kernel_size=1, stride=1, padding=0,
                            bias=True)
        self.bn2 = nn.BatchNorm1d(s_channel)

    def forward(self, x):
        rep = self.relu(self.bn1(self.en(x)))
        recon = self.relu(self.bn2(self.de(rep)))
        return rep, recon

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
    if args.in_useOther:
        return F.normalize(x.pow(2).mean(1).view(x.shape[0], -1))
    else:
        return x.pow(2).mean(1).view(x.shape[0], -1)
    # return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))


class Our_FWD_Conv_122(nn.Module):
    """ 用于计算 forward 过程中的 中间特征层 的 loss """

    def __init__(self, T, args):
        super(Our_FWD_Conv_122, self).__init__()
        self.T = 1.0
        self.fwd_t = FWD_ConvT(args.t_shape, args.s_shape)
        self.fwd_s = FWD_ConvS(args.s_shape)
        if args.in_method == 'Channel_MEAN':
            self.method_in = channel_mean
        elif args.in_method == 'Attention':
            self.method_in = at
        elif 'MEAN_STD' in args.in_method:
            self.method_in = mean_std

        if 'KL' in args.in_criterion:
            self.T = T


    def forward(self, feat_t_fwd_list, feat_s_fwd_list, args):
        N, C, H, W = feat_t_fwd_list[0].shape
        t_reshape_list = []
        s_reshape_list = []
        for each_t in feat_t_fwd_list:
            t_reshape_list.append(self.method_in(each_t, args))

        for each_s in feat_s_fwd_list:
            s_reshape_list.append(self.method_in(each_s, args))

        t_cat = torch.cat(t_reshape_list, 1)
        s_cat = torch.cat(s_reshape_list, 1)

        t_rep, t_recon = self.fwd_t(t_cat)
        s_fusion = self.fwd_s(s_cat)

        t_recon_split = torch.split(t_recon, [each[1] for each in args.t_dim], dim=1)
        loss_recon_fwd = 0.0
        # 重构损失
        for each_t, each_recon in zip(t_reshape_list, t_recon_split):
            loss_recon_fwd += F.mse_loss(each_t, each_recon)

        # 用 KDCL 的思想将 rep 和 fusion 进行均值处理，再蒸馏。
        rep_fusion_aggregate = torch.zeros(size=(2, N, t_rep.shape[1], 2), dtype=torch.float).cuda()
        rep_fusion_aggregate[0, ...] = t_rep
        rep_fusion_aggregate[1, ...] = s_fusion
        rep_fusion_stable = rep_fusion_aggregate.mean(dim=0)
        rep_fusion_stable.detach()

        # 均值和学生的loss
        if args.in_criterion == 'MSE':
            # loss_intfwd = (t_rep - s_fusion).pow(2).mean()
            loss_fusion_fwd = F.mse_loss(rep_fusion_stable, s_fusion)
        elif args.in_criterion == 'MSE_normalize':
            rep_fusion_stable = F.normalize(rep_fusion_stable, dim=-1)
            s_fusion = F.normalize(s_fusion, dim=-1)
            loss_fusion_fwd = F.mse_loss(rep_fusion_stable, s_fusion)
        elif args.in_criterion == 'MSE_softmax':
            loss_fusion_fwd = F.mse_loss(F.softmax(rep_fusion_stable, dim=-1), F.softmax(s_fusion, dim=-1))
        elif args.in_criterion == 'KL_softmax':
            # 因为展开后算上 batchsize 是三维，需要用 sum 后再平均
            criterion_int_kl = nn.KLDivLoss(reduction='sum')
            loss_fusion_fwd = criterion_int_kl(F.log_softmax(s_fusion / self.T, dim=-1),
                                           F.softmax(rep_fusion_stable / self.T, dim=-1)) * (self.T * self.T) / (N * C)

        with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
            f.write("CH_Recon_fwd:" + str(loss_recon_fwd.item()) + '\t')
            f.write("CH_Fusion_fwd:" + str(loss_fusion_fwd.item()) + '\t')

        # loss_chfwd = loss_recon_fwd + loss_fusion_fwd
        loss_chfwd_dict = dict(recon_fwd=loss_recon_fwd, fusion_fwd=loss_fusion_fwd)
        return loss_chfwd_dict


class Our_FB_Conv_122(nn.Module):
    """ 用于计算 feedback 过程中的 中间特征层 的 loss """

    def __init__(self, T, args):
        super(Our_FB_Conv_122, self).__init__()
        self.T = 1.0
        self.fb_t = FB_ConvT(args.t_shape)
        self.fb_s = FB_ConvS(args.t_shape, args.s_shape)
        if args.in_method == 'Channel_MEAN':
            self.method_in = channel_mean
        elif args.in_method == 'Attention':
            self.method_in = at
        elif 'MEAN_STD' in args.in_method:
            self.method_in = mean_std

        if 'KL' in args.in_criterion:
            self.T = T

    def forward(self, feat_t_fb_list, feat_s_fb_list, args):
        N, C, H, W = feat_t_fb_list[0].shape
        t_reshape_list = []
        s_reshape_list = []
        for each_t in feat_t_fb_list:
            t_reshape_list.append(self.method_in(each_t, args))

        for each_s in feat_s_fb_list:
            s_reshape_list.append(self.method_in(each_s, args))

        t_cat = torch.cat(t_reshape_list, 1)
        s_cat = torch.cat(s_reshape_list, 1)

        t_fusion = self.fb_t(t_cat)
        s_rep, s_recon = self.fb_s(s_cat)

        s_recon_split = torch.split(s_recon, [each[1] for each in args.s_dim], dim=1)
        loss_recon_fb = 0.0
        # 重构损失
        for each_s, each_recon in zip(s_reshape_list, s_recon_split):
            loss_recon_fb += F.mse_loss(each_s, each_recon)

        # 用 KDCL 的思想将 rep 和 fusion 进行均值处理，再蒸馏。
        rep_fusion_aggregate = torch.zeros(size=(2, N, s_rep.shape[1], 2), dtype=torch.float).cuda()
        rep_fusion_aggregate[0, ...] = s_rep
        rep_fusion_aggregate[1, ...] = t_fusion
        rep_fusion_stable = rep_fusion_aggregate.mean(dim=0)
        rep_fusion_stable.detach()

        # 教师和均值的loss
        if args.in_criterion == 'MSE':
            # loss_intfwd = (t_rep - s_fusion).pow(2).mean()
            loss_fusion_fb = F.mse_loss(rep_fusion_stable, t_fusion)
        elif args.in_criterion == 'MSE_normalize':
            rep_fusion_stable = F.normalize(rep_fusion_stable, dim=-1)
            t_fusion = F.normalize(t_fusion, dim=-1)
            loss_fusion_fb = F.mse_loss(rep_fusion_stable, t_fusion)
        elif args.in_criterion == 'MSE_softmax':
            loss_fusion_fb = F.mse_loss(F.softmax(rep_fusion_stable, dim=-1), F.softmax(t_fusion, dim=-1))
        elif args.in_criterion == 'KL_softmax':
            # 因为展开后算上 batchsize 是三维，需要用 sum 后再平均
            criterion_int_kl = nn.KLDivLoss(reduction='sum')
            loss_fusion_fb = criterion_int_kl(F.log_softmax(t_fusion / self.T, dim=-1),
                                               F.softmax(rep_fusion_stable / self.T, dim=-1)) * (self.T * self.T) / (N * C)

        with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
            f.write("CH_Recon_fb:" + str(loss_recon_fb.item()) + '\t')
            f.write("CH_Fusion_fb:" + str(loss_fusion_fb.item()) + '\t')

        # loss_chfb = loss_recon_fb + loss_fusion_fb
        loss_chfb_dict = dict(recon_fb=loss_recon_fb, fusion_fb=loss_fusion_fb)
        return loss_chfb_dict

