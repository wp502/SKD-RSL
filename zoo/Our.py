# -*- coding: utf-8 -*-
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from .CKA import linear_CKA_GPU, linear_CKA
import os
import numpy as np

__all__ = ["Our_FWD", "Our_FB"]

# Linear的方法

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


class FWD_tTOs(nn.Module):
    """ forward 过程中教师经过两层全连接变为学生的大小 （教师向学生兼容）"""

    def __init__(self, t_shape, s_shape, use_relu=True):
        super(FWD_tTOs, self).__init__()
        self.use_relu = use_relu
        self.fc_1 = nn.Linear(t_shape, t_shape)
        self.bn_1 = nn.BatchNorm1d(t_shape)
        self.relu = nn.ReLU(inplace=True)
        self.fc_2 = nn.Linear(t_shape, s_shape)
        self.bn_2 = nn.BatchNorm1d(s_shape)

    def forward(self, x):
        x = self.fc_1(x)
        x = self.relu(self.bn_1(x))
        x = self.fc_2(x)
        if self.use_relu:
            return self.relu(self.bn_2(x))
        else:
            return self.bn_2(x)


class FWD_sTOs(nn.Module):
    """ forward 过程中学生经过一层全连接融合知识 （教师向学生兼容）"""

    def __init__(self, t_shape, s_shape, use_relu=True):
        super(FWD_sTOs, self).__init__()
        self.use_relu = use_relu
        self.fc_1 = nn.Linear(s_shape, s_shape)
        self.bn_1 = nn.BatchNorm1d(s_shape)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc_1(x)
        if self.use_relu:
            return self.relu(self.bn_1(x))
        else:
            return self.bn_1(x)


class FWD_tTOt(nn.Module):
    """ forward 过程中教师经过一层全连接融合知识 （学生向教师扩大）"""

    def __init__(self, t_shape, s_shape, use_relu=True):
        super(FWD_tTOt, self).__init__()
        self.use_relu = use_relu
        self.fc_1 = nn.Linear(t_shape, t_shape)
        self.bn_1 = nn.BatchNorm1d(t_shape)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc_1(x)
        if self.use_relu:
            return self.relu(self.bn_1(x))
        else:
            return self.bn_1(x)


class FWD_sTOt(nn.Module):
    """ forward 过程中学生经过两层全连接变为教师的大小 （学生向教师扩大）"""

    def __init__(self, t_shape, s_shape, use_relu=True):
        super(FWD_sTOt, self).__init__()
        self.use_relu = use_relu
        self.fc_1 = nn.Linear(s_shape, s_shape)
        self.bn_1 = nn.BatchNorm1d(s_shape)
        self.relu = nn.ReLU(inplace=True)
        self.fc_2 = nn.Linear(s_shape, t_shape)
        self.bn_2 = nn.BatchNorm1d(t_shape)

    def forward(self, x):
        x = self.fc_1(x)
        x = self.relu(self.bn_1(x))
        x = self.fc_2(x)
        if self.use_relu:
            return self.relu(self.bn_2(x))
        else:
            return self.bn_2(x)


class FB_tTOs(nn.Module):
    """ feedback 过程中教师经过一层全连接融合知识 """

    def __init__(self, t_shape, s_shape, use_relu=True):
        super(FB_tTOs, self).__init__()
        self.use_relu = use_relu
        self.fc_1 = nn.Linear(t_shape, t_shape)
        self.bn_1 = nn.BatchNorm1d(t_shape)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc_1(x)
        if self.use_relu:
            return self.relu(self.bn_1(x))
        else:
            return self.bn_1(x)


class FB_sTOt(nn.Module):
    """ feedback 过程中学生经过两层全连接变为教师的大小 """

    def __init__(self, t_shape, s_shape, use_relu=True):
        super(FB_sTOt, self).__init__()
        self.use_relu = use_relu
        self.fc_1 = nn.Linear(s_shape, s_shape)
        self.bn_1 = nn.BatchNorm1d(s_shape)
        self.relu = nn.ReLU(inplace=True)
        self.fc_2 = nn.Linear(s_shape, t_shape)
        self.bn_2 = nn.BatchNorm1d(t_shape)

    def forward(self, x):
        x = self.fc_1(x)
        x = self.relu(self.bn_1(x))
        x = self.fc_2(x)
        if self.use_relu:
            return self.relu(self.bn_2(x))
        else:
            return self.bn_2(x)

def selfAT_c(x, args):
    B, C, H, W = x.shape
    x_reshape = x.reshape(B, C, -1)
    if args.in_useOther:
        x_reshape = F.softmax(x_reshape/args.T, dim=-1)
    x_transpose = x_reshape.transpose(1, 2)
    return torch.matmul(x_reshape, x_transpose).reshape(B, -1)

def selfAT_hw(x, args):
    B, C, H, W = x.shape
    x_reshape = x.reshape(B, C, -1)
    if args.in_useOther:
        x_reshape = F.softmax(x_reshape/args.T, dim=-1)
    x_transpose = x_reshape.transpose(1, 2)
    return torch.matmul(x_transpose, x_reshape).reshape(B, -1)

def mean_std(x, args):
    B, C, H, W = x.shape
    if args.in_useOther:
        mean_x = F.normalize(x.reshape(B, C, -1).mean(2))
        std_x = F.normalize(x.reshape(B, C, -1).std(2))
    else:
        mean_x = x.reshape(B, C, -1).mean(2)
        std_x = x.reshape(B, C, -1).std(2)
    mean_std_x = torch.cat([mean_x, std_x], 1)
    return mean_std_x

def channel_mean(x, args):
    return x.reshape(x.shape[0], x.shape[1], -1).mean(2)

def at(x, args):
    if args.in_useOther:
        return F.normalize(x.pow(2).mean(1).view(x.shape[0], -1))
    else:
        return x.pow(2).mean(1).view(x.shape[0], -1)
    # return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))


class Our_FWD(nn.Module):
    """ 用于计算 forward 过程中的 中间特征层 的 loss """

    def __init__(self, T, args):
        super(Our_FWD, self).__init__()
        self.T = 1.0
        if args.fwd_in_TtoS:
            self.fwd_t = FWD_tTOs(args.t_shape, args.s_shape, use_relu=args.ende_is_relu)
            self.fwd_s = FWD_sTOs(args.t_shape, args.s_shape, use_relu=args.ende_is_relu)
        else:
            self.fwd_t = FWD_tTOt(args.t_shape, args.s_shape, use_relu=args.ende_is_relu)
            self.fwd_s = FWD_sTOt(args.t_shape, args.s_shape, use_relu=args.ende_is_relu)
        if args.in_method == 'Channel_MEAN':
            self.method_in = channel_mean
        elif args.in_method == 'Attention':
            self.method_in = at
        elif args.in_method == 'MEAN_STD':
            self.method_in = mean_std
        elif args.in_method == 'SELF_AT_HW':
            self.method_in = selfAT_hw
        elif args.in_method == 'SELF_AT_C':
            self.method_in = selfAT_c
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

        t_output = self.fwd_t(t_cat)
        s_output = self.fwd_s(s_cat)

        # criterion_intfwd = nn.MSELoss(reduction='sum').cuda()
        # loss_intfwd = criterion_intfwd(tTOs_output, sTOt_output) / batch_size_now

        if args.in_criterion == 'MSE':
            loss_intfwd = (t_output - s_output).pow(2).mean()
        elif args.in_criterion == 'MSE_normalize':
            tTOs_output = F.normalize(t_output)
            sTOt_output = F.normalize(s_output)
            loss_intfwd = (tTOs_output - sTOt_output).pow(2).mean()
        elif args.in_criterion == 'MSE_softmax':
            # criterion_int_mse = nn.MSELoss(reduction='sum')
            # loss_intfwd = criterion_int_mse(F.softmax(sTOt_output, dim=-1), F.softmax(tTOs_output, dim=-1)) / batch_size_now
            criterion_int_mse = nn.MSELoss(reduction='mean')
            loss_intfwd = criterion_int_mse(F.softmax(s_output, dim=-1), F.softmax(t_output, dim=-1))
            # loss_intfwd = (F.softmax(sTOt_output, dim=-1), F.softmax(tTOs_output, dim=-1)).pow(2).mean()
        elif args.in_criterion == 'KL_softmax':
            # 因为展开后算上 batchsize 是二维，用 batchmean 即可，也可以使用 sum
            # way1
            criterion_int_kl = nn.KLDivLoss(reduction='batchmean')
            loss_intfwd = criterion_int_kl(F.log_softmax(s_output / self.T, dim=-1),
                                           F.softmax(t_output / self.T, dim=-1)) * (self.T * self.T)
            # way2
            # criterion_int_kl = nn.KLDivLoss(reduction='sum')
            # loss_intfwd = criterion_int_kl(F.log_softmax(s_output / self.T, dim=-1),
            #                                F.softmax(t_output / self.T, dim=-1)) * (self.T * self.T) / N
        return loss_intfwd


class Our_FB(nn.Module):
    """ 用于计算 feedback 过程中的 中间特征层 的 loss """

    def __init__(self, T, args):
        super(Our_FB, self).__init__()
        self.T = 1.0
        self.fb_t = FB_tTOs(args.t_shape, args.s_shape, use_relu=args.ende_is_relu)
        self.fb_s = FB_sTOt(args.t_shape, args.s_shape, use_relu=args.ende_is_relu)
        if args.in_method == 'Channel_MEAN':
            self.method_in = channel_mean
        elif args.in_method == 'Attention':
            self.method_in = at
        elif args.in_method == 'MEAN_STD':
            self.method_in = mean_std
        elif args.in_method == 'SELF_AT_HW':
            self.method_in = selfAT_hw
        elif args.in_method == 'SELF_AT_C':
            self.method_in = selfAT_c
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

        t_output = self.fb_t(t_cat)
        s_output = self.fb_s(s_cat)

        # criterion_intfwd = nn.MSELoss(reduction='sum').cuda()
        # loss_intfwd = criterion_intfwd(tTOs_output, sTOt_output) / batch_size_now

        if args.in_criterion == 'MSE':
            loss_intfb = (t_output - s_output).pow(2).mean()
        elif args.in_criterion == 'MSE_normalize':
            t_output = F.normalize(t_output)
            s_output = F.normalize(s_output)
            loss_intfb = (t_output - s_output).pow(2).mean()
        elif args.in_criterion == 'MSE_softmax':
            # criterion_int_mse = nn.MSELoss(reduction='sum')
            # loss_intfwd = criterion_int_mse(F.softmax(sTOt_output, dim=-1), F.softmax(tTOs_output, dim=-1)) / batch_size_now
            criterion_int_mse = nn.MSELoss(reduction='mean')
            loss_intfb = criterion_int_mse(F.softmax(s_output, dim=-1), F.softmax(t_output, dim=-1))
            # loss_intfwd = (F.softmax(sTOt_output, dim=-1), F.softmax(tTOs_output, dim=-1)).pow(2).mean()
        elif args.in_criterion == 'KL_softmax':
            # 因为展开后算上 batchsize 是二维，用 batchmean 即可，也可以使用 sum
            # way1
            criterion_int_kl = nn.KLDivLoss(reduction='batchmean')
            loss_intfb = criterion_int_kl(F.log_softmax(t_output / self.T, dim=-1),
                                           F.softmax(s_output / self.T, dim=-1)) * (self.T * self.T)
            # way2
            # criterion_int_kl = nn.KLDivLoss(reduction='sum')
            # loss_intfb = criterion_int_kl(F.log_softmax(t_output / self.T, dim=-1),
            #                                F.softmax(s_output / self.T, dim=-1)) * (self.T * self.T) / N
        return loss_intfb

