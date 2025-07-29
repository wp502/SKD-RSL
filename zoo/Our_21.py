# -*- coding: utf-8 -*-
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from .CKA import linear_CKA_GPU, linear_CKA
import os
import numpy as np

__all__ = ["FCDecoder", "ConvReg", "OurLoss_FWD", "OurLoss_FB"]


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


class FCDecoder(nn.Module):
    """
    fully connect Decoder
    """

    def __init__(self, kf_shape, output_shape, use_relu=True):
        super(FCDecoder, self).__init__()
        self.use_relu = use_relu
        self.fc = nn.Linear(kf_shape, output_shape)
        self.bn = nn.BatchNorm1d(output_shape)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc(x)
        if self.use_relu:
            return self.relu(self.bn(x))
        else:
            return self.bn(x)

class OurLoss_FKF_FWD(nn.Module):
    """forward 中用于单独更新 FKF 模块的类"""

    def __init__(self, T, args):
        super(OurLoss_FKF_FWD, self).__init__()
        self.T = T
        self.FKF_fwd = torch.randn(args.fkf_avg_size, dtype=torch.float).cuda()  # 特征融合模块
        self.de_fwd_toT_1 = FCDecoder(self.FKF_fwd.shape, args.t_dim[0], use_relu=args.decode_is_relu)
        self.de_fwd_toT_2 = FCDecoder(self.FKF_fwd.shape, args.t_dim[1], use_relu=args.decode_is_relu)
        self.de_fwd_toT_3 = FCDecoder(self.FKF_fwd.shape, args.t_dim[2], use_relu=args.decode_is_relu)

    def forward(self, feat_t, feat_s, args, is_feedback=False):
        pass

class OurLoss_FWD(nn.Module):
    """forward 中用于更新除 FKF 模块以外其他部分的类"""

    def __init__(self, T, args):
        super(OurLoss_FWD, self).__init__()
        self.T = T
        self.de_fwd_toS_1 = FCDecoder(self.FKF_fwd.shape, args.s_dim[0], use_relu=args.decode_is_relu)
        self.de_fwd_toS_2 = FCDecoder(self.FKF_fwd.shape, args.s_dim[1], use_relu=args.decode_is_relu)
        self.de_fwd_toS_3 = FCDecoder(self.FKF_fwd.shape, args.s_dim[2], use_relu=args.decode_is_relu)

    def forward(self, feat_t, feat_s, args, is_feedback=False):
        pass


class OurLoss_FKF_FB(nn.Module):
    """feedback 中用于单独更新 FKF 模块的类"""

    def __init__(self, T, args):
        super(OurLoss_FKF_FB, self).__init__()
        self.T = T
        self.FKF_fb = nn.Parameter(torch.randn(args.fkf_avg_size, dtype=torch.float).cuda())  # 特征融合模块
        self.de_fb_toS_1 = FCDecoder(self.FKF_fb.shape, args.s_dim[0], use_relu=args.decode_is_relu)
        self.de_fb_toS_2 = FCDecoder(self.FKF_fb.shape, args.s_dim[1], use_relu=args.decode_is_relu)
        self.de_fb_toS_3 = FCDecoder(self.FKF_fb.shape, args.s_dim[2], use_relu=args.decode_is_relu)

    def forward(self, feat_t, feat_s, args, is_feedback=True):
        pass


class OurLoss_FB(nn.Module):
    """feedback 中用于更新除 FKF 模块以外其他部分的类"""

    def __init__(self, T, args):
        super(OurLoss_FB, self).__init__()
        self.T = T
        self.de_fb_toT_1 = FCDecoder(self.FKF_fb.shape, args.t_dim[0], use_relu=args.decode_is_relu)
        self.de_fb_toT_2 = FCDecoder(self.FKF_fb.shape, args.t_dim[1], use_relu=args.decode_is_relu)
        self.de_fb_toT_3 = FCDecoder(self.FKF_fb.shape, args.t_dim[2], use_relu=args.decode_is_relu)

    def forward(self, feat_t, feat_s, args, is_feedback=True):
        pass