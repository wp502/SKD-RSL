# -*- coding: utf-8 -*-
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from .CKA import linear_CKA_GPU, linear_CKA
import os
import numpy as np

__all__ = ["Our_FWD_Conv_15", "Our_FB_Conv_15"]

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
        self.relu = nn.ReLU(inplace=True)

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
        self.relu = nn.ReLU(inplace=True)
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
        self.relu = nn.ReLU(inplace=True)
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

class FeatAuxCF(nn.Module):
    """ 给AE的特征层加一个辅助分类器 """
    def __init__(self, feat_shape, num_classes):
        super(FeatAuxCF, self).__init__()
        self.fc = nn.Linear(feat_shape, num_classes)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


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


class Our_FWD_Conv_15(nn.Module):
    """ 用于计算 forward 过程中的 中间特征层 的 loss """

    def __init__(self, T, args):
        super(Our_FWD_Conv_15, self).__init__()
        self.T = 1.0
        self.fwd_t = AEFusionSender(args.t_shape, args.s_shape, args.ende_use_relu)
        self.fwd_s = AEFusionReceiver(args.s_shape, args.ende_use_relu)
        if 'MEAN_STD' in args.in_method:
            self.fwd_auxcf = FeatAuxCF(args.s_shape*2, args.num_class)
        else:
            self.fwd_auxcf = FeatAuxCF(args.s_shape, args.num_class)
        if 'CHMEAN' in args.in_method:
            self.method_in = channel_mean
        elif 'AT' in args.in_method:
            self.method_in = at
        elif 'MEAN_STD' in args.in_method:
            self.method_in = mean_std

        if 'KL' in args.in_criterion or args.in_criterion == 'MSE_softmax_T':
            self.T = T


    def forward(self, feat_t_fwd_list, feat_s_fwd_list, fwd_labels, args):
        N, C, H, W = feat_t_fwd_list[0].shape
        t_reshape_list = []
        s_reshape_list = []
        for each_t in feat_t_fwd_list:
            t_reshape_list.append(self.method_in(each_t, args))

        for each_s in feat_s_fwd_list:
            s_reshape_list.append(self.method_in(each_s, args))

        t_cat = torch.cat(t_reshape_list, 1)
        s_cat = torch.cat(s_reshape_list, 1)

        t_fusion, t_recon = self.fwd_t(t_cat)
        s_fusion, s_recon = self.fwd_s(s_cat)

        s_aux_CF = self.fwd_auxcf(s_fusion)
        aux_loss_fwd = nn.CrossEntropyLoss()(s_aux_CF, fwd_labels)

        if 'CHMEAN' in args.in_method or 'MEAN_STD' in args.in_method:
            t_recon_split = torch.split(t_recon, [each[1] for each in args.t_dim], dim=1)
            s_recon_split = torch.split(s_recon, [each[1] for each in args.s_dim], dim=1)
        elif 'AT' in args.in_method:
            t_recon_split = torch.split(t_recon, [each[2]*each[3] for each in args.t_dim], dim=1)
            s_recon_split = torch.split(s_recon, [each[2]*each[3] for each in args.s_dim], dim=1)
        loss_recon_fwd_t = 0.0
        loss_recon_fwd_s = 0.0
        # 重构损失
        for each_t, each_recon in zip(t_reshape_list, t_recon_split):
            loss_recon_fwd_t += F.mse_loss(each_t, each_recon)
        for each_s, each_recon in zip(s_reshape_list, s_recon_split):
            loss_recon_fwd_s += F.mse_loss(each_s, each_recon)
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
            # 因为展开后算上 batchsize 是三维，需要用 sum 后再平均
            criterion_int_kl = nn.KLDivLoss(reduction='sum')
            loss_fusion_fwd = criterion_int_kl(F.log_softmax(s_fusion / self.T, dim=-1),
                                           F.softmax(t_fusion / self.T, dim=-1)) * (self.T * self.T) / (N * C)

        with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
            f.write("CH_Recon_T_fwd:" + str(loss_recon_fwd_t.item()) + '\t')
            f.write("CH_Recon_S_fwd:" + str(loss_recon_fwd_s.item()) + '\t')
            f.write("CH_Fusion_fwd:" + str(loss_fusion_fwd.item()) + '\t')

        # loss_chfwd = loss_recon_fwd + loss_fusion_fwd
        loss_chfwd_dict = dict(recon_T_fwd=loss_recon_fwd_t, recon_S_fwd=loss_recon_fwd_s, fusion_fwd=loss_fusion_fwd, auxCF_fwd=aux_loss_fwd)
        return loss_chfwd_dict


class Our_FB_Conv_15(nn.Module):
    """ 用于计算 feedback 过程中的 中间特征层 的 loss """

    def __init__(self, T, args):
        super(Our_FB_Conv_15, self).__init__()
        self.T = 1.0
        self.fb_t = AEFusionReceiver(args.t_shape, args.ende_use_relu)
        self.fb_s = AEFusionSender(args.s_shape, args.t_shape, args.ende_use_relu)
        if 'MEAN_STD' in args.in_method:
            self.fb_auxcf = FeatAuxCF(args.t_shape*2, args.num_class)
        else:
            self.fb_auxcf = FeatAuxCF(args.t_shape, args.num_class)
        if 'CHMEAN' in args.in_method:
            self.method_in = channel_mean
        elif 'AT' in args.in_method:
            self.method_in = at
        elif 'MEAN_STD' in args.in_method:
            self.method_in = mean_std

        if 'KL' in args.in_criterion or args.in_criterion == 'MSE_softmax_T':
            self.T = T

    def forward(self, feat_t_fb_list, feat_s_fb_list, fb_labels, args):
        N, C, H, W = feat_t_fb_list[0].shape
        t_reshape_list = []
        s_reshape_list = []
        for each_t in feat_t_fb_list:
            t_reshape_list.append(self.method_in(each_t, args))

        for each_s in feat_s_fb_list:
            s_reshape_list.append(self.method_in(each_s, args))

        t_cat = torch.cat(t_reshape_list, 1)
        s_cat = torch.cat(s_reshape_list, 1)

        t_fusion, t_recon = self.fb_t(t_cat)
        s_fusion, s_recon = self.fb_s(s_cat)

        t_aux_CF = self.fb_auxcf(t_fusion)
        aux_loss_fb = nn.CrossEntropyLoss()(t_aux_CF, fb_labels)

        # s_recon_split = torch.split(s_recon, [each[1] for each in args.s_dim], dim=1)
        # t_recon_split = torch.split(t_recon, [each[1] for each in args.t_dim], dim=1)
        if 'CHMEAN' in args.in_method or 'MEAN_STD' in args.in_method:
            t_recon_split = torch.split(t_recon, [each[1] for each in args.t_dim], dim=1)
            s_recon_split = torch.split(s_recon, [each[1] for each in args.s_dim], dim=1)
        elif 'AT' in args.in_method:
            t_recon_split = torch.split(t_recon, [each[2]*each[3] for each in args.t_dim], dim=1)
            s_recon_split = torch.split(s_recon, [each[2]*each[3] for each in args.s_dim], dim=1)

        loss_recon_fb_s = 0.0
        loss_recon_fb_t = 0.0

        # 重构损失
        for each_s, each_recon in zip(s_reshape_list, s_recon_split):
            loss_recon_fb_s += F.mse_loss(each_s, each_recon)

        for each_t, each_recon in zip(t_reshape_list, t_recon_split):
            loss_recon_fb_t += F.mse_loss(each_t, each_recon)
        # 教师和学生的loss
        if args.in_criterion == 'MSE':
            # loss_intfwd = (t_rep - s_fusion).pow(2).mean()
            loss_fusion_fb = F.mse_loss(s_fusion, t_fusion)
        elif args.in_criterion == 'MSE_normalize':
            s_rep = F.normalize(s_fusion, dim=-1)
            t_fusion = F.normalize(t_fusion, dim=-1)
            loss_fusion_fb = F.mse_loss(s_rep, t_fusion)
        elif args.in_criterion == 'MSE_softmax':
            loss_fusion_fb = F.mse_loss(F.softmax(s_fusion, dim=-1), F.softmax(t_fusion, dim=-1))
        elif args.in_criterion == 'MSE_softmax_T':
            loss_fusion_fb = F.mse_loss(F.softmax(s_fusion / self.T, dim=-1), F.softmax(t_fusion / self.T, dim=-1)) * (self.T * self.T)
        elif args.in_criterion == 'KL_softmax':
            # 因为展开后算上 batchsize 是三维，需要用 sum 后再平均
            criterion_int_kl = nn.KLDivLoss(reduction='sum')
            loss_fusion_fb = criterion_int_kl(F.log_softmax(t_fusion / self.T, dim=-1),
                                               F.softmax(s_fusion / self.T, dim=-1)) * (self.T * self.T) / (N * C)

        with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
            f.write("CH_Recon_T_fb:" + str(loss_recon_fb_t.item()) + '\t')
            f.write("CH_Recon_S_fb:" + str(loss_recon_fb_s.item()) + '\t')
            f.write("CH_Fusion_fb:" + str(loss_fusion_fb.item()) + '\t')

        # loss_chfb = loss_recon_fb + loss_fusion_fb
        loss_chfb_dict = dict(recon_T_fb=loss_recon_fb_t, recon_S_fb=loss_recon_fb_s, fusion_fb=loss_fusion_fb, auxCF_fb=aux_loss_fb)
        return loss_chfb_dict

