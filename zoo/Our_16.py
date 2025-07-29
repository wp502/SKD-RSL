# -*- coding: utf-8 -*-
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

__all__ = ["Our_FWD_Conv_16", "Our_FB_Conv_16"]

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


class aeFusionLinear(nn.Module):
    """ 经过 inAuxCF 后的特征，经过 Linear AE """

    def __init__(self, in_dim, fusion_dim, use_relu=False):
        super(aeFusionLinear, self).__init__()
        self.use_relu = use_relu
        self.en = nn.Linear(in_dim, fusion_dim)
        self.bn1 = nn.BatchNorm1d(fusion_dim)
        self.de = nn.Linear(fusion_dim, in_dim)
        self.bn2 = nn.BatchNorm1d(in_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        fusion = self.relu(self.bn1(self.en(x)))
        recon = self.bn2(self.de(fusion))
        if self.use_relu:
            recon = self.relu(recon)

        return fusion, recon

class aeFusionConv(nn.Module):
    """ 经过 inAuxCF 后的特征，经过 Conv AE """

    def __init__(self, in_dim, fusion_dim, use_relu=False):
        super(aeFusionConv, self).__init__()
        self.use_relu = use_relu
        self.en = nn.Conv1d(in_channels=in_dim, out_channels=fusion_dim, kernel_size=1, stride=1,
                            padding=0, bias=True)
        self.bn1 = nn.BatchNorm1d(fusion_dim)
        self.relu = nn.ReLU(inplace=True)
        self.de = nn.Conv1d(in_channels=fusion_dim, out_channels=in_dim, kernel_size=1, stride=1,
                            padding=0, bias=True)
        self.bn2 = nn.BatchNorm1d(in_dim)

    def forward(self, x):
        rep = self.relu(self.bn1(self.en(x)))

        x_de = self.bn2(self.de(rep))
        if self.use_relu:
            recon = self.relu(x_de)
        else:
            recon = x_de
        return rep, recon

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
        out = self.linear(out)

        return out

class BasicBlockMy(nn.Module):
    def __init__(self, planes):
        super(BasicBlockMy, self).__init__()
        self.conv1 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

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

    def forward(self, x, is_feat=False):
        x = self.basic_block_my(x)

        out = self.avgpool(x)
        out = out.view(out.size(0), -1)
        feat = out
        out = self.linear(out)

        if is_feat:
            return feat, out
        else:
            return out


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


class Our_FWD_Conv_16(nn.Module):
    """ 用于计算 forward 过程中的 中间特征层 的 loss """

    def __init__(self, T, args):
        super(Our_FWD_Conv_16, self).__init__()
        self.T = 1.0
        # self.fwd_ae_t = AEFusionSender(args.t_shape, args.s_shape, args.ende_use_relu)
        # self.fwd_ae_s = AEFusionReceiver(args.s_shape, args.ende_use_relu)
        self.in_dim = args.num_class * (args.blocks_amount+1)
        self.rep_dim = args.num_class
        self.use_relu = args.ende_use_relu
        if args.fusion_method_AUXCF == 'AELinear':
            self.fwd_ae_t = aeFusionLinear(self.in_dim, self.rep_dim, use_relu=self.use_relu)
            self.fwd_ae_s = aeFusionLinear(self.in_dim, self.rep_dim, use_relu=self.use_relu)
        elif args.fusion_method_AUXCF == 'AEConv':
            self.fwd_ae_t = aeFusionConv(self.in_dim, self.rep_dim, use_relu=self.use_relu)
            self.fwd_ae_s = aeFusionConv(self.in_dim, self.rep_dim, use_relu=self.use_relu)
        self.fwd_auxCF_t = {}
        self.fwd_auxCF_s = {}
        for i in range(args.blocks_amount):
            self.fwd_auxCF_t.update({str(i+1): inAuxCF(args.t_dim[i][1], args.auxCFAmount, args.num_class)})
        for i in range(args.blocks_amount):
            self.fwd_auxCF_s.update({str(i+1): inAuxCF(args.s_dim[i][1], args.auxCFAmount, args.num_class)})

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
        t_reshape_list = []
        s_reshape_list = []

        for i in range(args.blocks_amount):
            t_reshape_list.append(self.fwd_auxCF_t[str(i+1)](feat_t_fwd_list[i]))
            s_reshape_list.append(self.fwd_auxCF_s[str(i+1)](feat_s_fwd_list[i]))

        t_reshape_list.append(output_t_fwd)
        s_reshape_list.append(output_s_fwd)

        s_reshape_list = [each.float() for each in s_reshape_list]

        loss_ = []
        for s_1 in s_reshape_list:
            temp_ce = criterion_cls(s_1, fwd_labels)
            loss_.append(temp_ce)
            args.loss_csv_fwd.append(temp_ce.item())
            for s_2 in s_reshape_list:
                if s_1 is not s_2:
                    temp_kl = criterion_div(s_2, s_1)
                    loss_.append(temp_kl)
                    args.loss_csv_fwd.append(temp_kl.item())

        loss_self_supervised_fwd = sum(loss_)
        args.loss_csv_fwd.append(loss_self_supervised_fwd.item())

        if args.fusion_method_AUXCF == 'AEConv':
            for idx, each in enumerate(t_reshape_list):
                B_temp, C_temp = each.shape
                t_reshape_list[idx] = t_reshape_list[idx].reshape(B_temp, C_temp, 1)
            for idx, each in enumerate(s_reshape_list):
                B_temp, C_temp = each.shape
                s_reshape_list[idx] = s_reshape_list[idx].reshape(B_temp, C_temp, 1)

        t_cat = torch.cat(t_reshape_list, 1)
        s_cat = torch.cat(s_reshape_list, 1)

        t_fusion, t_recon = self.fwd_ae_t(t_cat)
        s_fusion, s_recon = self.fwd_ae_s(s_cat)

        t_recon_split = torch.split(t_recon, [args.num_class] * (args.blocks_amount+1), dim=1)
        s_recon_split = torch.split(s_recon, [args.num_class] * (args.blocks_amount+1), dim=1)

        loss_recon_fwd_t = 0.0
        loss_recon_fwd_s = 0.0
        # 重构损失
        for each_t, each_recon in zip(t_reshape_list, t_recon_split):
            loss_recon_fwd_t += F.mse_loss(each_t, each_recon)
        for each_s, each_recon in zip(s_reshape_list, s_recon_split):
            loss_recon_fwd_s += F.mse_loss(each_s, each_recon)

        args.loss_csv_fwd.append(loss_recon_fwd_t.item())
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
            # 因为展开后算上 batchsize 是三维，需要用 sum 后再平均
            criterion_int_kl = nn.KLDivLoss(reduction='sum')
            loss_fusion_fwd = criterion_int_kl(F.log_softmax(s_fusion / self.T, dim=-1),
                                           F.softmax(t_fusion / self.T, dim=-1)) * (self.T * self.T) / (N * C)

        # with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
        #     f.write("CH_Recon_T_fwd:" + str(loss_recon_fwd_t.item()) + '\t')
        #     f.write("CH_Recon_S_fwd:" + str(loss_recon_fwd_s.item()) + '\t')
        #     f.write("CH_Fusion_fwd:" + str(loss_fusion_fwd.item()) + '\t')
        #     f.write("Self-Supervised_fwd:" + str(loss_self_supervised_fwd.item()) + '\t')

        args.loss_csv_fwd.append(loss_fusion_fwd.item())
        # loss_chfwd = loss_recon_fwd + loss_fusion_fwd
        loss_chfwd_dict = dict(recon_T_fwd=loss_recon_fwd_t, recon_S_fwd=loss_recon_fwd_s, fusion_fwd=loss_fusion_fwd, self_supervised_fwd=loss_self_supervised_fwd)
        return loss_chfwd_dict


class Our_FB_Conv_16(nn.Module):
    """ 用于计算 feedback 过程中的 中间特征层 的 loss """

    def __init__(self, T, args):
        super(Our_FB_Conv_16, self).__init__()
        self.T = 1.0
        # self.fb_ae_t = AEFusionSender(args.t_shape, args.s_shape, args.ende_use_relu)
        # self.fb_ae_s = AEFusionReceiver(args.s_shape, args.ende_use_relu)
        self.in_dim = args.num_class * (args.blocks_amount+1)
        self.rep_dim = args.num_class
        self.use_relu = args.ende_use_relu
        if args.fusion_method_AUXCF == 'AELinear':
            self.fb_ae_t = aeFusionLinear(self.in_dim, self.rep_dim, use_relu=self.use_relu)
            self.fb_ae_s = aeFusionLinear(self.in_dim, self.rep_dim, use_relu=self.use_relu)
        elif args.fusion_method_AUXCF == 'AEConv':
            self.fb_ae_t = aeFusionConv(self.in_dim, self.rep_dim, use_relu=self.use_relu)
            self.fb_ae_s = aeFusionConv(self.in_dim, self.rep_dim, use_relu=self.use_relu)
        # self.fb_ae_t = aeFusionLinear(, args.num_class, use_relu=args.ende_use_relu)
        # self.fb_ae_s = aeFusionLinear(args.num_class * (args.blocks_amount+1), args.num_class, use_relu=args.ende_use_relu)
        self.fb_auxCF_t = {}
        self.fb_auxCF_s = {}
        for i in range(args.blocks_amount):
            self.fb_auxCF_t.update({str(i + 1): inAuxCF(args.t_dim[i][1], args.auxCFAmount, args.num_class)})
        for i in range(args.blocks_amount):
            self.fb_auxCF_s.update({str(i + 1): inAuxCF(args.s_dim[i][1], args.auxCFAmount, args.num_class)})

        # if 'CHMEAN' in args.in_method:
        #     self.method_in = channel_mean
        # elif 'AT' in args.in_method:
        #     self.method_in = at
        # elif 'MEAN_STD' in args.in_method:
        #     self.method_in = mean_std

        if 'KL' in args.in_criterion or args.in_criterion == 'MSE_softmax_T':
            self.T = T

    def forward(self, feat_t_fb_list, feat_s_fb_list, output_t_fb, output_s_fb, fb_labels, criterion_cls, criterion_div, model_t_weights, args):
        N, C, H, W = feat_t_fb_list[0].shape
        t_reshape_list = []
        s_reshape_list = []
        # for each_t in feat_t_fb_list:
        #     t_reshape_list.append(self.method_in(each_t, args))
        #
        # for each_s in feat_s_fb_list:
        #     s_reshape_list.append(self.method_in(each_s, args))

        for i in range(args.blocks_amount):
            t_reshape_list.append(self.fb_auxCF_t[str(i + 1)](feat_t_fb_list[i]))
            s_reshape_list.append(self.fb_auxCF_s[str(i + 1)](feat_s_fb_list[i]))

        t_reshape_list.append(output_t_fb)
        s_reshape_list.append(output_s_fb)

        t_reshape_list = [each.float() for each in t_reshape_list]

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
                    temp_kl = criterion_div(t_2, t_1)
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

        if args.fusion_method_AUXCF == 'AEConv':
            for idx, each in enumerate(t_reshape_list):
                B_temp, C_temp = each.shape
                t_reshape_list[idx] = t_reshape_list[idx].reshape(B_temp, C_temp, 1)
            for idx, each in enumerate(s_reshape_list):
                B_temp, C_temp = each.shape
                s_reshape_list[idx] = s_reshape_list[idx].reshape(B_temp, C_temp, 1)

        t_cat = torch.cat(t_reshape_list, 1)
        s_cat = torch.cat(s_reshape_list, 1)

        t_fusion, t_recon = self.fb_ae_t(t_cat)
        s_fusion, s_recon = self.fb_ae_s(s_cat)

        t_recon_split = torch.split(t_recon, [args.num_class] * (args.blocks_amount+1), dim=1)
        s_recon_split = torch.split(s_recon, [args.num_class] * (args.blocks_amount+1), dim=1)

        # if 'CHMEAN' in args.in_method or 'MEAN_STD' in args.in_method:
        #     t_recon_split = torch.split(t_recon, [each[1] for each in args.t_dim], dim=1)
        #     s_recon_split = torch.split(s_recon, [each[1] for each in args.s_dim], dim=1)
        # elif 'AT' in args.in_method:
        #     t_recon_split = torch.split(t_recon, [each[2]*each[3] for each in args.t_dim], dim=1)
        #     s_recon_split = torch.split(s_recon, [each[2]*each[3] for each in args.s_dim], dim=1)
        loss_recon_fb_t = 0.0
        loss_recon_fb_s = 0.0
        # 重构损失
        for each_t, each_recon in zip(t_reshape_list, t_recon_split):
            loss_recon_fb_t += F.mse_loss(each_t, each_recon)
        for each_s, each_recon in zip(s_reshape_list, s_recon_split):
            loss_recon_fb_s += F.mse_loss(each_s, each_recon)

        args.loss_csv_fb.append(loss_recon_fb_t.item())
        args.loss_csv_fb.append(loss_recon_fb_s.item())

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
            # 因为展开后算上 batchsize 是三维，需要用 sum 后再平均
            criterion_int_kl = nn.KLDivLoss(reduction='sum')
            loss_fusion_fb = criterion_int_kl(F.log_softmax(t_fusion / self.T, dim=-1),
                                               F.softmax(s_fusion / self.T, dim=-1)) * (self.T * self.T) / (N * C)

        loss_kl_.append(loss_fusion_fb)
        loss_ce_all = sum(loss_ce_)
        loss_kl_all = sum(loss_kl_)
        args.loss_csv_fb.append(loss_fusion_fb.item())
        args.loss_csv_fb.append(loss_ce_all.item())
        args.loss_csv_fb.append(loss_kl_all.item())
        # loss_self_supervised_fb = 0.0


        if args.fbUseGradSim is True:
            # 将 中间层 的融合损失 与 最后的logit 与 labels 的损失做 梯度相似性比较。
            cos_similarity_temp = cal_each_grad_sim(loss_ce_all, loss_kl_all, model_t_weights, args)
            args.loss_csv_fb.append(cos_similarity_temp.item())
            if cos_similarity_temp < 0:
                loss_self_supervised_fb = loss_ce_all
            else:
                loss_self_supervised_fb = loss_ce_all + loss_kl_all
        else:
            args.loss_csv_fb.append('None')
            loss_self_supervised_fb = loss_ce_all + loss_kl_all

        # with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
        #     f.write("CH_Recon_T_fb:" + str(loss_recon_fb_t.item()) + '\t')
        #     f.write("CH_Recon_S_fb:" + str(loss_recon_fb_s.item()) + '\t')
        #     f.write("CH_Fusion_fb:" + str(loss_fusion_fb.item()) + '\t')
        #     f.write("Self-Supervised_fb:" + str(loss_self_supervised_fb.item()) + '\t')

        # loss_chfb = loss_recon_fb + loss_fusion_fb
        loss_chfb_dict = dict(recon_T_fb=loss_recon_fb_t, recon_S_fb=loss_recon_fb_s,
                               self_supervised_fb=loss_self_supervised_fb)
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

    cosin_simility = []
    for each_cls, each_other in zip(cls_grad[:split_idx], other_grad[:split_idx]):
        cosin_simility.append(F.cosine_similarity(each_cls.reshape(-1), each_other.reshape(-1), dim=0))

    cosin_simility_all = torch.tensor(0.0).cuda()
    for each_sim in cosin_simility:
        cosin_simility_all += torch.mean(each_sim)

    return cosin_simility_all

def main():
    # a = torch.randn(2, 3, 32, 32)
    # auxcf = inAuxCF(3, 3, 100)
    # # print(torchinfo.summary(auxcf))
    # a_auxcf = auxcf(a)
    # print(a_auxcf.shape)

    a = torch.randn(2, 5, 4, 4)

    auxnet = inAuxCF2(5)

    out = auxnet(a, 2)

    print(out.shape)

    print(auxnet)

    print(123)


if __name__ == '__main__':
    main()