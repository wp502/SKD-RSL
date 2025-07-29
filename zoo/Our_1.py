# -*- coding: utf-8 -*-
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from .CKA import linear_CKA_GPU, linear_CKA
import os
import numpy as np

__all__ = ["ConvReg", "OurLoss"]


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


class OurLoss(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T, args):
        super(OurLoss, self).__init__()
        self.T = T
        self.ConvReg_fwd_1 = ConvReg(args.s_dim[0], args.t_dim[0])
        self.ConvReg_fwd_2 = ConvReg(args.s_dim[1], args.t_dim[1])
        self.ConvReg_fwd_3 = ConvReg(args.s_dim[2], args.t_dim[2])
        self.ConvReg_fwd_4 = ConvReg(args.s_dim[3], args.t_dim[3])

        self.ConvReg_fdbk_1 = ConvReg(args.s_dim[0], args.t_dim[0])
        self.ConvReg_fdbk_2 = ConvReg(args.s_dim[1], args.t_dim[1])
        self.ConvReg_fdbk_3 = ConvReg(args.s_dim[2], args.t_dim[2])
        self.ConvReg_fdbk_4 = ConvReg(args.s_dim[3], args.t_dim[3])

    def forward(self, feat_t, feat_s, args, is_feedback=False):
        channel_loss = 0.0
        cka_score_list = []
        channel_loss_list = []
        loss_record_str = ""
        cka_record_str = ""

        if is_feedback:
            ConvReg_list = [self.ConvReg_fdbk_1, self.ConvReg_fdbk_2, self.ConvReg_fdbk_3, self.ConvReg_fdbk_4]
        else:
            ConvReg_list = [self.ConvReg_fwd_1, self.ConvReg_fwd_2, self.ConvReg_fwd_3, self.ConvReg_fwd_4]
        for t_each, s_each, ConvReg_each in zip(feat_t, feat_s, ConvReg_list):
            t_N, t_C, t_H, t_W = t_each.shape
            s_N, s_C, s_H, s_W = s_each.shape
            cka_score_pair = linear_CKA_GPU(t_each.reshape((t_N, -1)).detach(),
                                            s_each.reshape((s_N, -1)).detach())

            # cka_score_pair = linear_CKA(t_each.reshape((t_N, -1)).detach().cpu().numpy(),
            #                             s_each.reshape((s_N, -1)).detach().cpu().numpy())
            if args.is_100cka:
                cka_score_pair *= 100.0
            else:
                pass
            cka_score_list.append(cka_score_pair)

            s_each = ConvReg_each(s_each)

            s_each_temp = (s_each / self.T).reshape((t_N, t_C, -1))
            t_each_temp = (t_each / self.T).reshape((t_N, t_C, -1))

            # KL Loss
            if is_feedback:
                each_channel_loss = nn.KLDivLoss(reduction='sum')(F.log_softmax(t_each_temp, dim=-1),
                                                                  F.softmax(s_each_temp, dim=-1)) * ((self.T * self.T) / (t_N * t_C))
            else:
                each_channel_loss = nn.KLDivLoss(reduction='sum')(F.log_softmax(s_each_temp, dim=-1),
                                                                  F.softmax(t_each_temp, dim=-1)) * ((self.T * self.T) / (t_N * t_C))

            # MSE Loss 1
            # s_each_temp = (s_each / self.T).reshape((t_N, t_C, -1))
            # t_each_temp = (t_each / self.T).reshape((t_N, t_C, -1))
            # each_channel_loss = nn.MSELoss(reduction='sum')(F.softmax(s_each_temp, dim=-1),
            #                                                 F.softmax(t_each_temp, dim=-1)) * ((self.T * self.T) / (t_N * t_C))

            # MSE Loss 2
            # s_each = s_each.mean(dim=(2, 3), keepdim=False)
            # t_each = t_each.mean(dim=(2, 3), keepdim=False)
            # each_channel_loss = torch.mean(torch.pow(s_each - t_each, 2))

            channel_loss_list.append(each_channel_loss)
            loss_record_str += str(each_channel_loss.item()) + "\t"
            cka_record_str += str(cka_score_pair.item()) + "\t"

        # if args.is_use_cka:
        #     # 使用CKA来引导
        #     if is_feedback:
        #         cka_mean = torch.mean(torch.Tensor(cka_score_list))
        #         for idx, each_cka in enumerate(cka_score_list):
        #             if each_cka <= cka_mean:  # or each_cka <= 0.6
        #                 channel_loss += channel_loss_list[idx]
        #     else:
        #         # for each_ch_loss in channel_loss_list:
        #         #     channel_loss += each_ch_loss
        #         for idx, each_ch_loss in enumerate(channel_loss_list):
        #             channel_loss += (cka_score_list[idx] * each_ch_loss)
        #             # channel_loss += (1.0 - cka_score_list[idx]) * each_ch_loss
        # else:
        #     # 不使用CKA来引导
        #     for each_ch_loss in channel_loss_list:
        #         channel_loss += each_ch_loss

        if is_feedback:
            if args.is_use_fdbk_cka:
                cka_mean = torch.mean(torch.Tensor(cka_score_list))
                for idx, each_cka in enumerate(cka_score_list):
                    if each_cka <= cka_mean:  # or each_cka <= 0.6
                        channel_loss += channel_loss_list[idx]
            else:
                for each_ch_loss in channel_loss_list:
                    channel_loss += each_ch_loss
        else:
            if args.is_use_fwd_cka:
                for idx, each_ch_loss in enumerate(channel_loss_list):
                    channel_loss += (cka_score_list[idx] * each_ch_loss)
                    # channel_loss += (1.0 - cka_score_list[idx]) * each_ch_loss
            else:
                for each_ch_loss in channel_loss_list:
                    channel_loss += each_ch_loss

        with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
            f.write("\t" + "pair_loss:" + loss_record_str + '\n')
            f.write("\t" + "cka_score:" + cka_record_str + '\n')
        return channel_loss, cka_score_list

