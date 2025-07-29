from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils_loss import ConvReg

__all__ = ['CCSLoss']

class Embed(nn.Module):
    def __init__(self, dim_in=256, dim_out=128):
        super(Embed, self).__init__()
        self.conv2d = nn.Conv2d(
            dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.l2norm = nn.BatchNorm2d(dim_out)  # Normalize(2)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.l2norm(x)
        return x

class CCSLoss(nn.Module):

    def __init__(self, args):
        super(CCSLoss, self).__init__()
        self.Embed_1 = Embed(args.s_dim[0], args.t_dim[0])
        self.Embed_2 = Embed(args.s_dim[1], args.t_dim[1])
        self.Embed_3 = Embed(args.s_dim[2], args.t_dim[2])
        self.Embed_4 = Embed(args.s_dim[3], args.t_dim[3])

    def forward(self, feat_s, feat_t):
        Embed_list = [self.Embed_1, self.Embed_2, self.Embed_3, self.Embed_4]
        loss = [ccs_loss(embed_each(s1), t1) for s1, t1, embed_each in zip(feat_s, feat_t, Embed_list)]
        # loss = ccs_loss(fm_s1, fm_t1)
        return loss


def ccs_loss(fm_s1, fm_t1):  # CCS方案
    # N_s, C_s, H_s, W_s = fm_s1.shape
    # N_t, C_t, H_t, W_t = fm_t1.shape
    # if not C_s == C_t:
    #     fm_s1 = ConvReg(s_shape=fm_s1.shape, t_shape=fm_t1.shape).cuda()(fm_s1)
    w_intra = 1.0
    w_inter = 1.0
    inter_fm_s1 = euclidean_dist_fm_inter(fm_s1, squared=True)
    inter_fm_t1 = euclidean_dist_fm_inter(fm_t1, squared=True)
    intra_fm_s1 = euclidean_dist_fm_intra(fm_s1, squared=True)
    intra_fm_t1 = euclidean_dist_fm_intra(fm_t1, squared=True)
    loss_inter = F.mse_loss(inter_fm_s1,  inter_fm_t1)
    loss_intra = F.mse_loss(intra_fm_s1,  intra_fm_t1)
    loss = w_intra * loss_intra + w_inter * loss_inter
    return loss


def euclidean_dist_fm_inter(fm, squared=False, eps=1e-12):  # CCS方案
    # print('feature map:----------------')
    # print(fm.shape)					# 128x16x32x32
    fm = fm.view(fm.size(0), fm.size(1), -1)
    fm_square = fm.pow(2).sum(dim=2)
    fm_prod = torch.bmm(fm, fm.permute(0, 2, 1))
    fm_dist = (fm_square.unsqueeze(1) + fm_square.unsqueeze(2) -
               2 * fm_prod).clamp(min=eps)
    if not squared:
        fm_dist = fm_dist.sqrt()
    fm_dist = fm_dist.clone()
    fm_dist[:, range(fm.size(1)), range(fm.size(1))] = 0
    fm_dist = fm_dist / fm_dist.max()
    return fm_dist


def euclidean_dist_fm_intra(fm, squared=False, eps=1e-12):  # CCS方案
    fm = fm.view(fm.size(0), -1)
    fm_square = fm.pow(2).sum(dim=1)
    fm_prod = torch.mm(fm, fm.t())
    fm_dist = (fm_square.unsqueeze(0) + fm_square.unsqueeze(1) -
               2 * fm_prod).clamp(min=eps)
    if not squared:
        fm_dist = fm_dist.sqrt()
    fm_dist = fm_dist.clone()
    fm_dist[range(len(fm)), range(len(fm))] = 0
    fm_dist = fm_dist / fm_dist.max()
    return fm_dist


if __name__ == '__main__':
    s1 = torch.randn(128, 64, 32, 32).cuda()
    s2 = torch.randn(128, 128, 16, 16).cuda()
    s3 = torch.randn(128, 256, 8, 8).cuda()
    t1 = torch.randn(128, 128, 32, 32).cuda()
    t2 = torch.randn(128, 256, 16, 16).cuda()
    t3 = torch.randn(128, 512, 8, 8).cuda()

    criterion = CCSLoss()
    loss = criterion([s1, s2, s3], [t1, t2, t3])
    print(loss, sum(loss))
