from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['CDLoss', 'KDLossv2']

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


class CDLoss(nn.Module):
    """Channel Distillation Loss"""

    def __init__(self, args):
        super().__init__()
        self.Embed_1 = Embed(args.s_dim[0], args.t_dim[0])
        self.Embed_2 = Embed(args.s_dim[1], args.t_dim[1])
        self.Embed_3 = Embed(args.s_dim[2], args.t_dim[2])
        self.Embed_4 = Embed(args.s_dim[3], args.t_dim[3])

    def forward(self, stu_features: list, tea_features: list):
        loss = 0.
        Embed_list = [self.Embed_1, self.Embed_2, self.Embed_3, self.Embed_4]
        for s, t, embed_each in zip(stu_features, tea_features, Embed_list):
            s = embed_each(s)
            s = s.mean(dim=(2, 3), keepdim=False)
            t = t.mean(dim=(2, 3), keepdim=False)
            loss += torch.mean(torch.pow(s - t, 2))
        return loss


class KDLossv2(nn.Module):
    """Guided Knowledge Distillation Loss"""

    def __init__(self, T):
        super().__init__()
        self.t = T

    def forward(self, stu_pred, tea_pred, label):
        s = F.log_softmax(stu_pred / self.t, dim=1)
        t = F.softmax(tea_pred / self.t, dim=1)
        t_argmax = torch.argmax(t, dim=1)
        mask = torch.eq(label, t_argmax).float()
        count = (mask[mask == 1]).size(0)
        mask = mask.unsqueeze(-1)
        correct_s = s.mul(mask)
        correct_t = t.mul(mask)
        correct_t[correct_t == 0.0] = 1.0

        loss = F.kl_div(correct_s, correct_t, reduction='sum') * (self.t**2) / count
        return loss