from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.autograd import Variable

__all__ = ['Fusion_module', 'KLLoss', 'get_current_consistency_weight']
# class Fusion_module(nn.Module):
#     def __init__(self, channel, numclass, sptial=4):
#         super(Fusion_module, self).__init__()
#         self.fc2 = nn.Linear(channel, numclass)
#         self.conv1 = nn.Conv2d(channel*2, channel*2, kernel_size=3,
#                                stride=1, padding=1, groups=channel*2, bias=False)
#         self.bn1 = nn.BatchNorm2d(channel * 2)
#         self.conv1_1 = nn.Conv2d(
#             channel*2, channel, kernel_size=1, groups=1, bias=False)
#         self.bn1_1 = nn.BatchNorm2d(channel)
#
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#
#         self.sptial = sptial
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#         #self.avg = channel
#
#     def forward(self, x, y):
#         bias = False
#         atmap = []
#         input = torch.cat((x, y), 1)
#
#         x = F.relu(self.bn1((self.conv1(input))))
#         x = F.relu(self.bn1_1(self.conv1_1(x)))
#
#         atmap.append(x)
#         # x = F.avg_pool2d(x, self.sptial)
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#
#         out = self.fc2(x)
#         atmap.append(out)
#
#         return out

class Fusion_module(nn.Module):
    def __init__(self, channel, numclass, sptial=4):
        super(Fusion_module, self).__init__()
        # 输入的 channel 为 x+y 的和，所以需要 // 2。和之前的 *2 是一样的。
        self.fc2 = nn.Linear(channel//2, numclass)
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3,
                               stride=1, padding=1, groups=channel, bias=False)
        self.bn1 = nn.BatchNorm2d(channel)
        self.conv1_1 = nn.Conv2d(
            channel, channel//2, kernel_size=1, groups=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(channel//2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.sptial = sptial

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #self.avg = channel

    def forward(self, x, y):
        bias = False
        atmap = []
        input = torch.cat((x, y), 1)

        x = F.relu(self.bn1((self.conv1(input))))
        x = F.relu(self.bn1_1(self.conv1_1(x)))

        atmap.append(x)
        # x = F.avg_pool2d(x, self.sptial)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        out = self.fc2(x)
        atmap.append(out)

        return out


class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()
    def forward(self, pred, label):
        # pred: 2D matrix (batch_size, num_classes)
        # label: 1D vector indicating class number
        T=3

        predict = F.log_softmax(pred/T,dim=1)
        target_data = F.softmax(label/T,dim=1)
        target_data =target_data+10**(-7)
        target = Variable(target_data.data.cuda(),requires_grad=False)
        loss=T*T*((target*(target.log()-predict)).sum(1).sum()/target.size()[0])
        return loss

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return sigmoid_rampup(epoch, 80)


if __name__ == '__main__':

    fmodule = Fusion_module(channel=7, numclass=100)

    a = torch.randn(10, 2, 4, 4)
    b = torch.randn(10, 5, 4, 4)

    fmodule(a, b)
