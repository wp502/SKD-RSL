from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
import torch
from copy import deepcopy

__all__ = ['SOKDLoss', 'auxiliary_forward']


class auxiliary_forward(nn.Module):
    def __init__(self, model_t_name, model_t):
        super(auxiliary_forward, self).__init__()
        self.model_t_name = model_t_name
        self.relu = nn.ReLU(inplace=True)
        if 'ResNet' in self.model_t_name:
            # ResNet V2
            self.auxiliary_block = nn.Sequential(deepcopy(model_t.layer4))
            self.auxiliary_bn = deepcopy(model_t.avgpool)
            self.auxiliary_fc = deepcopy(model_t.linear)
        elif 'VGG' in self.model_t_name:
            self.auxiliary_block = nn.Sequential(deepcopy(model_t.block4))
            self.auxiliary_bn = deepcopy(model_t.pool4)
            self.auxiliary_fc = deepcopy(model_t.classifier)
        elif 'ShuffleV2' in self.model_t_name:
            self.auxiliary_block = nn.Sequential(deepcopy(model_t.conv5))
            self.auxiliary_fc = deepcopy(model_t.fc)
        elif 'MobileNetV2' in self.model_t_name:
            self.auxiliary_block = nn.Sequential(deepcopy(model_t.blocks[5]))
            self.auxiliary_block2 = nn.Sequential(deepcopy(model_t.blocks[6]))
            self.auxiliary_block3 = nn.Sequential(deepcopy(model_t.conv2))
            self.auxiliary_bn = deepcopy(model_t.avgpool)
            self.auxiliary_fc = nn.Sequential(deepcopy(model_t.classifier))
        elif 'WRN' in self.model_t_name:
            self.auxiliary_block = nn.Sequential(deepcopy(model_t.block3))
            self.auxiliary_bn = deepcopy(model_t.bn1)
            self.auxiliary_bn2 = deepcopy(model_t.avgpool)
            self.auxiliary_fc = deepcopy(model_t.fc)


    def forward(self, feat):
        if 'ResNet' in self.model_t_name:
            out = self.auxiliary_block(feat)
            f0 = out
            out = self.auxiliary_bn(out)
            out = out.view(out.size(0), -1)
            f1 = out
            out = self.auxiliary_fc(out)
            # return [f0, f1], out
        elif 'VGG' in self.model_t_name:
            # VGG 中 f3 下面有一句 if h == 64，这个 h是最早输入的x的shape[2]，也就是图像的H，我们的图像是32和224，所以直接就不加了。
            out = self.auxiliary_block(feat)
            f0 = out
            out = self.relu(out)
            out = self.auxiliary_bn(out)
            out = out.view(out.size(0), -1)
            f1 = out
            out = self.auxiliary_fc(out)
        elif 'ShuffleV2' in self.model_t_name:
            out = self.auxiliary_block(feat)
            f0 = out
            if 'img' in self.model_t_name:
                out = out.mean([2, 3])
            else:
                out = F.adaptive_avg_pool2d(out, 1)
                out = out.view(out.size(0), -1)
            f1 = out
            out = self.auxiliary_fc(out)
        elif 'MobileNetV2' in self.model_t_name:
            out = self.auxiliary_block(feat)
            out = self.auxiliary_block2(out)
            f0 = out
            out = self.auxiliary_block3(out)
            out = self.auxiliary_bn(out)
            out = out.view(out.size(0), -1)
            f1 = out
            out = self.auxiliary_fc(out)
        elif 'WRN' in self.model_t_name:
            widen_factor = int(self.model_t_name.split('_')[-1])
            nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
            out = self.auxiliary_block(feat)
            f0 = out
            out = self.relu(self.auxiliary_bn(out))
            out = self.auxiliary_bn2(out)
            out = out.view(-1, nChannels[3])
            f1 = out
            out = self.auxiliary_fc(out)

        return [f0, f1], out


class SOKDLoss(nn.Module):
    """SOKD"""

    def __init__(self, T, sokd_aux_t, sokd_aux_s, sokd_kd):
        super(SOKDLoss, self).__init__()
        self.T = T
        self.criterion_kd = F.kl_div
        self.auxiliary_lambda_kd_t = sokd_aux_t
        self.auxiliary_lambda_kd_s = sokd_aux_s
        self.lambda_kd = sokd_kd

    def forward(self, output_list):
        loss_kd_T_A = self.criterion_kd(
                            F.log_softmax(output_list[2]/self.T, dim=1), 
                            F.softmax(output_list[0].detach()/self.T, dim=1),
                            reduction='batchmean') * self.T * self.T * self.auxiliary_lambda_kd_t
        loss_kd_S_A = self.criterion_kd(
                            F.log_softmax(output_list[2]/self.T, dim=1),
                            F.softmax(output_list[1].detach()/self.T, dim=1),
                            reduction='batchmean') * self.T * self.T * self.auxiliary_lambda_kd_s
        loss_S = self.criterion_kd(
                            F.log_softmax(output_list[1]/self.T, dim=1), 
                            F.softmax(output_list[2].detach()/self.T, dim=1),
                            reduction='batchmean') * self.T * self.T * self.lambda_kd
        return loss_kd_T_A, loss_kd_S_A, loss_S