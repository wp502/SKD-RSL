from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

__all__ = ['DMLLoss']

class DMLLoss(nn.Module):
    """DML"""

    def __init__(self):
        super(DMLLoss, self).__init__()

    def forward(self, outputs_list, now, num):
        kl_loss = 0.0
        for other in range(num):
            if now != other:
                kl_loss += nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs_list[now], dim=1),
                                        F.softmax(Variable(outputs_list[other]), dim=1))

        return kl_loss