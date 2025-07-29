from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
import torch

__all__ = ['KDCLLoss']

class KDCLLoss(nn.Module):
    """KDCL"""

    def __init__(self):
        super(KDCLLoss, self).__init__()

    def forward(self, output_list, stable_out, T, model_idx):
        div_loss = torch.nn.functional.kl_div(
                        torch.nn.functional.log_softmax(output_list[model_idx] / T, dim=1),
                        torch.nn.functional.softmax(stable_out / T, dim=1),
                        reduction='batchmean'
                    ) * T * T
        return div_loss