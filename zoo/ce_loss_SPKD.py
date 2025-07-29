import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning.miners import TripletMarginMiner
import torch
from scipy.stats import norm

def batch_similarity(fm): # batch similarity
    fm = fm.view(fm.size(0), -1)
    Q = torch.mm(fm, fm.transpose(0,1))
    normalized_Q = Q / torch.norm(Q,2,dim=1).unsqueeze(1).expand(Q.shape)
    return normalized_Q

def spatial_similarity(fm): # spatial similarity
    fm = fm.view(fm.size(0), fm.size(1),-1)
    norm_fm = fm / (torch.sqrt(torch.sum(torch.pow(fm,2), 1)).unsqueeze(1).expand(fm.shape) + 0.0000001 )
    s = norm_fm.transpose(1,2).bmm(norm_fm)
    s = s.unsqueeze(1)
    return s

def channel_similarity(fm): # channel_similarity
    fm = fm.view(fm.size(0), fm.size(1), -1)
    norm_fm = fm / (torch.sqrt(torch.sum(torch.pow(fm,2), 2)).unsqueeze(2).expand(fm.shape) + 0.0000001)
    s = norm_fm.bmm(norm_fm.transpose(1,2))
    s = s.unsqueeze(1)
    return s

class CELoss(nn.Module):
    """Cross Entropy Loss"""

    def __init__(self):
        super().__init__()

    def forward(self, stu_pred, tea_pred, label):
        l2 = nn.MSELoss()
        sp_loss = 0.
        aggregated_student_fms = []
        aggregated_teacher_fms = []
        student_fms = []
        teacher_fms = []
#         student_fms = stu_pred[0]
#         teacher_fms = tea_pred[0]
        student_fms.append(stu_pred[0])
        student_fms.append(stu_pred[1])
        student_fms.append(stu_pred[2])
        student_fms.append(stu_pred[3])
        teacher_fms.append(tea_pred[0])
        teacher_fms.append(tea_pred[1])
        teacher_fms.append(tea_pred[2])
        teacher_fms.append(tea_pred[3])
        
        feature_maps = '[0,1,2,3]'
        KD_weight = '[1,1,1]'
        KD_weight = eval(KD_weight)
#         print(eval(feature_maps))
        selected_student_fms = [student_fms[ind] for ind in eval(feature_maps)]
        selected_teacher_fms = [teacher_fms[ind] for ind in eval(feature_maps)]
        
        revised_student_fms = [student_fms[ind] for ind in eval(feature_maps)]
        revised_teacher_fms = [teacher_fms[ind] for ind in eval(feature_maps)]
        
#         aggregated_student_fms.append([batch_similarity(fm) for fm in selected_student_fms])
#         aggregated_teacher_fms.append([batch_similarity(fm) for fm in selected_teacher_fms])
        
        aggregated_student_fms.append([spatial_similarity(fm) for fm in selected_student_fms])
        aggregated_teacher_fms.append([spatial_similarity(fm) for fm in selected_teacher_fms])
        
#         aggregated_student_fms.append([channel_similarity(fm) for fm in revised_student_fms])
#         aggregated_teacher_fms.append([channel_similarity(fm) for fm in revised_teacher_fms])
        
        for i in range(len(aggregated_student_fms)):
            for j in range(len(aggregated_student_fms[i])):
                sp_loss += l2(aggregated_student_fms[i][j], aggregated_teacher_fms[i][j]) * KD_weight[i]
        loss = F.cross_entropy(stu_pred[-1], label)
        return loss, sp_loss
#         print(sp_loss)
#         return sp_loss