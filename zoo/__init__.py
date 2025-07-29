from .CCS import CCSLoss
from .CCS_2 import CCSLoss as CCSLoss_2
from .KD import DistillKL
from .ICKD import ICKDLoss
from .SPKD import SPKDLoss
from .CRD import CRDLoss
from .DML import DMLLoss
from .KDCL import KDCLLoss
from .SOKD import SOKDLoss, auxiliary_forward
from .Our import Our_FWD, Our_FB
from .Our_22 import Our_FWD_Conv, Our_FB_Conv
from .Our_12 import Our_FWD_Conv_12, Our_FB_Conv_12
from .Our_122 import Our_FWD_Conv_122, Our_FB_Conv_122
from .Our_13 import Our_FWD_Conv_13, Our_FB_Conv_13
from .Our_23 import Our_FWD_Conv_23, Our_FB_Conv_23
from .Our_15 import Our_FWD_Conv_15, Our_FB_Conv_15
from .Our_16 import Our_FWD_Conv_16, Our_FB_Conv_16
from .Our_16_2 import Our_FWD_Conv_16_2, Our_FB_Conv_16_2
from .Our_16_3 import Our_FWD_Conv_16_3, Our_FB_Conv_16_3
from .Our_18 import Our_FWD_Conv_18, Our_FB_Conv_18, initAUXCF
from .Our_19 import Our_FWD_Conv_19, Our_FB_Conv_19, initAUXCFAndAE19
from .Our_19_1 import  Our_FWD_Conv_19_1, Our_FB_Conv_19_1, initAUXCFAndAE19_1
from .Our_19_2 import Our_FWD_Conv_19_2, Our_FB_Conv_19_2, initAUXCFAndAE19_2
from .Our_19_4 import Our_FWD_Conv_19_4, Our_FB_Conv_19_4, initAUXCFAndAE19_4
from .Our_110 import Our_FWD_Conv_110, Our_FB_Conv_110, initAUXCFAndAE110
from .Our_111 import Our_FWD_Conv_111, Our_FB_Conv_111, initAUXCFAndAE111
from .Our_111_ddp import Our_FWD_Conv_111 as Our_FWD_ddp
from .Our_111_ddp import Our_FB_Conv_111 as Our_FB_ddp
from .Our_111_ddp import initAUXCFAndAE111 as initAUXAndAE_ddp
from .CD import KDLossv2, CDLoss
from .CKA import linear_CKA_GPU, linear_CKA, kernel_CKA
from .gkd20220912_MGD import MGD
from .FitNet import HintLoss
from .AT import Attention
from .ConvReg import ConvReg
from .FFL import *
from .SwitOKD import *
from .ReviewKD import *