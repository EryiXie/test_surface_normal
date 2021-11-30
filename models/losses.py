import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.utils import *
from data.config import cfg


class Loss(nn.Module):

    def __init__(self):
        super(Loss, self).__init__()

        # Losses funcs
        #self.normal_loss = CossimLoss()
        self.normal_loss = nn.L1Loss()

    def forward(self, net, normal_preds, gt_normals):
        # normal loss
        gt_normals = Variable(gt_normals, requires_grad=False)
        #valid_mask = (gt_normals.sum() > 0) # All ground truth with 0 value is considered as invalid/non-informative pixels

        point_wise_loss = self.normal_loss(normal_preds, gt_normals)

        return {'point': point_wise_loss}

class CossimLoss(nn.Module):
    def __init__(self):
        super(CossimLoss, self).__init__()
    
    def forward(self, normal_preds, gt_normals):
        cossim = torch.nn.functional.cosine_similarity(normal_preds, gt_normals)
        normal_error = (1 - cossim).mean()
        return normal_error




