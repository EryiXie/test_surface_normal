import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.utils import *
from data.config import cfg
from models.funs import Euclidean2Sphere


class Loss(nn.Module):

    def __init__(self):
        super(Loss, self).__init__()

        # Losses funcs
        #self.pts_loss = CossimLoss()
        self.pts_loss = nn.L1Loss()
        #self.pts_loss = circle_loss()


    def forward(self, net, normal_preds, gt_normals):
        # normal loss
        gt_normals = Variable(gt_normals, requires_grad=False)
        gt_normals = Euclidean2Sphere(gt_normals)
        point_wise_loss = self.pts_loss(normal_preds, gt_normals)
        return {'point': point_wise_loss}

class CossimLoss(nn.Module):
    def __init__(self):
        super(CossimLoss, self).__init__()
    
    def forward(self, normal_preds, gt_normals):
        cossim = torch.nn.functional.cosine_similarity(normal_preds, gt_normals)
        normal_error = (1 - cossim).mean()
        return normal_error

class circle_loss(nn.Module):
    def __init__(self):
        super(circle_loss, self).__init__()
    
    def forward(self, normal_preds, gt_normals):
        B, C, H, W = normal_preds.shape
        term_1 = torch.abs(normal_preds[:,1,:,:] - gt_normals[:,1,:,:])
        term_2 = torch.abs(normal_preds[:,0,:,:] - gt_normals[:,0,:,:])
        term_2 = 2 * torch.minimum(term_2, 1 - term_2)
        loss = term_1 + term_2
        return loss.mean()
