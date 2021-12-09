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
        #self.pts_loss = nn.L1Loss()
        self.pts_loss = circle_loss()


    def forward(self, net, normal_preds, gt_normals):
        # normal loss
        gt_normals = Variable(gt_normals, requires_grad=False)
        valid_mask_gt = torch.norm(gt_normals, p=2, dim=1)
        gt_normals = Euclidean2Sphere(gt_normals)
        point_wise_loss = self.pts_loss(normal_preds, gt_normals, valid_mask_gt)
        return {'point': point_wise_loss}

class circle_loss(nn.Module):
    def __init__(self):
        super(circle_loss, self).__init__()
    
    def forward(self, normal_preds, gt_normals, valid_mask_gt):
        B, C, H, W = normal_preds.shape
        term_phi = torch.abs(normal_preds[:,1,:,:] - gt_normals[:,1,:,:])
        term_theta = torch.abs(normal_preds[:,0,:,:] - gt_normals[:,0,:,:])
        term_theta = 2 * torch.minimum(term_theta, 1 - term_theta)
        term_valid = torch.abs(normal_preds[:,2,:,:] - valid_mask_gt)
        loss = term_phi + term_theta + term_valid
        return loss.mean()
