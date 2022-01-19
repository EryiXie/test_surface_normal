import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from data.config import cfg
from models.funs import Euclidean2Sphere


class Loss(nn.Module):

    def __init__(self):
        super(Loss, self).__init__()

        # Losses funcs
        #self.pts_loss = nn.L1Loss()
        self.pts_loss = circle_loss()
        self.valid_loss = nn.BCELoss()

    def forward(self, net, normal_preds, gt_normals):
        # normal loss

        gt_valid_mask = torch.norm(gt_normals, p=2, dim=1)
        segmentation_loss = self.valid_loss(normal_preds[:,2,:,:], gt_valid_mask) * 0.1
        gt_normals = Euclidean2Sphere(gt_normals)
        point_wise_loss = self.pts_loss(normal_preds[:,:2,:,:], gt_normals)
        return {'pts': point_wise_loss, 'seg': segmentation_loss}

class circle_loss(nn.Module):
    def __init__(self):
        super(circle_loss, self).__init__()
    
    def forward(self, normal_preds, gt_normals):
        #B, C, H, W = normal_preds.shape
        term_phi = torch.abs(normal_preds[:,1,:,:] - gt_normals[:,1,:,:])
        term_theta = torch.abs(normal_preds[:,0,:,:] - gt_normals[:,0,:,:])
        term_theta = 2 * torch.minimum(term_theta, 1 - term_theta)
        loss = term_phi.mean() + term_theta.mean()
        return loss
