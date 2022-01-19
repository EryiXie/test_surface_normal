import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def Euclidean2Sphere(normal_euclidean):
    x = normal_euclidean[:, 0, :,:]
    y = normal_euclidean[:, 1, :,:]
    z = normal_euclidean[:, 2, :,:]

    theta = torch.atan2(y, x) / (2 * math.pi) + 0.5 # [-pi, pi] -> [0,1]
    phi = torch.atan2(torch.sqrt(x*x + y*y), z) / math.pi # [0, pi] -> [0, 1]

    normal_2dsphere = torch.stack([theta, phi], dim=1)
    return normal_2dsphere

def Sphere2Euclidean(normal_2dsphere):
    theta = (normal_2dsphere[:, 0, :, :] - 0.5)*2*math.pi
    phi = normal_2dsphere[:, 1, :, :]*math.pi
    valid_mask = normal_2dsphere[:, 2, :, :].unsqueeze(dim=1).repeat(1,3,1,1)
    #valid_mask = torch.logical_or((theta > 1e-2), (phi > 1e-2)).unsqueeze(dim=1).repeat(1,3,1,1)
    x = torch.sin(phi)*torch.cos(theta)
    y = torch.sin(phi)*torch.sin(theta)
    z = torch.cos(phi)
    
    normal_euclidean = torch.stack([x,y,z], dim=1)*valid_mask
    return normal_euclidean
