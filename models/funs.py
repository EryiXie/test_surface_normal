import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def Euclidean2Sphere(normal_map):
    x = normal_map[:, 0, :,:]
    y = normal_map[:, 1, :,:]
    z = normal_map[:, 2, :,:]

    theta = torch.atan2(y, x)
    phi = torch.atan2(z, torch.sqrt(x*x + y*y))

    normal_2dsphere = torch.stack([theta, phi], dim=1)
    return normal_2dsphere

def Sphere2Euclidean(normal_map):
    theta = normal_map[:, 0, :, :]
    phi = normal_map[:, 1, :, :]
    z = torch.sin(phi)
    x = torch.cos(phi)*torch.cos(theta)
    y = torch.cos(phi)*torch.sin(theta)
    
    normal_3deuclidean = torch.stack([x,y,z], dim=1)
    return normal_3deuclidean
