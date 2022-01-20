"""Adapted from:
    @dbolya yolact: https://github.com/dbolya/yolact/data/config.py
    Licensed under The MIT License [see LICENSE for details]
"""

import argparse
import random
import os
import numpy as np
import cv2
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from normal_net import TestNet
from data.web_dataset import HolicityDataset
from data.config import set_cfg, set_dataset, cfg, MEANS, STD
from data.augmentations import BaseTransform
from utils.utils import MovingAverage, ProgressBar, SavePath
from utils import timer
from models.funs import Sphere2Euclidean


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='PlaneSegNet Evaluation')
    parser.add_argument('--trained_model',
                        default=None, type=str,
                        help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
    parser.add_argument('--max_images', default=-1, type=int,
                        help='The maximum number of images from the dataset to consider. Use -1 for all.')
    parser.add_argument('--config', default=None,
                        help='The config object to use.')
    parser.add_argument('--no_bar', dest='no_bar', action='store_true',
                        help='Do not output the status bar. This is useful for when piping to a file.')
    parser.add_argument('--dataset', default=None, type=str,
                        help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
    global args
    args = parser.parse_args(argv)

normal_metrics = ["mean", "median", "rmse", "a1", "a2", "a3"]

def tensorborad_visual_log(epoch, iteration, net: TestNet, dataset, writer: SummaryWriter,  eval_nums = 5):
    dataset_indices = list(range(len(dataset)))
    random.shuffle(dataset_indices)
    dataset_indices = dataset_indices[:eval_nums]
    means = torch.Tensor(MEANS).float().cuda()[None, :, None, None]
    std = torch.Tensor(STD).float().cuda()[None, :, None, None]
    try:
        # Main eval loop
        for it, image_idx in enumerate(dataset_indices):
            image, gt_normal = dataset[image_idx]
            image_ori = image[(2, 1, 0), :, :].contiguous()
            image_ori = (image_ori * std.to(image.device) + means.to(image.device)).squeeze().permute(1,2,0).cpu().numpy().astype(np.uint8)
            batch = Variable(image.unsqueeze(0)).cuda()

            batched_result = net(batch) # if batch_size = 1, result = batched_result[0]
            batched_result = Sphere2Euclidean(batched_result)
            norm_np = batched_result[0].squeeze().permute(1,2,0).cpu().numpy()
            norm_draw = (((norm_np + 1) / 2) * 255).astype(np.uint8)
            #gt_norm_normalize_np = gt_normal.squeeze().permute(1,2,0).cpu().numpy()
            #gt_norm_normalize_draw = (((gt_norm_normalize_np + 1) / 2) * 255).astype(np.uint8)

            #writer.add_image("{}/Input/GT".format(it), image_ori, iteration, dataformats='HWC')
            #writer.add_image("{}/Normal/GT".format(it), gt_norm_normalize_draw, iteration, dataformats='HWC')
            writer.add_image("{}/Normal/Pred".format(it), norm_draw, iteration, dataformats='HWC')

    except KeyboardInterrupt:
        print('Stopping...')

def evaluate(net: TestNet, dataset, during_training=False, eval_nums=-1):
    frame_times = MovingAverage()
    if eval_nums < 0:
        eval_nums = cfg.dataset.valid_length if args.max_images < 0 else min(args.max_images, cfg.dataset.valid_length)
    progress_bar = ProgressBar(30, eval_nums)

    print()

    dataset_indices = list(range(cfg.dataset.valid_length))
    random.shuffle(dataset_indices)
    dataset_indices = dataset_indices[:eval_nums]

    infos = []
    it = 0
    try:
        # Main eval loop
        for it, image_idx in enumerate(dataset_indices):
            timer.reset()

            image, gt_normal = dataset[image_idx]
            batch = Variable(image.unsqueeze(0)).cuda()

            batched_result = net(batch) # if batch_size = 1, result = batched_result[0]
            pred_normal = Sphere2Euclidean(batched_result)[0]
            pred_normal = pred_normal.squeeze(dim=0)

            gt_normal = gt_normal.cuda()
            normal_error_per_frame = compute_normal_metrics(pred_normal, gt_normal, median_scaling=True)
            infos.append(normal_error_per_frame)

            # First couple of images take longer because we're constructing the graph.
            # Since that's technically initialization, don't include those in the FPS calculations.
            if it > 1:
                frame_times.add(timer.total_time())

            if not args.no_bar:
                if it > 1:
                    fps = 1000 / frame_times.get_avg()
                else:
                    fps = 0
                progress = (it+1) / eval_nums * 100
                progress_bar.set_val(it+1)
                print('\rProcessing Images  %s %6d / %6d (%5.2f%%)    %5.2f fps        '
                      % (repr(progress_bar), it+1, eval_nums, progress, fps), end='')
            it = it + 1
            if it >= eval_nums:
                break
        infos = np.asarray(infos, dtype=np.double)
        infos = infos.sum(axis=0)/infos.shape[0]
        print("Normal Metrics:")
        print("{}: {:.3f}, {}: {:.3f}, {}: {:.3f}, {}: {:.3f}, {}: {:.3f}, {}: {:.3f}".format(
            normal_metrics[0], infos[0], normal_metrics[1], infos[1], normal_metrics[2], infos[2],
            normal_metrics[3], infos[3], normal_metrics[4], infos[4], normal_metrics[5], infos[5]
        ))
        print()
        return normal_metrics, infos
    except KeyboardInterrupt:
        print('Stopping...')


def compute_normal_metrics(pred_normal, gt_normal, median_scaling=True):
    """
    Computation of error metrics between predicted and ground truth normals.
    Prediction and ground turth need to be converted to the same unit e.g. [meter].

    Arguments: pred_normal, gt_normal: Tensor [3, H, W], dense normal map
               median_scaling: If True, use median value to scale pred_normal
    Returns: mean_theta, median_theta, rmse, a1, a2, a3: normal metrics
    """

    _, H, W = gt_normal.shape
    pred_normals_flat = pred_normal.view(3, H*W)
    gt_normals_flat = gt_normal.view(3, H*W)
    valid_mask = (gt_normals_flat.sum(dim=0) > 0)#.logical_and(pred_normals_flat.sum(dim=0) > 0)
    pred_normals_flat = pred_normals_flat[:,valid_mask]
    gt_normals_flat = gt_normals_flat[:,valid_mask]

    theta = torch.acos(F.cosine_similarity(gt_normals_flat, pred_normals_flat, dim=0))/math.pi * 180
    theta = theta[torch.logical_not(theta.isnan())]
    if theta.shape[0] > 0:
        mean_theta = torch.mean(theta)
        median_theta = torch.median(theta)

        a1 = (theta < 11.25 ).type(torch.cuda.DoubleTensor).mean()
        a2 = (theta < 22.5  ).type(torch.cuda.DoubleTensor).mean()
        a3 = (theta < 30    ).type(torch.cuda.DoubleTensor).mean()

        rmse = theta ** 2
        rmse = torch.sqrt(rmse.mean())
    else:
        mean_theta, median_theta, rmse, a1, a2, a3 = 0, 0, 0, 0, 0, 0

    return mean_theta, median_theta, rmse, a1, a2, a3

if __name__ == '__main__':
    parse_args()

    if args.config is not None:
        set_cfg(args.config)
    
    if args.trained_model == 'interrupt':
        args.trained_model = SavePath.get_interrupt('weights/')
    elif args.trained_model == 'latest':
        args.trained_model = SavePath.get_latest('weights/', cfg.name)
    
    if args.config is None:
        model_path = SavePath.from_str(args.trained_model)
        args.config = model_path.model_name + '_config'
        print('Config not specified. Parsed %s from the file name.\n' % args.config)
        set_cfg(args.config)
    
    if args.dataset is not None:
        set_dataset(args.dataset)
    
    with torch.no_grad():
        if not os.path.exists('results'):
            os.makedirs('results')
        
        cudnn.fastest = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        
        dataset = HolicityDataset(root=cfg.dataset.root_path,
                            split_file=cfg.dataset.valid_split,
                            transform=BaseTransform(MEANS))
        print("Loading model...", end='')
        net = TestNet(cfg)
        net.load_weights(args.trained_model)
        net.eval()
        print("done.")

        net = net.cuda()
        evaluate(net, dataset)






