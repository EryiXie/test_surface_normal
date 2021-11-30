"""Adapted from:
    @dbolya yolact: https://github.com/dbolya/yolact/eval.py
    Licensed under The MIT License [see LICENSE for details]
"""
import argparse
import cv2
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from normal_net import TestNet
from data.augmentations import FastBaseTransform
from data.config import set_cfg, cfg
from utils import timer
from utils.utils import MovingAverage
from collections import defaultdict
from models.funs import Sphere2Euclidean
import numpy as np


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="DepNet Net Inference")
    parser.add_argument("--trained_model",default=None, type=str, help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
    parser.add_argument("--config", default="NormalNet_base_config", help="The config object to use.")
    # Inference Mode
    parser.add_argument("--image", default=None, type=str, help='Inference with a single image.')
    parser.add_argument("--images", default=None, type=str, help='Inference with multiple images.')
    parser.add_argument("--max_img", default=0, type=int, help="The maximum number of inference images in a folder.")
    global args
    args = parser.parse_args(argv)


def inference_image(net: TestNet, path: str, save_path: str = None):
    frame_np = cv2.imread(path)
    if frame_np is None:
        return
    frame = torch.from_numpy(frame_np).cuda().float()
    batch = FastBaseTransform()(frame.unsqueeze(0)) # Check FastBaseTransform
    batched_normal = net(batch)
    batched_normal = Sphere2Euclidean(batched_normal)
    normal = batched_normal[0]
    name, ext = os.path.splitext(save_path)
    print(normal.shape)
    norm_np = normal.squeeze().permute(1,2,0).detach().cpu().numpy()
    norm_draw = (((norm_np + 1) / 2) * 255).astype(np.uint8)
    name, ext = os.path.splitext(save_path)
    norm_path = os.path.join(name+"_norm"+ ".png")
    norm_draw = norm_draw[:, :, (2, 1, 0)]
    cv2.imwrite(norm_path, norm_draw)
   

def inference_images(net: TestNet, in_folder: str, out_folder: str, max_img: int=0):
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    print()
    index = 0
    input_list = list(Path(in_folder).glob('*'))
    max_img = min(max_img, len(input_list)) if max_img > 0 else len(input_list)
    for p in sorted(input_list):
        img_path = str(p)
        name, ext = os.path.splitext(os.path.basename(img_path))
        if ext != ".png" and ext != ".jpg":
            continue
        out_path = os.path.join(out_folder, name+ext)
        inference_image(net, img_path, out_path)
        print(img_path + ' -> ' + out_path,  end='\r')
        index = index + 1
        if index >= max_img:
            break
    print()
    print("Done.")


if __name__ == "__main__":
    nms_config = parse_args()
    timer.disable_all()

    set_cfg(args.config)
    net = TestNet(cfg)
    net.freeze_bn(True)
    if args.trained_model is not None:
        net.load_weights(args.trained_model)
    else:
        net.init_weights(backbone_path="weights/" + cfg.backbone.path)
        print(cfg.backbone.name)
        
    net.train(mode=False)
    net = net.cuda()
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

    if args.image is not None:
        if ':' in args.image:
            inp, out = args.image.split(':')
            inference_image(net, inp, out)
        else:
            inference_image(net, args.image)
    
    if args.images is not None:
        inp, out = args.images.split(':')
        inference_images(net, inp, out, args.max_img)