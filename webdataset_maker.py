import torch
import numpy
import json
import os
from data.datasets import HolicityDataset
from data.config import cfg, set_cfg, MEANS, set_dataset
from data.augmentations import SSDAugmentation


import webdataset as wds
import torchvision
import sys

if __name__ == "__main__":

    set_cfg("NormalNet_base_config")
    set_dataset("dataset_base")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    
    dataset = HolicityDataset(root=cfg.dataset.root_path,
                            split_file=cfg.dataset.valid_split, 
                            transform=SSDAugmentation(MEANS))

    sink = wds.TarWriter("holicity_valid.tar")
    for idx in range(len(dataset)):
        img = dataset.pull_image(idx)
        gt_normal = dataset.pull_normal(idx)
        sink.write({
            "__key__": "sample%06d" % idx,
            "img.jpg": img,
            'normal.npz': {'normal':gt_normal},
        })
        #if idx > 10:
            #break
    sink.close()



