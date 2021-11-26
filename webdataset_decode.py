import io
import os
from argparse import ArgumentParser
import PIL.Image
import cv2
import torch

import numpy as np
import webdataset as wds
from data.augmentations import SSDAugmentation

from data.config import cfg, MEANS, set_cfg

class HolicityDatasetParser:

    def __init__(self, transform=None):
        self.transform = transform
    
    @staticmethod
    def get_image(sample):
        raw = sample["img.jpg"]
        with io.BytesIO(raw) as stream:
            img = PIL.Image.open(stream)
            img.load()
            img = img.convert("RGB")
        return np.asarray(img).astype(np.float32)[:,:,(2,1,0)]
    @staticmethod
    def get_normal(sample):
        raw = sample["normal.npz"]
        with io.BytesIO(raw) as stream:
            normal = np.load(stream)['normal']
        return normal

    def __call__(self, sample):
        img = self.get_image(sample)
        normal = self.get_normal(sample)
        if self.transform is not None:
            img, normal = self.transform(img, normal)
        return torch.from_numpy(img).permute(2, 0, 1), torch.from_numpy(normal).permute(2, 0, 1)


def HolicityDataset(root: str, split_file: str, transform=None):
    path = os.path.join(root, split_file)
    wds.WebDataset(path,)
    datastream = wds.WebDataset(path).shuffle(1000)
    datastream = wds.Processor(datastream, wds.map, HolicityDatasetParser(transform=transform))
    
    return datastream


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset-path',
                        type=str,
                        dest='dataset_path',
                        default='/ds-av/public_datasets/fashion_mnist/wds',
                        help='The dataset to use')
    set_cfg('NormalNet_base_config')
    args = parser.parse_args()

    dataset = HolicityDataset(root=cfg.dataset.root_path, split_file=cfg.dataset.valid_split)
    dataloader = torch.utils.data.DataLoader(dataset.batched(batchsize=4), num_workers=4, batch_size=None)
    for images, normals in dataloader:
        img_show = images[0].squeeze().permute(1,2,0).cpu().numpy().astype(np.uint8)
        print(img_show.shape)
        cv2.imshow('raw', img_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(images.shape, normals.shape)
        break
    


