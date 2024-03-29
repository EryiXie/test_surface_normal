import io
import os
from argparse import ArgumentParser
import PIL.Image
import cv2
import torch
import tarfile

import numpy as np
import webdataset as wds
from data.augmentations import SSDAugmentation
from collections.abc import Iterable

from data.config import cfg, MEANS, set_cfg

class HolicityDataset:
    def __init__(self, root: str, split_file: str = "train", transform=None):
        self.tar_path = os.path.join(root, split_file)
        self.data = {}
        self.transform = transform
        with tarfile.open(self.tar_path, mode="r|*") as stream:
            for info in stream:
                if not info.isfile(): continue
                filename = info.name
                if filename is None:
                    continue
                data = stream.extractfile(info).read()
                self.data[filename] = data

    def __len__(self):
        return int(len(self.data) / 2)
    
    
    def get_image(self, idx_str: str):
        raw = self.data[idx_str + ".img.jpg"]
        with io.BytesIO(raw) as stream:
            img = PIL.Image.open(stream)
            img.load()
            img = img.convert("RGB")
        return np.asarray(img).astype(np.float32)[:,:,(2,1,0)]
    
    def get_normal(self, idx_str: str):
        raw = self.data[idx_str + ".normal.npz"]
        with io.BytesIO(raw) as stream:
            normal = np.load(stream)['normal']
        return normal

    def __getitem__(self, idx: int):
        idx_str = f"sample{idx:06d}"
        image = self.get_image(idx_str)
        normal = self.get_normal(idx_str)
        if self.transform is not None:
            image, normal = self.transform(image, normal)
        return torch.from_numpy(image).permute(2, 0, 1), torch.from_numpy(normal).permute(2, 0, 1)


if __name__ == "__main__":
    from data.augmentations import SSDAugmentation
    parser = ArgumentParser()
    parser.add_argument('--dataset-path',
                        type=str,
                        dest='dataset_path',
                        default='./holicity/holicity_valid.tar',
                        help='The dataset to use')
    set_cfg('NormalNet_base_config')
    args = parser.parse_args()

    dataset = HolicityDataset(root=cfg.dataset.root_path, split_file=cfg.dataset.valid_split, transform=SSDAugmentation())
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=1, batch_size=4)
    i = 0
    for images, normals in dataloader:
        img_show = images[0].squeeze()[(2, 1, 0), :, :].contiguous().permute(1,2,0).cpu().numpy().astype(np.uint8)
        print(img_show.shape)
        cv2.imshow('raw', img_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #print(images.shape, normals.shape)
        i = i + 1
        if i>3:
            break
    


