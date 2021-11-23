import io
import os
from argparse import ArgumentParser
import PIL.Image
import cv2
import numpy as np
import webdataset as wds


class FashionMNISTParser:
    @staticmethod
    def get_image(sample):
        raw = sample["img.jpg"]
        with io.BytesIO(raw) as stream:
            img = PIL.Image.open(stream)
            img.load()
            img = img.convert("RGB")
        return np.asarray(img)
    
    @staticmethod
    def get_normal(sample):
        raw = sample["normal.npz"]
        with io.BytesIO(raw) as stream:
            print(type(stream))
            normal = np.load(stream)['normal']
        return normal

    def __call__(self, sample):
        for key in sample:
            print(key)
        image = self.get_image(sample)
        normal = self.get_normal(sample)
        return image, normal


def FashionMNISTDataset(path: str, split: str = "train"):
    #tars = [os.path.join(path, split + ".tar")]
    datastream = wds.WebDataset(path)
    datastream = wds.Processor(datastream, wds.map, FashionMNISTParser())
    return datastream


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset-path',
                        type=str,
                        dest='dataset_path',
                        default='/ds-av/public_datasets/fashion_mnist/wds',
                        help='The dataset to use')
    args = parser.parse_args()
    dataset = FashionMNISTDataset(args.dataset_path)
    # print('Dataset length:', len(dataset)) -> Exception, iterable datasets do not support len()
    for image, normal in dataset:
        print(image.shape, normal.shape)
        break
    


