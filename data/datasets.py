import os
import os.path as osp
import torch
import torch.utils.data as data
import torch.nn.functional as F
import cv2
import numpy as np
from .config import cfg

class HolicityDataset(data.Dataset):
    """ 
    """
    def __init__(self, root=cfg.dataset.root_path, 
                    split_file=cfg.dataset.valid_split, 
                    transform=None):
        self.root = root
        f = open(os.path.join(root, split_file), 'r')
        self.ids = f.readlines()
        f.close()
        self.transform = transform
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, normal).
        """
        image, normal = self.pull_item(index)
        return image, normal

    def pull_item(self, index):

        path = os.path.join(self.root, 'image-v1', self.ids[index].strip('\n') +  "_imag.jpg")
        assert osp.exists(path), 'Image path does not exist: {}'.format(path)

        img = cv2.imread(path).astype(np.float32)
        H, W, _ = img.shape

        normal_path = path.replace("image-v1", "normal-v1").replace("_imag.jpg", "_nrml.npz")
        normal = np.load(normal_path)['normal'].astype(np.float32)
        if self.transform is not None:
            img, normal = self.transform(img, normal)
        return torch.from_numpy(img).permute(2, 0, 1), torch.from_numpy(normal).permute(2, 0, 1)
    
    def __len__(self):
        return len(self.ids)
    

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            cv2 img
        '''
        file_name = os.path.join(self.root, 'image-v1', self.ids[index].strip('\n') +  "_imag.jpg")
        return cv2.imread(osp.join(self.root, file_name), cv2.IMREAD_COLOR)
    
    def pull_normal(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            numpy np.double
        '''
        path = os.path.join(self.root, 'normal-v1', self.ids[index].strip('\n') +  "_nrml.npz")
        normal = np.load(path)['normal'].astype(np.float32)
        return normal

def enforce_size(img, normal, new_w, new_h):
    """ Ensures that the image is the given size without distorting aspect ratio. """
    with torch.no_grad():
        _, h, w = img.size()

        if h == new_h and w == new_w:
            return img, normal
        
        # Resize the image so that it fits within new_w, new_h
        w_prime = new_w
        h_prime = h * new_w / w

        if h_prime > new_h:
            w_prime *= new_h / h_prime
            h_prime = new_h

        w_prime = int(w_prime)
        h_prime = int(h_prime)

        # Do all the resizing
        img = F.interpolate(img.unsqueeze(0), (h_prime, w_prime), mode='bilinear', align_corners=False)
        img.squeeze_(0)

        normal = F.interpolate(normal.unsqueeze(0), (h_prime, w_prime), mode='bilinear', align_corners=False)
        normal.squeeze_(0)

        # Finally, pad everything to be the new_w, new_h
        pad_dims = (0, new_w - w_prime, 0, new_h - h_prime)
        img   = F.pad(  img, pad_dims, mode='constant', value=0)
        normal = F.pad(normal, pad_dims, mode='constant', value=0)

        return img, normal


if __name__ == "__main__":
    from data.config import cfg
    from data.augmentations import BaseTransform, MEANS
    dataset = HolicityDataset(root=cfg.dataset.root_path,
                            split_file=cfg.dataset.valid_split, 
                                    transform=BaseTransform(MEANS))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    for idx, data_ele in enumerate(dataloader):
        image, normal = data_ele
        print(image.shape, normal.shape)
        print(image.dtype, normal.dtype)
        if idx > 3:
            break






