import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random
from math import sqrt

from .config import cfg, MEANS, STD


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, normal=None):
        for t in self.transforms:
            img, normal = t(img, normal)
        return img, normal


class Resize_and_Pad(object):
    """
    Resize the image to its long side == cfg.max_size, filling the
    area: [(long side - short side) * long side] to with mean and 
    putting the image in the top-left.
    """
    def __init__(self, resize_gt=True, mean=MEANS, pad_gt=True):
        self.mean = mean
        self.pad_gt = pad_gt
        self.resize_gt = resize_gt
        self.max_size = cfg.max_size
    
    def __call__(self, image, normal):
        img_h, img_w, channels = image.shape
        
        if img_h != self.max_size or img_w != self.max_size:
            height, width = (self.max_size, int(img_w * (self.max_size / img_h))) if img_h > img_w  else (int(img_h * (self.max_size / img_w)), self.max_size)

            image = cv2.resize(image, (width, height))
            normal = cv2.resize(normal, (width, height))

            expand_image = np.zeros((self.max_size, self.max_size, channels), dtype=image.dtype)
            expand_image[:, :, :] = self.mean
            expand_image[:height, :width] = image

            expand_normal = np.zeros((self.max_size, self.max_size), dtype=normal.dtype)
            expand_normal[:height, :width] = normal
            
            return expand_image, expand_normal
        else:
            return image, normal


class Pad(object):
    """
    Pads the image to the input width and height, filling the
    background with mean and putting the image in the top-left.

    Note: this expects im_w <= width and im_h <= height
    """
    def __init__(self, width, height, mean=MEANS, pad_gt=True):
        self.mean = mean
        self.width = width
        self.height = height
        self.pad_gt = pad_gt

    def __call__(self, image, normal):
        im_h, im_w, channels = image.shape

        expand_image = np.zeros(
            (self.height, self.width, channels),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[:im_h, :im_w] = image

        expand_normal = np.zeros((self.height, self.width), dtype=normal.dtype)
        expand_normal[:im_h, :im_w] = normal

        return expand_image, expand_normal


class Resize(object):
    """ If preserve_aspect_ratio is true, this resizes to an approximate area of max_size * max_size """
    # TODO: change the above line of intro

    def __init__(self, resize_gt=True):
        self.resize_gt = resize_gt
        self.max_size = cfg.max_size

    def __call__(self, image, normal):
        img_h, img_w, _ = image.shape

        if img_h != self.max_size or img_w != self.max_size:
            width, height = self.max_size, self.max_size
            image = cv2.resize(image, (width, height))
            normal = cv2.resize(normal, (width, height))
        return image, normal


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, normal=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, normal


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, normal=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, normal


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, normal=None):
        # Don't shuffle the channels please, why would you do this

        # if random.randint(2):
        #     swap = self.perms[random.randint(len(self.perms))]
        #     shuffle = SwapChannels(swap)  # shuffle channels
        #     image = shuffle(image)
        return image, normal


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, normal=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, normal


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, normal=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, normal

class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, normal=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, normal


class ToCV2Image(object):
    def __call__(self, tensor):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0))

class ToTensor(object):
    def __call__(self, cvimage):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1)

class RandomMirror(object):
    def __call__(self, image, normal):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            normal = normal[:, ::-1]
        return image, normal


class RandomFlip(object):
    def __call__(self, image, normal):
        height , _ , _ = image.shape
        if random.randint(2):
            image = image[::-1, :]
            normal = normal[::-1, :]
        return image, normal


class RandomRot90(object):
    def __call__(self, image, normal):
        old_height , old_width , _ = image.shape
        k = random.randint(4)
        image = np.rot90(image,k)
        normal = np.rot90(normal,k)
        return image, normal

class RandomRotation(object):
    def __init__(self, mu=0, sigma=10):
        self.mu = mu
        self.sigma = sigma
        self.angle_sample = np.random.normal(self.mu, self.sigma, 10000)
        print(len(self.angle_sample))

    def __call__(self, image, normal):
        height, width, _ = image.shape
        angle = np.random.choice(self.angle_sample)
        rotate_matrix = cv2.getRotationMatrix2D(center=(int(height/2 - 1),int(width/2 - 1)), angle=angle, scale=1)
        rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(height, width))
        rotated_norm = cv2.warpAffine(src=normal, M=rotate_matrix, dsize=(height, width))

        return rotated_image, rotated_norm


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, normal):
        im = image.copy()
        im, normal = self.rand_brightness(im, normal)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, normal = distort(im, normal)
        return self.rand_light_noise(im, normal)


class BackboneTransform(object):
    """
    Transforms a BRG image made of floats in the range [0, 255] to whatever
    input the current backbone network needs.

    transform is a transform config object (see config.py).
    in_channel_order is probably 'BGR' but you do you, kid.
    """
    # TODO: Check how and what to change of this one.
    def __init__(self, transform, mean, std, in_channel_order):
        self.mean = np.array(mean, dtype=np.float32)
        self.std  = np.array(std,  dtype=np.float32)
        self.transform = transform

        # Here I use "Algorithms and Coding" to convert string permutations to numbers
        self.channel_map = {c: idx for idx, c in enumerate(in_channel_order)}
        self.channel_permutation = [self.channel_map[c] for c in transform.channel_order]

    def __call__(self, img, normal):

        img = img.astype(np.float32)
        normal = normal.astype(np.float32)

        
        if self.transform.normalize:
            img = (img - self.mean) / self.std
            
        elif self.transform.subtract_means:
            img = (img - self.mean)
        elif self.transform.to_float:
            img = img / 255

        img = img[:, :, self.channel_permutation]
        

        return img.astype(np.float32), normal.astype(np.float32)


class RandomMotionBlur(object):
    def __init__(self, lower_degree=6, upper_degree=12, angle=180):
        self.upper_degree = upper_degree
        self.lower_degree = lower_degree
        self.angle = angle
        assert self.lower_degree >= 3
        assert self.lower_degree < self.upper_degree
        assert self.angle >= 0

    def __call__(self, image, normal):

        if random.randint(2) :
            degree = random.randint(self.lower_degree, self.upper_degree)
            angle = random.randint(0, self.angle)

            M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
            motion_blur_kernel = np.diag(np.ones(degree))
            motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

            motion_blur_kernel = motion_blur_kernel / degree
            blurred = cv2.filter2D(image, -1, motion_blur_kernel)

            # convert to uint8
            cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
            blurred = np.array(blurred, dtype=np.uint8)

            return blurred, normal
        else:
            return image, normal


###########################################
#  Augmentation/Transformation in Queue  #
##########################################


def do_nothing(img=None, normal=None):
    return img, normal


def enable_if(condition, obj):
    return obj if condition else do_nothing


class SSDAugmentation(object):
    """ Transform to be used when training. """

    def __init__(self, mean=MEANS, std=STD):
        self.augment = Compose([
            enable_if(cfg.augment.photometric_distort, PhotometricDistort()),
            #enable_if(cfg.augment.random_mirror, RandomMirror()),
            #enable_if(cfg.augment.random_flip, RandomFlip()),
            #enable_if(cfg.augment.random_rot90, RandomRot90()),
            enable_if(cfg.augment.random_rotation, RandomRotation()),
            Resize(resize_gt=True),
            BackboneTransform(cfg.backbone.transform, mean, std, 'BGR')
        ])

    def __call__(self, img, normal=None):
        return self.augment(img, normal)

class BaseTransform(object):
    """ Transorm to be used when evaluating. """

    def __init__(self, mean=MEANS, std=STD):
        self.augment = Compose([
            Resize(resize_gt=False),
            BackboneTransform(cfg.backbone.transform, mean, std, 'BGR')
        ])

    def __call__(self, img, normal=None):
        return self.augment(img, normal)


class FastBaseTransform(torch.nn.Module):
    """
    Transform that does all operations on the GPU for super speed.
    This doesn't suppport a lot of config settings and should only be used for production.
    Maintain this as necessary.
    """

    def __init__(self):
        super().__init__()

        self.mean = torch.Tensor(MEANS).float().cuda()[None, :, None, None]
        self.std  = torch.Tensor( STD ).float().cuda()[None, :, None, None]
        self.transform = cfg.backbone.transform

    def forward(self, img):
        self.mean = self.mean.to(img.device)
        self.std  = self.std.to(img.device)
        
        # img assumed to be a pytorch BGR image with channel order [n, h, w, c]
        img_size = (cfg.max_size, cfg.max_size)

        img = img.permute(0, 3, 1, 2).contiguous()
        #img = F.interpolate(img, img_size, mode='bilinear', align_corners=False)

        if self.transform.normalize:
            img = (img - self.mean) / self.std
        elif self.transform.subtract_means:
            img = (img - self.mean)
        elif self.transform.to_float:
            img = img / 255
        
        if self.transform.channel_order != 'RGB':
            raise NotImplementedError
        
        img = img[:, (2, 1, 0), :, :].contiguous()

        # Return value is in channel order [n, c, h, w] and RGB
        return img


