"""Adapted from:
    @dbolya yolact: https://github.com/dbolya/yolact/data/config.py
    Licensed under The MIT License [see LICENSE for details]
"""

from models.backbone import ResNetBackbone
import torch

COLORS = (
    (244, 67, 54),
    (233, 30, 99),
    (156, 39, 176),
    (103, 58, 183),
    (63, 81, 181),
    (33, 150, 243),
    (3, 169, 244),
    (0, 188, 212),
    (0, 150, 136),
    (76, 175, 80),
    (139, 195, 74),
    (205, 220, 57),
    (255, 235, 59),
    (255, 193, 7),
    (255, 152, 0),
    (255, 87, 34),
    (121, 85, 72),
    (158, 158, 158),
    (96, 125, 139),
)


# These are in BGR and are for ImageNet
MEANS = (103.94, 116.78, 123.68)
STD = (57.38, 57.12, 58.40)

PLANE_CLASSES = ('plane',)
PLANE_LABEL_MAP = {1: 1}

# ----------------------- CONFIG CLASS ----------------------- #


class Config(object):
    '''
    Holds the configuration for anything you want it to.
    To get the currently active config, call get_cfg().

    To use, just do cfg.x instead of cfg['x'].
    I made this because doing cfg['x'] all the time is dumb.
    '''

    def __init__(self, config_dict):
        for key, val in config_dict.items():
            self.__setattr__(key, val)

    def copy(self, new_config_dict={}):
        '''
        Copies this config into a new config object, making
        the changes given by new_config_dict.
        '''

        ret = Config(vars(self))

        for key, val in new_config_dict.items():
            ret.__setattr__(key, val)

        return ret

    def replace(self, new_config_dict):
        '''
        Copies new_config_dict into this config object.
        Note: new_config_dict can also be a config object.
        '''
        if isinstance(new_config_dict, Config):
            new_config_dict = vars(new_config_dict)

        for key, val in new_config_dict.items():
            self.__setattr__(key, val)

    def print(self):
        for k, v in vars(self).items():
            print(k, ' = ', v)

# ----------------------- DATASETS ----------------------- #

dataset_base = Config(
    {
    'name': 'Holicity Dataset',

    # Training images and annotations
    'root_path': './holicity',
    'train_split': 'holicity_train.tar',

    # Validation images and annotations.
    'valid_split': 'holicity_valid.tar',

    # Whether or not to load GT. If this is False, eval.py quantitative evaluation won't work.
    'has_gt': True,

    'train_length': 45600,
    'valid_length': 2496,
}
)

dataset_server = Config(
    {
    'name': 'Holicity Dataset',

    # Training images and annotations
    'root_path': '/netscratch/xie/holicity',
    'train_split': 'holicity_train.tar',

    # Validation images and annotations.
    'valid_split': 'holicity_valid.tar',

    # Whether or not to load GT. If this is False, eval.py quantitative evaluation won't work.
    'has_gt': True,

    'train_length': 45600,
    'valid_length': 2496,
}
)


# ----------------------- DATA AUGMENTATION ---------------- #

data_augment = Config(
    {
    # SSD data augmentation parameters
    # Randomize hue, vibrance, etc.
    'photometric_distort': True,
    # Mirror the image with a probability of 1/2
    'random_mirror': True,
    # Flip the image vertically with a probability of 1/2
    'random_flip': False,
    # With uniform probability, rotate the image [0,90,180,270] degrees
    'random_rot90': False,
    # With mothin blur
    'motion_blur': True,
    # With Gaussian nosie
    'gaussian_noise': False,
    }
)

# ----------------------- TRANSFORMS ----------------------- #

resnet_transform = Config(
    {
        'channel_order': 'RGB',
        'normalize': True,
        'subtract_means': False,
        'to_float': False,
    }
)


# ----------------------- BACKBONES ----------------------- #

backbone_base = Config(
    {
        'name': 'Base Backbone',
        'path': 'path/to/pretrained/weights',
        'type': object,
        'args': tuple(),
        'transform': resnet_transform,
        'selected_layers': list(),
        'dataset': dataset_base,
    }
)

resnet101_backbone = backbone_base.copy(
    {
        'name': 'ResNet101',
        'path': 'resnet101_reducedfc.pth',
        'type': ResNetBackbone,
        'args': ([3, 4, 23, 3],),
        'transform': resnet_transform,
        'selected_layers': list(range(3, 7)),
    }
)

resnet101_dcn_inter3_backbone = resnet101_backbone.copy(
    {
        'name': 'ResNet101_DCN_Interval3', 'args': ([3, 4, 23, 3], [0, 4, 23, 3], 3),
        'selected_layers': list(range(2, 4)),
    }
)

resnet50_backbone = resnet101_backbone.copy(
    {
        'name': 'ResNet50',
        'path': 'resnet50-19c8e357.pth',
        'type': ResNetBackbone,
        'args': ([3, 4, 6, 3],),
        'transform': resnet_transform,
    }
)

resnet50_dcnv2_backbone = resnet50_backbone.copy(
    {
        'name': 'ResNet50_DCNv2', 'args': ([3, 4, 6, 3], [0, 4, 6, 3]),
        'selected_layers': list(range(2, 4)),
    }
)

# ----------------------- NormalNet CONFIGS ----------------------- #
NormalNet_base_config = Config(
    {
        'name': 'NormalNet_base',

        # Dataset Settings
        'dataset': dataset_base,

        # Data Augmentations
        'augment': data_augment,
        
        # Training Settings
        'max_iter': 114000,
        'lr_steps': (57000, 85500),
        # dw' = momentum * dw - lr * (grad + decay * w)
        'lr': 1e-4,
        'momentum': 0.9,
        'decay': 5e-4,

        'freeze_bn': False,
        # Warm Up Learning Rate
        'lr_warmup_init': 1e-5,
        'lr_warmup_until': 500,
        # For each lr step, what to multiply the lr with
        'gamma': 0.1,

        # A list of settings to apply after the specified iteration. Each element of the list should look like
        # (iteration, config_dict) where config_dict is a dictionary you'd pass into a config object's init.
        'delayed_settings': [],

        # Backbone Settings
        'backbone': resnet50_dcnv2_backbone.copy(
            {
                'selected_layers': list(range(2, 4)),
            }
        ),

        # Loss Settings
        'depth_weight': 0.5,

        # Image Size
        'max_size': 512,
        # Device
        'device': 'cuda',
        # Whether or not to preserve aspect ratio when resizing the image.
        # If True, this will resize all images to be max_size^2 pixels in area while keeping aspect ratio.
        # If False, all images are resized to max_size x max_size
        'preserve_aspect_ratio': False,
    }
)


# Default config
cfg = NormalNet_base_config.copy()

def set_cfg(config_name: str):
    ''' Sets the active config. Works even if cfg is already imported! '''
    global cfg

    cfg.replace(eval(config_name))

    if cfg.name is None:
        cfg.name = config_name.split('_config')[0]

def set_dataset(dataset_name: str):
    ''' Sets the dataset of the current config. '''
    cfg.dataset = eval(dataset_name)



# Just for Testing
if __name__ == "__main__" :
    set_cfg('resnet101_dcn_inter3_backbone')
    cfg.print()

