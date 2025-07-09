from .mini_imagenet import MiniImageNet
from .tiered_imagenet import TieredImageNet
from .cifarfs import CIFARFS
from .fc100 import FC100
from .cub import CUB
from .ISIC import ISIC
from .EuroSAT import EuroSAT
from .chestx import chestx
from .CropDiseases import CropDiseases
from .omniglot import omniglot
from .samplers import *
dataset_dict = {
    'MiniImageNet': MiniImageNet,
    'TieredImageNet': TieredImageNet,
    'CIFAR-FS': CIFARFS,
    'FC100': FC100,
    'CUB':CUB,
    'ISIC':ISIC,
    'EuroSAT':EuroSAT,
    'chestx':chestx,
    'CropDiseases':CropDiseases,
    'omniglot':omniglot
}