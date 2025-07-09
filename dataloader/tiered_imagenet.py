from __future__ import print_function

import os
import pickle
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader, make_dataset, IMG_EXTENSIONS
from .base import BaseDataset
from PIL import Image
import numpy as np


def search_dir_or_file(dirs, description='data directory'):
    found = None
    for d in dirs:
        if os.path.exists(d):
            found = d
            break
    if found is None:
        raise FileNotFoundError(f'{description} not found')
    print(f'{description} : {found}')
    return found


class TieredImageNet(BaseDataset):
    def __init__(self, setname, unsupervised, args, augment='none'):
        self.IMAGE_PATH = os.path.join(args.data_root, 'tiered_imagenet')
        super().__init__(setname, unsupervised, args, augment)
        print("{:} use TieredImageNet".format(self.setname))
        if setname == "train":
            print('{:}_transform: {:}'.format(self.augment,self.strong_transform))
        else:
            print('test_transform: {:}'.format(self.ori_transform))

    def get_data(self, setname):
        root = search_dir_or_file([os.path.join(self.IMAGE_PATH,setname)])
        classes, class_to_idx = self._find_classes(root)
        samples = make_dataset(root, class_to_idx, IMG_EXTENSIONS, None)
        label = [s[1] for s in samples]
        data = [s[0] for s in samples]
        return data, label


    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


    def __len__(self):
        return len(self.data)

    @property
    def image_size(self):
        return 84
