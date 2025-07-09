import os.path as osp
import os
import PIL
from PIL import Image

import pickle
import numpy as np
from .base import BaseDataset
from torchvision import transforms


def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)

    return label2inds


class CIFARFS(BaseDataset):

    def __init__(self, setname, unsupervised, args, augment='none'):
        self.DATA_PATH = osp.join(args.data_root, 'CIFAR_FS')
        self.SPLIT_PATH = osp.join(args.data_root, 'CIFAR_FS/split')
        super().__init__(setname, unsupervised, args, augment)
        print("{:} use CIFARFS".format(self.setname))
        if setname == "train":
            print('{:}_transform: {:}'.format(self.augment,self.strong_transform))
        else:
            print('test_transform: {:}'.format(self.ori_transform))

    def get_data(self, setname):
        f = open(osp.join(self.SPLIT_PATH, setname + ".txt"), "r")
        lines = f.readlines()  
        f.close()
        data = []
        label = []
        lb = 0
        for l in lines:
            lb_root = osp.join(self.DATA_PATH, l.strip())
            filelist = os.listdir(lb_root)
            for file in filelist:
                data.append(osp.join(lb_root, file))
                label.append(lb)
            lb += 1
        return data, label

    @property
    def image_size(self):
        return 32
