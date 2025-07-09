import os.path as osp
import pickle
import os
import numpy as np

from .base import BaseDataset


class FC100(BaseDataset):

    def __init__(self, setname, unsupervised, args, augment='none'):
        self.DATA_PATH = osp.join(args.data_root, 'FC100')
        super().__init__(setname, unsupervised, args, augment)
        print("{:} use FC100".format(self.setname))

    def get_data(self, setname):
        wenjianjia  = os.listdir(osp.join(self.DATA_PATH,setname))
        data = []
        label = []
        lb = 0
        for wj in wenjianjia:
            lb_root = osp.join(self.DATA_PATH,setname, wj)
            filelist = os.listdir(lb_root)
            for file in filelist:
                data.append(osp.join(lb_root, file))
                label.append(lb)
            lb += 1
        return data, label

    @property
    def image_size(self):
        return 32
