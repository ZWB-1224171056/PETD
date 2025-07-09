import os.path as osp
import pandas as pd
from dataloader.base import BaseDataset
import numpy as np
# from .base import ROOT_DIRS, search_dir
from torchvision import transforms
THIS_PATH = osp.dirname(__file__)
ROOT_PATH1 = osp.abspath(osp.join(THIS_PATH, '../model', '..', '..'))
ROOT_PATH2 = osp.abspath(osp.join(THIS_PATH, '../model', '..'))

class ISIC(BaseDataset):

    def __init__(self, setname, unsupervised, args, augment='none'):
        self.IMAGE_PATH = osp.join(args.data_root, 'ISIC2018/ISIC2018_Task3_Training_Input')
        self.target_file = osp.join(args.data_root,  'ISIC2018/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv')
        super().__init__(setname, unsupervised, args, augment)
        print("{:} use ISIC".format(self.setname))
        if setname == "train":
            print('{:}_transform: {:}'.format(self.augment,self.strong_transform))
        else:
            print('test_transform: {:}'.format(self.ori_transform))
    @property
    def image_size(self):
        return 224


    def get_data(self, setname):
        data = []
        label = []
        data_info = pd.read_csv(self.target_file, skiprows=[0], header=None)
        image_names = np.asarray(data_info.iloc[:, 0])
        targets = np.asarray(data_info.iloc[:, 1:])
        targets = (targets != 0).argmax(axis=1)
        for image_name,target in zip(image_names,targets):
            path = osp.join(self.IMAGE_PATH, image_name+".jpg")
            data.append(path)
            label.append(target)
        return data, label

