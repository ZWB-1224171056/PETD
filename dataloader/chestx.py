import os.path as osp
import pandas as pd
from dataloader.base import BaseDataset
import numpy as np
# from .base import ROOT_DIRS, search_dir
from torchvision import transforms
THIS_PATH = osp.dirname(__file__)
ROOT_PATH1 = osp.abspath(osp.join(THIS_PATH, '../model', '..', '..'))
ROOT_PATH2 = osp.abspath(osp.join(THIS_PATH, '../model', '..'))

class chestx(BaseDataset):

    def __init__(self, setname, unsupervised, args, augment='none'):
        self.IMAGE_PATH = osp.join(args.data_root, 'ChestX/images')
        self.target_file = osp.join(args.data_root,  'ChestX/Data_Entry_2017.csv')
        super().__init__(setname, unsupervised, args, augment)
        print("{:} use chestx".format(self.setname))
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
        used_labels = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax"]
        labels_maps = {"Atelectasis": 0, "Cardiomegaly": 1, "Effusion": 2, "Infiltration": 3, "Mass": 4, "Nodule": 5,  "Pneumothorax": 6}

        data_info = pd.read_csv(self.target_file, skiprows=[0], header=None)
        image_names_all = np.asarray(data_info.iloc[:, 0])
        targets_all = np.asarray(data_info.iloc[:, 1])

        for image_name, target in zip(image_names_all, targets_all):
            target = target.split("|")
            if len(target) == 1 and target[0] != "No Finding" and target[0] != "Pneumonia" and target[0] in used_labels:
                path = osp.join(self.IMAGE_PATH, image_name)
                data.append(path)
                label.append(labels_maps[target[0]])
        return data, label

