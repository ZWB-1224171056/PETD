import os.path as osp
from tqdm import tqdm

from dataloader.base import BaseDataset

THIS_PATH = osp.dirname(__file__)
ROOT_PATH = osp.abspath(osp.join(THIS_PATH, '../model', '..'))
ROOT_PATH2 = osp.abspath(osp.join(THIS_PATH, '../model', '..', '..', '..'))
data_dirs = [
    'mini_imagenet',
    'mini-imagenet',
]


class MiniImageNet(BaseDataset):

    def __init__(self, setname, unsupervised, args, augment='none'):
        self.IMAGE_PATH = osp.join(args.data_root, 'mini_imagenet/images')
        self.SPLIT_PATH = osp.join(args.data_root, 'mini_imagenet/split')
        super().__init__(setname, unsupervised, args, augment)
        print("{:} use MiniImageNet".format(self.setname))
        if setname == "train":
            print('{:}_transform: {:}'.format(self.augment,self.strong_transform))
        else:
            print('test_transform: {:}'.format(self.ori_transform))
    @property
    def image_size(self):
        return 84

    @property
    def split_path(self):
        return self.SPLIT_PATH

    def parse_csv(self, csv_path):
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
        data = []
        label = []
        lb = -1

        for l in tqdm(lines, ncols=64):
            name, wnid = l.split(',')
            path = osp.join(self.IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        return data, label
