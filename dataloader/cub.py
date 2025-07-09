import os.path as osp

from dataloader.base import BaseDataset
# from .base import ROOT_DIRS, search_dir


class CUB(BaseDataset):

    def __init__(self, setname, unsupervised, args, augment='none'):
        self.IMAGE_PATH = osp.join(args.data_root, 'cub/images')
        self.SPLIT_PATH = osp.join(args.data_root, 'cub/split')
        self.CACHE_PATH = osp.join(args.data_root, '.cache/')
        super().__init__(setname, unsupervised, args, augment)
        print("{:} use CUB".format(self.setname))
        if setname == "train":
            print('{:}_transform: {:}'.format(self.augment,self.strong_transform))
        else:
            print('test_transform: {:}'.format(self.ori_transform))
    @property
    def image_size(self):
        return 84

    @property
    def eval_setting(self):
        return [(5, 1), (5, 5), (5, 20)]

    @property
    def split_path(self):
        return self.SPLIT_PATH

    @property
    def cache_path(self):
        return self.CACHE_PATH

    def parse_csv(self, txt_path):
        data = []
        label = []
        lb = -1
        lines = [x.strip() for x in open(txt_path, 'r').readlines()][1:]

        for l in lines:
            context = l.split(',')
            name = context[0]
            wnid = context[1]
            path = osp.join(self.IMAGE_PATH, wnid,name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1

            data.append(path)
            label.append(lb)
        return data, label
