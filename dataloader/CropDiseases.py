import os.path as osp
import os

from dataloader.base import BaseDataset
from torchvision import transforms


# THIS_PATH = osp.dirname(__file__)
# ROOT_PATH = osp.abspath(osp.join(THIS_PATH, '..', '..', '..'))


def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)

    return label2inds


class CropDiseases(BaseDataset):

    def __init__(self, setname, unsupervised, args, augment='none'):
        self.DATA_PATH = osp.join(args.data_root, 'CropDiseases/train')
        super().__init__(setname, unsupervised, args, augment)
        print("{:} use CropDiseases".format(self.setname))
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
        lb = 0
        for i, (root, _, filenames) in enumerate(os.walk('data/CropDiseases/train')):
            if i == 0:
                continue
            for image_name in filenames:
                path = osp.join(root, image_name)
                data.append(path)
                label.append(i-1)
            print(len(label))
        return data, label

if __name__=="__main__":
    for root, dirnames, filenames in os.walk('../data/EuroSAT'):
        print(root)
        print(dirnames)
        print(filenames)
