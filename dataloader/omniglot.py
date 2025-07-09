from __future__ import print_function

import os
from torchvision.datasets.folder import default_loader, make_dataset, IMG_EXTENSIONS
from .base import BaseDataset
from torchvision import transforms

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


class omniglot(BaseDataset):
    def __init__(self, setname, unsupervised, args, augment='none'):
        super().__init__(setname, unsupervised, args, augment)
        print("{:} use omniglot".format(self.setname))
        self.strong_transform = transforms.Compose(
            [transforms.RandomResizedCrop(28, scale=(0.2, 1.), interpolation=transforms.InterpolationMode.BICUBIC),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.92206], std=[0.08426])])
        self.ori_transform = transforms.Compose([transforms.Resize(self.image_size), transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.92206], std=[0.08426])])

    def get_data(self, setname):
        root = search_dir_or_file([os.path.join('./data/omniglot',setname)])
        classes, class_to_idx = self._find_classes(root)
        samples = make_dataset(root, class_to_idx, IMG_EXTENSIONS, None)
        # self.samples = samples
        label = [s[1] for s in samples]
        data = [s[0] for s in samples]
        return data, label


    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


    @property
    def image_size(self):
        return 28

