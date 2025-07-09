import os.path as osp
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from dataloader.transforms import *

_imagenet_pca = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])
}


class BaseDataset(Dataset):
    def __init__(self, setname, unsupervised, args, augment='none'):
        self.args = args
        self.unsupervised = unsupervised
        self.setname = setname
        self.repeat = args.n_shot + args.n_query
        self.augment = augment
        self.wnids = []
        self.data, self.label = self.get_data(self.setname)
        self.wnids = sorted(set(self.label))
        self.num_class = len(set(self.label))
        self.strong_transform = self.get_transform(augment, self.image_size, setname)
        self.ori_transform = self.get_transform('test', self.image_size, setname)
        self.idx = torch.arange(self.__len__())
        self.pesudo_label = torch.randint(0,args.cluster_num,(self.__len__(),),dtype=torch.long)

    def change_assign_label(self,pesudo_label):
        self.pesudo_label = pesudo_label

    # @profile
    def __getitem__(self, i):
        data, label,assign_label = self.data[i], self.label[i],self.pesudo_label[i]
        if isinstance(data, str):
            im = Image.open(data).convert('RGB')
        elif isinstance(data, np.ndarray):
            im = Image.fromarray(data)
        elif isinstance(data, torch.Tensor):
            im = Image.fromarray(data.numpy())
        if self.unsupervised:
            image_list = []
            image_list_ori = []
            image_list_neighboor= []

            image_list_ori.append(self.ori_transform(im))
            if self.augment == 'AMDIM':
                im = self.flip_lr(im)
            for _ in range(self.repeat):
                image_list.append(self.strong_transform(im))
            sample = self.idx[self.pesudo_label == assign_label]

            if len(sample)<self.args.n_query:
                shortfall = self.args.n_query - len(sample)
                indices = torch.randint(0, len(sample), (shortfall,))
                additional_elements =sample[indices]
                sample_sel = torch.cat((sample, additional_elements), dim=0)
            else:
                sample_sel = sample[torch.randperm(sample.size(0))][0:(self.args.n_query)]

            for _,kk in enumerate(sample_sel):
                sample_data = self.data[kk]
                if isinstance(sample_data, str):
                    sample_data = Image.open(sample_data).convert('RGB')
                elif isinstance(sample_data, np.ndarray):
                    sample_data = Image.fromarray(sample_data)
                elif isinstance(sample_data, torch.Tensor):
                    sample_data = Image.fromarray(sample_data.numpy())
                if self.augment == 'AMDIM':
                    sample_data=self.flip_lr(sample_data)
                image_list_neighboor.append(self.strong_transform(sample_data))
            return image_list,image_list_ori,image_list_neighboor, i
        else:
            image = self.ori_transform(im)
        return image, label

    def get_transform(self, augment, image_size, setname):
        if setname == 'train':
            if augment == 'AMDIM':
                transforms_list = self.AMDIM_transforms(image_size)
            elif augment == 'SimCLR':
                transforms_list = self.SimCLR_transforms(image_size)
            elif augment == 'AutoAug':
                transforms_list = self.AutoAug_transforms(image_size)
            elif augment == 'RandAug':
                transforms_list = self.RandAug_transforms(image_size)
            elif augment == 'augment':
                transforms_list = self.augment_transforms(image_size)
            elif augment == 'test':
                transforms_list = self.test_transforms(image_size)
            else:
                raise ValueError(
                    f'Non-supported Augmentation Type: {augment}. Please Revise Data Pre-Processing Scripts.')
        else:
            transforms_list = self.test_transforms(image_size)

        transform = transforms.Compose(
            transforms_list + [
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
            ])
        return transform

    def test_transforms(self, image_size):
        if image_size == 32:
            resize = image_size
        elif image_size ==224:
            resize = int(1.16*image_size)
        else:
            resize = int((92 / 84) * image_size)
        transforms_list = [
            transforms.Resize(resize),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
        return transforms_list

    def SimCLR_transforms(self, image_size):
        s = 1
        color_jitter = transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        transforms_list = [
            transforms.RandomResizedCrop(size=image_size),
            transforms.RandomHorizontalFlip(),  # with 0.5 probability
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ]
        return transforms_list

    def AMDIM_transforms(self, image_size):
        self.flip_lr = transforms.RandomHorizontalFlip(p=0.5)
        transforms_list = [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomApply([RandomTranslateWithReflect(4)], p=0.8),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.25),
            transforms.ToTensor(),

        ]
        return transforms_list

    def RandAug_transforms(self, image_size):
        from .RandAugment import rand_augment_transform
        rgb_mean = (0.485, 0.456, 0.406)
        ra_params = dict(
            translate_const=int(224 * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),
        )
        transforms_list = [
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
            ], p=0.8),
            transforms.RandomApply([GaussianBlur(22)], p=0.5),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10),
                                   ra_params),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ]
        return transforms_list

    def AutoAug_transforms(self, image_size):
        from .autoaug import RandAugment
        transforms_list = [
            RandAugment(2, 12),
            ERandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.8, 0.8, 0.8),
            transforms.ToTensor(),
            Lighting(0.1, _imagenet_pca['eigval'], _imagenet_pca['eigvec']),
        ]
        return transforms_list


    def augment_transforms(self, image_size):
        transforms_list = [
            transforms.RandomResizedCrop(image_size),
            transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        return transforms_list


    def get_data(self, setname):
        csv_path = osp.join(self.split_path, setname + '.csv')
        data, label = self.parse_csv(csv_path)
        return data, label

    def all_img(self):
        image_list = []
        for i,data in enumerate(self.data):
            if isinstance(data, str):
                im = Image.open(data).convert('RGB')
            elif isinstance(data, np.ndarray):
                im = Image.fromarray(data)
            elif isinstance(data, torch.Tensor):
                im = Image.fromarray(data.numpy())
            image = self.ori_transform(im)
            image_list.append(image)
            print(i)
        all_data=torch.stack(image_list)
        return all_data

    @property
    def split_path(self):
        raise NotImplementedError

    def parse_csv(self, csv_path):
        raise NotImplementedError

    @property
    def image_size(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)


