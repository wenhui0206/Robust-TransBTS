import os
import torch
from torch.utils.data import Dataset
import random
import numpy as np
from torchvision.transforms import transforms
import pickle
from scipy import ndimage


def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


class MaxMinNormalization(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        Max = np.max(image)
        Min = np.min(image)
        image = (image - Min) / (Max - Min)

        return {'image': image, 'label': label}


class Random_Flip(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        if random.random() < 0.5:
            image = np.flip(image, 0)
            label = np.flip(label, 0)
        if random.random() < 0.5:
            image = np.flip(image, 1)
            label = np.flip(label, 1)
        if random.random() < 0.5:
            image = np.flip(image, 2)
            label = np.flip(label, 2)

        return {'image': image, 'label': label}


class Random_Crop(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        H = random.randint(0, 240 - 128)
        W = random.randint(0, 240 - 128)
        D = random.randint(0, 160 - 128)

        image = image[H: H + 128, W: W + 128, D: D + 128, ...]
        label = label[..., H: H + 128, W: W + 128, D: D + 128]

        return {'image': image, 'label': label}


class Random_intencity_shift(object):
    def __call__(self, sample, factor=0.1):
        image = sample['image']
        label = sample['label']

        scale_factor = np.random.uniform(1.0-factor, 1.0+factor, size=[1, image.shape[1], 1, image.shape[-1]])
        shift_factor = np.random.uniform(-factor, factor, size=[1, image.shape[1], 1, image.shape[-1]])

        image = image*scale_factor+shift_factor

        return {'image': image, 'label': label}


class Random_rotate(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        angle = round(np.random.uniform(-10, 10), 2)
        image = ndimage.rotate(image, angle, axes=(0, 1), reshape=False)
        label = ndimage.rotate(label, angle, axes=(0, 1), reshape=False)

        return {'image': image, 'label': label}


class Pad(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        image = np.pad(image, ((0, 0), (0, 0), (0, 5), (0, 0)), mode='constant')
        label = np.pad(label, ((0, 0), (0, 0), (0, 5)), mode='constant')
        return {'image': image, 'label': label}
    #(240,240,155)>(240,240,160)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample['image']
        image = np.ascontiguousarray(image.transpose(3, 0, 1, 2))
        label = sample['label']
        label = np.ascontiguousarray(label)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()

        return {'image': image, 'label': label}


def transform(sample):
    trans = transforms.Compose([
        Pad(),
        # Random_rotate(),  # time-consuming
        Random_Crop(),
        Random_Flip(),
        Random_intencity_shift(),
        ToTensor()
    ])

    return trans(sample)


def transform_valid(sample):
    trans = transforms.Compose([
        # Pad(),
        # MaxMinNormalization(),
        ToTensor()
    ])

    return trans(sample)


class BraTS(Dataset):
    def __init__(self, names, root='', mode='train'):
        self.lines = []
        paths = []
        for name in names:
            path = os.path.join(root, name, name + '_')
            paths.append(path)
            self.lines.append(name)
        self.mode = mode
        self.names = names
        self.paths = paths


    def __getitem__(self, item):
        path = self.paths[item]
        if self.mode == 'train':
            image, label = pkload(path + 'data_f32b0.pkl')
            sample = {'image': image, 'label': label}
            sample = transform(sample)
            return sample['image'], sample['label']
        elif self.mode == 'valid':
            image, label = pkload(path + 'data_f32b0.pkl')
            image = np.pad(image, ((0, 0), (0, 0), (0, 5), (0, 0)), mode='constant')
            # only pad the image, do not pad the target
            sample = {'image': image, 'label': label}
            sample = transform_valid(sample)
            return sample['image'], sample['label']
        else:
            image = pkload(path + 'data_f32b0.pkl')
            image = np.pad(image, ((0, 0), (0, 0), (0, 5), (0, 0)), mode='constant')
            image = np.ascontiguousarray(image.transpose(3, 0, 1, 2))
            image = torch.from_numpy(image).float()
            return image


    def __len__(self):
        return len(self.names)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]


class BraTS_noisy(Dataset):
    def __init__(self, names, root='', mode='train', corrupt_r=0.5, fold=0):
        self.lines = []
        paths = []
        for name in names:
            path = os.path.join(root, name, name + '_')
            paths.append(path)
            self.lines.append(name)
        self.mode = mode
        self.names = names
        self.paths = paths
        self.pct_train = int(10 - corrupt_r*10)
        if fold > 0:
            self.fold_name = '_f' + str(fold)
        else:
            self.fold_name = ''


    def __getitem__(self, item):
        path = self.paths[item]
        if self.mode == 'train':
            image, _ = pkload(path + 'data_f32b0.pkl')
            # label = pkload(path + 'noisy_label_f32b0.pkl') # noisy label
            label = pkload(path + 'noisy_label_0.' + str(self.pct_train) + self.fold_name + '.pkl')
            sample = {'image': image, 'label': label}
            sample = transform(sample)
            return sample['image'], sample['label']
        elif self.mode == 'valid':
            image, label = pkload(path + 'data_f32b0.pkl')
            image = np.pad(image, ((0, 0), (0, 0), (0, 5), (0, 0)), mode='constant')
            # only pad the image, do not pad the target
            sample = {'image': image, 'label': label}
            sample = transform_valid(sample)
            return sample['image'], sample['label']
        else:
            image = pkload(path + 'data_f32b0.pkl')
            image = np.pad(image, ((0, 0), (0, 0), (0, 5), (0, 0)), mode='constant')
            image = np.ascontiguousarray(image.transpose(3, 0, 1, 2))
            image = torch.from_numpy(image).float()
            return image


    def __len__(self):
        return len(self.names)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]

