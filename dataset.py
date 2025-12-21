import random

from torchvision import transforms
from PIL import Image
import os
import torch
import glob
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, ImageFolder
import numpy as np
import torch.multiprocessing
import json

# import imgaug.augmenters as iaa
# from perlin import rand_perlin_2d_np

torch.multiprocessing.set_sharing_strategy('file_system')


def get_data_transforms(size, isize, mean_train=None, std_train=None):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])
    return data_transforms, gt_transforms



class RealIADDataset(torch.utils.data.Dataset):
    def __init__(self, root, category, transform, gt_transform, phase):
        self.transform = transform
        self.gt_transform = gt_transform
        self.phase = phase

        json_path = os.path.join(root, 'meta.json')
        with open(json_path, encoding='utf-8') as file:
            meta = json.load(file)

        meta_section = meta.get(phase, meta.get("test", {}))
        samples = meta_section.get(category, [])

        self.img_paths, self.gt_paths, self.labels, self.types = [], [], [], []

        for sample in samples:
            img_rel = sample.get('img_path', '')
            mask_rel = sample.get('mask_path', None)
            specie_name = sample.get('specie_name', '')
            anomaly_flag = sample.get('anomaly')
            if anomaly_flag is None:
                anomaly_flag = 0 if specie_name == 'OK' else 1
            label = bool(anomaly_flag)

            img_abs = os.path.join(root, img_rel)
            mask_abs = os.path.join(root, mask_rel) if (mask_rel not in [None, ""] and label) else None

            self.img_paths.append(img_abs)
            self.gt_paths.append(mask_abs)
            self.labels.append(label)
            self.types.append(specie_name if specie_name != '' else ('OK' if not label else 'NG'))

        self.img_paths = np.array(self.img_paths)
        self.gt_paths = np.array(self.gt_paths)
        self.labels = np.array(self.labels)
        self.types = np.array(self.types)
        self.cls_idx = 0

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        if self.phase == 'train':
            return img, label

        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            if gt is None or not os.path.exists(gt):
                gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
            else:
                gt = Image.open(gt).convert('L')
                gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path

class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.cls_idx = 0

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return np.array(img_tot_paths), np.array(gt_tot_paths), np.array(tot_labels), np.array(tot_types)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path
