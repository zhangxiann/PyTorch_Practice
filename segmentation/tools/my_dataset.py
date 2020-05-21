# -*- coding: utf-8 -*-

import numpy as np
import torch
import os
import random
from PIL import Image
from torch.utils.data import Dataset

random.seed(1)
rmb_label = {"1": 0, "100": 1}


class RMBDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        rmb面额分类任务的Dataset
        :param data_dir: str, 数据集所在路径
        :param transform: torch.transform，数据预处理
        """
        self.label_name = {"1": 0, "100": 1}
        self.data_info = self.get_img_info(data_dir)  # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.transform = transform

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')     # 0~255

        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，转为tensor等等

        return img, label

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_img_info(data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            # 遍历类别
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))

                # 遍历图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = rmb_label[sub_dir]
                    data_info.append((path_img, int(label)))

        return data_info


class AntsDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.label_name = {"ants": 0, "bees": 1}
        self.data_info = self.get_img_info(data_dir)
        self.transform = transform

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.data_info)

    def get_img_info(self, data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            # 遍历类别
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))

                # 遍历图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = self.label_name[sub_dir]
                    data_info.append((path_img, int(label)))

        if len(data_info) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(data_dir))
        return data_info


# 用于训练图像分割，人像抠图
class PortraitDataset(Dataset):
    def __init__(self, data_dir, transform=None, in_size = 224):
        super(PortraitDataset, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.label_path_list = list()
        self.in_size = in_size

        # 获取mask的path
        self._get_img_path()

    def __getitem__(self, index):

        path_label = self.label_path_list[index]
        path_img = path_label[:-10] + ".png"

        img_pil = Image.open(path_img).convert('RGB')
        img_pil = img_pil.resize((self.in_size, self.in_size), Image.BILINEAR)
        img_hwc = np.array(img_pil)
        img_chw = img_hwc.transpose((2, 0, 1))

        # 读取 label 的灰度图像
        label_pil = Image.open(path_label).convert('L')
        label_pil = label_pil.resize((self.in_size, self.in_size), Image.NEAREST)
        label_hw = np.array(label_pil)
        label_chw = label_hw[np.newaxis, :, :]
        # 由于我们是人像抠图，这里进行标签 2 值化，最终标签只有 0 和 1
        label_hw[label_hw != 0] = 1

        if self.transform is not None:
            img_chw_tensor = torch.from_numpy(self.transform(img_chw.numpy())).float()
            label_chw_tensor = torch.from_numpy(self.transform(label_chw.numpy())).float()
        else:
            img_chw_tensor = torch.from_numpy(img_chw).float()
            label_chw_tensor = torch.from_numpy(label_chw).float()

        return img_chw_tensor, label_chw_tensor

    def __len__(self):
        return len(self.label_path_list)

    def _get_img_path(self):
        file_list = os.listdir(self.data_dir)
        file_list = list(filter(lambda x: x.endswith("_matte.png"), file_list))
        path_list = [os.path.join(self.data_dir, name) for name in file_list]
        random.shuffle(path_list)
        if len(path_list) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(data_dir))
        self.label_path_list = path_list