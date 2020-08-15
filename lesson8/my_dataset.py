# -*- coding: utf-8 -*-
"""
各数据集的Dataset定义
"""
import numpy as np
import torch
import os
import random
from PIL import Image
from torch.utils.data import Dataset

random.seed(1)
rmb_label = {"1": 0, "100": 1}


class PennFudanDataset(object):
    def __init__(self, data_dir, transforms):

        self.data_dir = data_dir
        self.transforms = transforms
        self.img_dir = os.path.join(data_dir, "PNGImages")
        self.txt_dir = os.path.join(data_dir, "Annotation")
        # 保存所有图片的文件名，后面用于查找对应的 txt 标签文件
        self.names = [name[:-4] for name in list(filter(lambda x: x.endswith(".png"), os.listdir(self.img_dir)))]

    def __getitem__(self, index):
        """
        返回img和target
        :param idx:
        :return:
        """

        name = self.names[index]
        path_img = os.path.join(self.img_dir, name + ".png")
        path_txt = os.path.join(self.txt_dir, name + ".txt")

        # load img
        img = Image.open(path_img).convert("RGB")

        # load boxes and label
        f = open(path_txt, "r")
        import re
        # 查找每一行是否有数字，有数字的则是带有标签的行
        points = [re.findall(r"\d+", line) for line in f.readlines() if "Xmin" in line]
        boxes_list = list()
        for point in points:
            box = [int(p) for p in point]
            boxes_list.append(box[-4:])
        boxes = torch.tensor(boxes_list, dtype=torch.float)
        labels = torch.ones((boxes.shape[0],), dtype=torch.long)

        # iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target = {}
        # 组成 label，是一个 dict，包括 boxes 和 labels
        target["boxes"] = boxes
        target["labels"] = labels
        # target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        if len(self.names) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(data_dir))
        return len(self.names)

class CelebADataset(object):

    def __init__(self, data_dir, transforms):

        self.data_dir = data_dir
        self.transform = transforms
        self.img_names = [name for name in list(filter(lambda x: x.endswith(".jpg"), os.listdir(self.data_dir)))]

    def __getitem__(self, index):
        path_img = os.path.join(self.data_dir, self.img_names[index])
        img = Image.open(path_img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        if len(self.img_names) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(self.data_dir))
        return len(self.img_names)