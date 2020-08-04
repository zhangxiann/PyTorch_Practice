# -*- coding: utf-8 -*-

import os
import random
from PIL import Image
from torch.utils.data import Dataset

random.seed(1)
rmb_label = {"1": 0, "100": 1}
ants_label={'ants':0, 'bees':1}

class RMBDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        rmb面额分类任务的Dataset
        :param data_dir: str, 数据集所在路径
        :param transform: torch.transform，数据预处理
        """
        # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.data_info = self.get_img_info(data_dir)
        self.transform = transform

    def __getitem__(self, index):
        # 通过 index 读取样本
        path_img, label = self.data_info[index]
        # 注意这里需要 convert('RGB')
        img = Image.open(path_img).convert('RGB')     # 0~255
        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，转为tensor等等
        # 返回是样本和标签
        return img, label

    # 返回所有样本的数量
    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_img_info(data_dir):
        data_info = list()
        # data_dir 是训练集、验证集或者测试集的路径
        for root, dirs, _ in os.walk(data_dir):
            # 遍历类别
            # dirs ['1', '100']
            for sub_dir in dirs:
                # 文件列表
                img_names = os.listdir(os.path.join(root, sub_dir))
                # 取出 jpg 结尾的文件
                img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))
                # 遍历图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    # 图片的绝对路径
                    path_img = os.path.join(root, sub_dir, img_name)
                    # 标签，这里需要映射为 0、1 两个类别
                    label = rmb_label[sub_dir]
                    # 保存在 data_info 变量中
                    data_info.append((path_img, int(label)))
        return data_info

class AntsDataset(Dataset):
    def __init__(self,data_dir, transform=None):
        self.label_name={'ants':0, 'bees':1}
        self.data_info=self.get_item_info(data_dir)
        self.transform=transform

    def  __getitem__(self, index):
        path,label=self.data_info[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img=self.transform(img)
        return img, label

    @staticmethod
    def get_item_info(data_dir):
        data_info=list()
        for root,dirs,_ in os.walk(data_dir):
            for sub_dir in dirs:
                img_names=os.listdir(os.path.join(root,sub_dir))
                img_names=list(filter(lambda x:x.endswith('.jpg'), img_names))

                for i in range(len(img_names)):
                    path_img=os.path.join(root,sub_dir,img_names[i])
                    label=ants_label[sub_dir]
                    data_info.append((path_img, int(label)))

        if len(data_info)==0:
            raise Exception('\ndata_dir:{} is a empty dir! please check your image paths!'.format(data_dir))

        return data_info

    def __len__(self):
        return len(self.data_info)












