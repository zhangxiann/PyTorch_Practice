# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import random
import math
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
from enviroments import rmb_split_dir
from lesson2.transforms.addPepperNoise import AddPepperNoise
def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

set_seed(1)  # 设置随机种子

# 参数设置
MAX_EPOCH = 10
BATCH_SIZE = 1
LR = 0.01
log_interval = 10
val_interval = 1
rmb_label = {"1": 0, "100": 1}

#对 tensor 进行反标准化操作，并且把 tensor 转换为 image，方便可视化。
def transform_invert(img_, transform_train):
    """
    将data 进行反transfrom操作
    :param img_: tensor
    :param transform_train: torchvision.transforms
    :return: PIL image
    """

    # 如果有标准化操作
    if 'Normalize' in str(transform_train):
        # 取出标准化的 transform
        norm_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform_train.transforms))
        # 取出均值
        mean = torch.tensor(norm_transform[0].mean, dtype=img_.dtype, device=img_.device)
        # 取出标准差
        std = torch.tensor(norm_transform[0].std, dtype=img_.dtype, device=img_.device)
        # 乘以标准差，加上均值
        img_.mul_(std[:, None, None]).add_(mean[:, None, None])

    # 把 C*H*W 变为 H*W*C
    img_ = img_.transpose(0, 2).transpose(0, 1)  # C*H*W --> H*W*C
    # 把 0~1 的值变为 0~255
    img_ = np.array(img_) * 255

    # 如果是 RGB 图
    if img_.shape[2] == 3:
        img_ = Image.fromarray(img_.astype('uint8')).convert('RGB')
        # 如果是灰度图
    elif img_.shape[2] == 1:
        img_ = Image.fromarray(img_.astype('uint8').squeeze())
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_.shape[2]) )

    return img_


norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    # 缩放到 (224, 224) 大小，会拉伸
    transforms.Resize((224, 224)),

    # 1 CenterCrop 中心裁剪
    # transforms.CenterCrop(512),     # 512
    # transforms.CenterCrop(196),     # 512

    # 2 RandomCrop
    # transforms.RandomCrop(224, padding=16),
    # transforms.RandomCrop(224, padding=(16, 64)),
    # transforms.RandomCrop(224, padding=16, fill=(255, 0, 0)),
    # transforms.RandomCrop(512, pad_if_needed=True),   # pad_if_needed=True
    # transforms.RandomCrop(224, padding=64, padding_mode='edge'),
    # transforms.RandomCrop(224, padding=64, padding_mode='reflect'),
    # transforms.RandomCrop(1024, padding=1024, padding_mode='symmetric'),

    # 3 RandomResizedCrop
    # transforms.RandomResizedCrop(size=224, scale=(0.08, 1)),
    # transforms.RandomResizedCrop(size=224, scale=(0.5, 0.5)),

    # 4 FiveCrop
    # transforms.FiveCrop(112),
    # 返回的是 tuple，因此需要转换为 tensor
    # transforms.Lambda(lambda crops: torch.stack([(transforms.ToTensor()(crop)) for crop in crops])),

    # 5 TenCrop
    # transforms.TenCrop(112, vertical_flip=False),
    # transforms.Lambda(lambda crops: torch.stack([(transforms.ToTensor()(crop)) for crop in crops])),

    # 1 Horizontal Flip
    # transforms.RandomHorizontalFlip(p=1),

    # 2 Vertical Flip
    # transforms.RandomVerticalFlip(p=1),

    # 3 RandomRotation
    # transforms.RandomRotation(90),
    # transforms.RandomRotation((90), expand=True),
    # transforms.RandomRotation(30, center=(0, 0)),
    # transforms.RandomRotation(30, center=(0, 0), expand=True),   # expand only for center rotation


    # 1 Pad
    # transforms.Pad(padding=32, fill=(255, 0, 0), padding_mode='constant'),
    # transforms.Pad(padding=(8, 64), fill=(255, 0, 0), padding_mode='constant'),
    # transforms.Pad(padding=(8, 16, 32, 64), fill=(255, 0, 0), padding_mode='constant'),
    # transforms.Pad(padding=(8, 16, 32, 64), fill=(255, 0, 0), padding_mode='symmetric'),

    # 2 ColorJitter
    # transforms.ColorJitter(brightness=0.5),
    # transforms.ColorJitter(contrast=0.5),
    # transforms.ColorJitter(saturation=0.5),
    # transforms.ColorJitter(hue=0.3),

    # 3 Grayscale
    # transforms.Grayscale(num_output_channels=3),

    # 4 Affine
    # transforms.RandomAffine(degrees=30),
    # transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), fillcolor=(255, 0, 0)),
    # transforms.RandomAffine(degrees=0, scale=(0.7, 0.7)),
    # transforms.RandomAffine(degrees=0, shear=(0, 0, 0, 45)),
    # transforms.RandomAffine(degrees=0, shear=90, fillcolor=(255, 0, 0)),

    # 5 Erasing
    # transforms.ToTensor(),
    # transforms.RandomErasing(p=1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=(254/255, 0, 0)),
    # transforms.RandomErasing(p=1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='fads43'),

    # 1 RandomChoice
    # transforms.RandomChoice([transforms.RandomVerticalFlip(p=1), transforms.RandomHorizontalFlip(p=1)]),

    # 2 RandomApply
    # transforms.RandomApply([transforms.RandomAffine(degrees=0, shear=45, fillcolor=(255, 0, 0)),
    #                         transforms.Grayscale(num_output_channels=3)], p=0.5),
    # 3 RandomOrder
    # transforms.RandomOrder([transforms.RandomRotation(15),
    #                         transforms.Pad(padding=32),
    #                         transforms.RandomAffine(degrees=0, translate=(0.01, 0.1), scale=(0.9, 1.1))]),

    AddPepperNoise(0.9, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

path_img=os.path.join(rmb_split_dir, "train", "100","0A4DSPGE.jpg")
img = Image.open(path_img).convert('RGB')  # 0~255
img=transforms.Resize((224, 224))(img)
img_tensor = train_transform(img)



## 展示单张图片
# 这里把转换后的 tensor 再转换为图片
convert_img=transform_invert(img_tensor, train_transform)
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.imshow(convert_img)
plt.show()
plt.pause(0.5)
plt.close()


## 展示 FiveCrop 和 TenCrop 的图片
# ncrops, c, h, w = img_tensor.shape
# columns=2 # 两列
# rows= math.ceil(ncrops/2) # 计算多少行
# # 把每个 tensor ([c,h,w]) 转换为 image
# for i in range(ncrops):
#     img = transform_invert(img_tensor[i], train_transform)
#     plt.subplot(rows, columns, i+1)
#     plt.imshow(img)
# plt.show()






