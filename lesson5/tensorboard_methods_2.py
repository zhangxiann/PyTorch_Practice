# -*- coding:utf-8 -*-
import os
import torch
import time
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.utils as vutils
from lesson2.rmb_classification.tools.my_dataset import RMBDataset
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from common_tools import set_seed
from lesson2.rmb_classification.model.lenet import LeNet
import enviroments
from torchsummary import summary

set_seed(1)  # 设置随机种子


# ----------------------------------- 3 image -----------------------------------
flag = 0
# flag = 1
if flag:

    writer = SummaryWriter(comment='test_your_comment', filename_suffix="_test_your_filename_suffix")

    # img 1     random
    # 随机噪声的图片
    fake_img = torch.randn(3, 512, 512)
    writer.add_image("fake_img", fake_img, 1)
    time.sleep(1)

    # img 2     ones
    # 像素值全为 1 的图片，会乘以 255，所以是白色的图片
    fake_img = torch.ones(3, 512, 512)
    time.sleep(1)
    writer.add_image("fake_img", fake_img, 2)

    # img 3     1.1
    # 像素值全为 1.1 的图片，不会乘以 255，所以是黑色的图片
    fake_img = torch.ones(3, 512, 512) * 1.1
    time.sleep(1)
    writer.add_image("fake_img", fake_img, 3)

    # img 4     HW
    fake_img = torch.rand(512, 512)
    writer.add_image("fake_img", fake_img, 4, dataformats="HW")

    # img 5     HWC
    fake_img = torch.rand(512, 512, 3)
    writer.add_image("fake_img", fake_img, 5, dataformats="HWC")

    writer.close()

# ----------------------------------- 4 make_grid -----------------------------------
flag = 0
# flag = 1
if flag:
    writer = SummaryWriter(comment='test_your_comment', filename_suffix="_test_your_filename_suffix")

    split_dir = os.path.join(enviroments.project_dir, "data", "rmb_split")
    train_dir = os.path.join(split_dir, "train")
    # train_dir = "path to your training data"
    # 先把宽高缩放到 [32， 64] 之间，然后使用 toTensor 把 Image 转化为 tensor，并把像素值缩放到 [0, 1] 之间
    transform_compose = transforms.Compose([transforms.Resize((32, 64)), transforms.ToTensor()])
    train_data = RMBDataset(data_dir=train_dir, transform=transform_compose)
    train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
    # 通过 next(iter()) 方法获取 data 和 label
    data_batch, label_batch = next(iter(train_loader))

    img_grid = vutils.make_grid(data_batch, nrow=4, normalize=True, scale_each=True)
    # img_grid = vutils.make_grid(data_batch, nrow=4, normalize=False, scale_each=False)
    writer.add_image("input img", img_grid, 0)

    writer.close()


# ----------------------------------- 5 add_graph -----------------------------------

# flag = 0
flag = 1
if flag:

    writer = SummaryWriter(comment='test_your_comment', filename_suffix="_test_your_filename_suffix")

    # 模型
    fake_img = torch.randn(3, 3, 32, 32)

    lenet = LeNet(classes=2)

    writer.add_graph(lenet, fake_img)

    writer.close()
    # print(summary(lenet, (3, 32, 32), device="cpu"))










