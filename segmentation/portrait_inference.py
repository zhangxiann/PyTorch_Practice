# -*- coding: utf-8 -*-

import os
import time
import random
import torch.nn as nn
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import torch.optim as optim
import torchvision.models as models
from segmentation.tools.common_tools import set_seed
from segmentation.tools.my_dataset import PortraitDataset
from segmentation.tools.unet import UNet

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_seed()  # 设置随机种子


def compute_dice(y_pred, y_true):
    """
    :param y_pred: 4-d tensor, value = [0,1]
    :param y_true: 4-d tensor, value = [0,1]
    :return:
    """
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    y_pred, y_true = np.round(y_pred).astype(int), np.round(y_true).astype(int)
    return np.sum(y_pred[y_true == 1]) * 2.0 / (np.sum(y_pred) + np.sum(y_true))


def get_img_name(img_dir, format="jpg"):
    """
    获取文件夹下format格式的文件名
    :param img_dir: str
    :param format: str
    :return: list
    """
    file_names = os.listdir(img_dir)
    img_names = list(filter(lambda x: x.endswith(format), file_names))
    img_names = list(filter(lambda x: not x.endswith("matte.png"), img_names))

    if len(img_names) < 1:
        raise ValueError("{}下找不到{}格式数据".format(img_dir, format))
    return img_names


def get_model(m_path):

    unet = UNet(in_channels=3, out_channels=1, init_features=32)
    checkpoint = torch.load(m_path, map_location="cpu")

    # remove module.
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint['model_state_dict'].items():
        namekey = k[7:] if k.startswith('module.') else k
        new_state_dict[namekey] = v

    unet.load_state_dict(new_state_dict)

    return unet


if __name__ == "__main__":

    img_dir = os.path.join(BASE_DIR, "..", "data", "PortraitDataset", "valid")
    model_path = "checkpoint_399_epoch.pkl"
    time_total = 0
    num_infer = 5
    mask_thres = .5

    # 1. data
    img_names = get_img_name(img_dir, format="png")
    random.shuffle(img_names)
    num_img = len(img_names)

    # 2. model
    unet = get_model(model_path)
    unet.to(device)
    unet.eval()

    for idx, img_name in enumerate(img_names):
        if idx > num_infer:
            break

        path_img = os.path.join(img_dir, img_name)
        # path_img = "C:\\Users\\Administrator\\Desktop\\Andrew-wu.png"
        #
        # step 1/4 : path --> img_chw
        img_hwc = Image.open(path_img).convert('RGB')
        img_hwc = img_hwc.resize((224, 224))
        img_arr = np.array(img_hwc)
        img_chw = img_arr.transpose((2, 0, 1))

        # step 2/4 : img --> tensor
        img_tensor = torch.tensor(img_chw).to(torch.float)
        img_tensor.unsqueeze_(0)
        img_tensor = img_tensor.to(device)

        # step 3/4 : tensor --> features
        time_tic = time.time()
        outputs = unet(img_tensor)
        time_toc = time.time()

        # step 4/4 : visualization
        pred = outputs.ge(mask_thres)
        mask_pred = outputs.ge(0.5).cpu().data.numpy().astype("uint8")

        img_hwc = img_tensor.cpu().data.numpy()[0, :, :, :].transpose((1, 2, 0)).astype("uint8")
        plt.subplot(121).imshow(img_hwc)
        mask_pred_gray = mask_pred.squeeze() * 255
        plt.subplot(122).imshow(mask_pred_gray, cmap="gray")
        plt.show()
        # plt.pause(0.5)
        plt.close()

        time_s = time_toc - time_tic
        time_total += time_s

        print('{:d}/{:d}: {} {:.3f}s '.format(idx + 1, num_img, img_name, time_s))













