# -*- coding:utf-8 -*-
"""
积核和特征图的可视化
"""
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from common_tools import set_seed
import torchvision.models as models

set_seed(1)  # 设置随机种子


# ----------------------------------- kernel visualization -----------------------------------
flag = 0
# flag = 1
if flag:
    writer = SummaryWriter(comment='test_your_comment', filename_suffix="_test_your_filename_suffix")

    alexnet = models.alexnet(pretrained=True)

    # 当前遍历到第几层网络的卷积核了
    kernel_num = -1
    # 最多显示两层网络的卷积核:第 0 层和第 1 层
    vis_max = 1

    # 获取网络的每一层
    for sub_module in alexnet.modules():
        # 判断这一层是否为 2 维卷积层
        if isinstance(sub_module, nn.Conv2d):
            kernel_num += 1
            # 如果当前层大于1，则停止记录权值
            if kernel_num > vis_max:
                break
            # 获取这一层的权值
            kernels = sub_module.weight
            # 权值的形状是 [c_out, c_int, k_w, k_h]
            c_out, c_int, k_w, k_h = tuple(kernels.shape)

            # 根据输出的每个维度进行可视化
            for o_idx in range(c_out):
                # 取出的数据形状是 (c_int, k_w, k_h)，对应 BHW; 需要扩展为 (c_int, 1, k_w, k_h)，对应 BCHW
                kernel_idx = kernels[o_idx, :, :, :].unsqueeze(1)   # make_grid需要 BCHW，这里拓展C维度
                # 注意 nrow 设置为 c_int，所以行数为 1。在 for 循环中每 添加一个，就会多一个 global_step
                kernel_grid = vutils.make_grid(kernel_idx, normalize=True, scale_each=True, nrow=c_int)
                writer.add_image('{}_Convlayer_split_in_channel'.format(kernel_num), kernel_grid, global_step=o_idx)
            # 因为 channe 为 3 时才能进行可视化，所以这里 reshape
            kernel_all = kernels.view(-1, 3, k_h, k_w)  #b, 3, h, w
            kernel_grid = vutils.make_grid(kernel_all, normalize=True, scale_each=True, nrow=8)  # c, h, w
            writer.add_image('{}_all'.format(kernel_num), kernel_grid, global_step=kernel_num+1)

            print("{}_convlayer shape:{}".format(kernel_num, tuple(kernels.shape)))

    writer.close()

# ----------------------------------- feature map visualization -----------------------------------
flag = 0
# flag = 1
if flag:
    writer = SummaryWriter(comment='test_your_comment', filename_suffix="_test_your_filename_suffix")

    # 数据
    path_img = "imgs/lena.png"     # your path to image
    normMean = [0.49139968, 0.48215827, 0.44653124]
    normStd = [0.24703233, 0.24348505, 0.26158768]

    norm_transform = transforms.Normalize(normMean, normStd)
    img_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        norm_transform
    ])

    img_pil = Image.open(path_img).convert('RGB')
    if img_transforms is not None:
        img_tensor = img_transforms(img_pil)
    img_tensor.unsqueeze_(0)    # chw --> bchw

    # 模型
    alexnet = models.alexnet(pretrained=True)

    # forward
    # 由于在定义模型时，网络层通过nn.Sequential() 堆叠，保存在 features 变量中。因此通过 features 获取第一个卷积层
    convlayer1 = alexnet.features[0]
    # 把图片输入第一个卷积层
    fmap_1 = convlayer1(img_tensor)

    # 预处理
    fmap_1.transpose_(0, 1)  # bchw=(1, 64, 55, 55) --> (64, 1, 55, 55)
    fmap_1_grid = vutils.make_grid(fmap_1, normalize=True, scale_each=True, nrow=8)

    writer.add_image('feature map in conv1', fmap_1_grid, global_step=322)
    writer.close()
    
























