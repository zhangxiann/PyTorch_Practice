# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
from torchvision import transforms
from matplotlib import pyplot as plt
from PIL import Image
from common_tools import transform_invert, set_seed

set_seed(1)  # 设置随机种子

# ================================= load img ==================================
path_img = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imgs/lena.png")
img = Image.open(path_img).convert('RGB')  # 0~255

# convert to tensor
img_transform = transforms.Compose([transforms.ToTensor()])
img_tensor = img_transform(img)
img_tensor.unsqueeze_(dim=0)    # C*H*W to B*C*H*W

# ================================= create convolution layer ==================================

# ================ maxpool
# flag = 1
flag = 0
if flag:
    maxpool_layer = nn.MaxPool2d((2, 2), stride=(2, 2))   # input:(i, o, size) weights:(o, i , h, w)
    img_pool = maxpool_layer(img_tensor)

# ================ avgpool
# flag = 1
flag = 0
if flag:
    avgpoollayer = nn.AvgPool2d((2, 2), stride=(2, 2))   # input:(i, o, size) weights:(o, i , h, w)
    img_pool = avgpoollayer(img_tensor)

# ================ avgpool divisor_override
# flag = 1
flag = 0
if flag:
    img_tensor = torch.ones((1, 1, 4, 4))
    avgpool_layer = nn.AvgPool2d((2, 2), stride=(2, 2), divisor_override=3)
    img_pool = avgpool_layer(img_tensor)

    print("raw_img:\n{}\npooling_img:\n{}".format(img_tensor, img_pool))


# ================ max unpool
# flag = 1
flag = 0
if flag:
    # pooling
    img_tensor = torch.randint(high=5, size=(1, 1, 4, 4), dtype=torch.float)
    maxpool_layer = nn.MaxPool2d((2, 2), stride=(2, 2), return_indices=True)
    img_pool, indices = maxpool_layer(img_tensor)

    # unpooling
    img_reconstruct = torch.randn_like(img_pool, dtype=torch.float)
    maxunpool_layer = nn.MaxUnpool2d((2, 2), stride=(2, 2))
    img_unpool = maxunpool_layer(img_reconstruct, indices)

    print("raw_img:\n{}\nimg_pool:\n{}".format(img_tensor, img_pool))
    print("img_reconstruct:\n{}\nimg_unpool:\n{}".format(img_reconstruct, img_unpool))


# ================ linear
flag = 1
# flag = 0
if flag:
    inputs = torch.tensor([[1., 2, 3]])
    linear_layer = nn.Linear(3, 4)
    linear_layer.weight.data = torch.tensor([[1., 1., 1.],
                                             [2., 2., 2.],
                                             [3., 3., 3.],
                                             [4., 4., 4.]])

    linear_layer.bias.data.fill_(0.5)
    output = linear_layer(inputs)
    print(inputs, inputs.shape)
    print(linear_layer.weight.data, linear_layer.weight.data.shape)
    print(output, output.shape)

# ================================= visualization ==================================
# print("池化前尺寸:{}\n池化后尺寸:{}".format(img_tensor.shape, img_pool.shape))
# img_pool = transform_invert(img_pool[0, 0:3, ...], img_transform)
# img_raw = transform_invert(img_tensor.squeeze(), img_transform)
# plt.subplot(122).imshow(img_pool)
# plt.subplot(121).imshow(img_raw)
# plt.show()










