# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
from common_tools import set_seed

set_seed(1)  # 设置随机种子

# ----------------------------------- 1 tensor hook 1 -----------------------------------
flag = 0
# flag = 1
if flag:

    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)
    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    # 保存梯度的 list
    a_grad = list()

    # 定义 hook 函数，把梯度添加到 list 中
    def grad_hook(grad):
        a_grad.append(grad)

    # 一个张量注册 hook 函数
    handle = a.register_hook(grad_hook)

    y.backward()

    # 查看梯度
    print("gradient:", w.grad, x.grad, a.grad, b.grad, y.grad)
    # 查看在 hook 函数里 list 记录的梯度
    print("a_grad[0]: ", a_grad[0])
    handle.remove()


# ----------------------------------- 2 tensor hook 2 -----------------------------------
flag = 0
# flag = 1
if flag:

    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)
    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    a_grad = list()

    def grad_hook(grad):
        grad *= 2
        return grad*3

    handle = w.register_hook(grad_hook)

    y.backward()

    # 查看梯度
    print("w.grad: ", w.grad)
    handle.remove()


# ----------------------------------- 3 Module.register_forward_hook and pre hook -----------------------------------
# flag = 0
flag = 1
if flag:

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 2, 3)
            self.pool1 = nn.MaxPool2d(2, 2)

        def forward(self, x):
            x = self.conv1(x)
            x = self.pool1(x)
            return x

    def forward_hook(module, data_input, data_output):
        fmap_block.append(data_output)
        input_block.append(data_input)

    def forward_pre_hook(module, data_input):
        print("forward_pre_hook input:{}".format(data_input))

    def backward_hook(module, grad_input, grad_output):
        print("backward hook input:{}".format(grad_input))
        print("backward hook output:{}".format(grad_output))

    # 初始化网络
    net = Net()
    net.conv1.weight[0].detach().fill_(1)
    net.conv1.weight[1].detach().fill_(2)
    net.conv1.bias.data.detach().zero_()

    # 注册hook
    fmap_block = list()
    input_block = list()
    net.conv1.register_forward_hook(forward_hook)
    net.conv1.register_forward_pre_hook(forward_pre_hook)
    net.conv1.register_backward_hook(backward_hook)

    # inference
    fake_img = torch.ones((1, 1, 4, 4))   # batch size * channel * H * W
    output = net(fake_img)

    loss_fnc = nn.L1Loss()
    target = torch.randn_like(output)
    loss = loss_fnc(target, output)
    loss.backward()

    # 观察
    # print("output shape: {}\noutput value: {}\n".format(output.shape, output))
    # print("feature maps shape: {}\noutput value: {}\n".format(fmap_block[0].shape, fmap_block[0]))
    # print("input shape: {}\ninput value: {}".format(input_block[0][0].shape, input_block[0]))






















