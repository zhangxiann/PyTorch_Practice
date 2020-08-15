# -*- coding: utf-8 -*-
"""
使用多 GPU
"""
import os
import numpy as np
import torch
import torch.nn as nn

# ============================ 手动选择gpu
flag = 0
# flag = 1
if flag:

    gpu_list = [0]
    gpu_list_str = ','.join(map(str, gpu_list))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================ 依内存情况自动选择主gpu
flag = 0
# flag = 1
if flag:
    def get_gpu_memory():
        import platform
        if 'Windows' != platform.system():
            import os
            os.system('nvidia-smi -q -d Memory | grep -A4 GPU | grep Free > tmp.txt')
            memory_gpu = [int(x.split()[2]) for x in open('tmp.txt', 'r').readlines()]
            os.system('rm tmp.txt')
        else:
            memory_gpu = False
            print("显存计算功能暂不支持windows操作系统")
        return memory_gpu


    gpu_memory = get_gpu_memory()
    if not gpu_memory:
        print("\ngpu free memory: {}".format(gpu_memory))
        gpu_list = np.argsort(gpu_memory)[::-1]

        gpu_list_str = ','.join(map(str, gpu_list))
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FooNet(nn.Module):
    def __init__(self, neural_num, layers=3):
        super(FooNet, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(neural_num, neural_num, bias=False) for i in range(layers)])

    def forward(self, x):

        print("\nbatch size in forward: {}".format(x.size()[0]))

        for (i, linear) in enumerate(self.linears):
            x = linear(x)
            x = torch.relu(x)
        return x


if __name__ == "__main__":
    # 设置 2 个可见 GPU
    gpu_list = [0,1]
    gpu_list_str = ','.join(map(str, gpu_list))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 16

    # data
    inputs = torch.randn(batch_size, 3)
    labels = torch.randn(batch_size, 3)

    inputs, labels = inputs.to(device), labels.to(device)

    # model
    net = FooNet(neural_num=3, layers=3)
    net = nn.DataParallel(net)
    net.to(device)

    # training
    for epoch in range(1):

        outputs = net(inputs)

        print("model outputs.size: {}".format(outputs.size()))

    print("CUDA_VISIBLE_DEVICES :{}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print("device_count :{}".format(torch.cuda.device_count()))





















