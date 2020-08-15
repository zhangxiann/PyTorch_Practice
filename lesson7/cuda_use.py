# -*- coding: utf-8 -*-
"""
数据迁移至cuda的方法
"""
import torch
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ========================== tensor to cuda
flag = 0
# flag = 1
if flag:
    x_cpu = torch.ones((3, 3))
    print("x_cpu:\ndevice: {} is_cuda: {} id: {}".format(x_cpu.device, x_cpu.is_cuda, id(x_cpu)))

    x_gpu = x_cpu.to(device)
    print("x_gpu:\ndevice: {} is_cuda: {} id: {}".format(x_gpu.device, x_gpu.is_cuda, id(x_gpu)))

# 弃用
# x_gpu = x_cpu.cuda()

# ========================== module to cuda
flag = 0
# flag = 1
if flag:
    net = nn.Sequential(nn.Linear(3, 3))

    print("\nid:{} is_cuda: {}".format(id(net), next(net.parameters()).is_cuda))

    net.to(device)
    print("\nid:{} is_cuda: {}".format(id(net), next(net.parameters()).is_cuda))


# ========================== forward in cuda
# flag = 0
flag = 1
if flag:
    output = net(x_gpu)
    print("output is_cuda: {}".format(output.is_cuda))

    # output = net(x_cpu)









# ========================== 查看当前gpu 序号，尝试修改可见gpu，以及主gpu
flag = 0
# flag = 1
if flag:
    current_device = torch.cuda.current_device()
    print("current_device: ", current_device)

    torch.cuda.set_device(0)
    current_device = torch.cuda.current_device()
    print("current_device: ", current_device)


    #
    cap = torch.cuda.get_device_capability(device=None)
    print(cap)
    #
    name = torch.cuda.get_device_name()
    print(name)

    is_available = torch.cuda.is_available()
    print(is_available)



    # ===================== seed
    seed = 2
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    current_seed = torch.cuda.initial_seed()
    print(current_seed)


    s = torch.cuda.seed()
    s_all = torch.cuda.seed_all()




