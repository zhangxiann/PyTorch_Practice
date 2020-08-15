# -*- coding: utf-8 -*-

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import torch
import torch.optim as optim
from common_tools import set_seed

set_seed(1)  # 设置随机种子

weight = torch.randn((2, 2), requires_grad=True)
weight.grad = torch.ones((2, 2))

optimizer = optim.SGD([weight], lr=1)

# ----------------------------------- step -----------------------------------
flag = 0
# flag = 1
if flag:
    print("weight before step:{}".format(weight.data))
    optimizer.step()        # 修改lr=1 0.1观察结果
    print("weight after step:{}".format(weight.data))


# ----------------------------------- zero_grad -----------------------------------
flag = 0
# flag = 1
if flag:

    print("weight before step:{}".format(weight.data))
    optimizer.step()        # 修改lr=1 0.1观察结果
    print("weight after step:{}".format(weight.data))

    print("weight in optimizer:{}\nweight in weight:{}\n".format(id(optimizer.param_groups[0]['params'][0]), id(weight)))

    print("weight.grad is {}\n".format(weight.grad))
    optimizer.zero_grad()
    print("after optimizer.zero_grad(), weight.grad is\n{}".format(weight.grad))


# ----------------------------------- add_param_group -----------------------------------
flag = 0
# flag = 1
if flag:
    print("optimizer.param_groups is\n{}".format(optimizer.param_groups))

    w2 = torch.randn((3, 3), requires_grad=True)

    optimizer.add_param_group({"params": w2, 'lr': 0.0001})

    print("optimizer.param_groups is\n{}".format(optimizer.param_groups))

# ----------------------------------- state_dict -----------------------------------
flag = 0
# flag = 1
if flag:

    optimizer = optim.SGD([weight], lr=0.1, momentum=0.9)
    opt_state_dict = optimizer.state_dict()

    print("state_dict before step:\n", opt_state_dict)

    for i in range(10):
        optimizer.step()

    print("state_dict after step:\n", optimizer.state_dict())

    torch.save(optimizer.state_dict(), os.path.join(BASE_DIR, "optimizer_state_dict.pkl"))

# -----------------------------------load state_dict -----------------------------------
# flag = 0
flag = 1
if flag:

    optimizer = optim.SGD([weight], lr=0.1, momentum=0.9)
    state_dict = torch.load(os.path.join(BASE_DIR, "optimizer_state_dict.pkl"))

    print("state_dict before load state:\n", optimizer.state_dict())
    optimizer.load_state_dict(state_dict)
    print("state_dict after load state:\n", optimizer.state_dict())












