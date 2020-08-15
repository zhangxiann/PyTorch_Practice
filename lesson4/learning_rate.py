# -*- coding:utf-8 -*-
import torch
import numpy as np
import matplotlib.pyplot as plt
torch.manual_seed(1)


def func(x_t):
    """
    y = (2x)^2 = 4*x^2      dy/dx = 8x
    """
    return torch.pow(2*x_t, 2)


# init
x = torch.tensor([2.], requires_grad=True)


# ------------------------------ plot data ------------------------------
flag = 0
# flag = 1
if flag:

    x_t = torch.linspace(-3, 3, 100)
    y = func(x_t)
    plt.plot(x_t.numpy(), y.numpy(), label="y = 4*x^2")
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


# ------------------------------ gradient descent ------------------------------
flag = 0
# flag = 1
if flag:
    iter_rec, loss_rec, x_rec = list(), list(), list()

    lr = 1    # /1. /.5 /.2 /.1 /.125
    max_iteration = 20   # /1. 4     /.5 4   /.2 20 200

    for i in range(max_iteration):

        y = func(x)
        y.backward()

        print("Iter:{}, X:{:8}, X.grad:{:8}, loss:{:10}".format(
            i, x.detach().numpy()[0], x.grad.detach().numpy()[0], y.item()))

        x_rec.append(x.item())

        x.data.sub_(lr * x.grad)    # x -= x.grad  数学表达式意义:  x = x - x.grad    # 0.5 0.2 0.1 0.125
        x.grad.zero_()

        iter_rec.append(i)
        loss_rec.append(y)

    plt.subplot(121).plot(iter_rec, loss_rec, '-ro')
    plt.xlabel("Iteration")
    plt.ylabel("Loss value")

    x_t = torch.linspace(-3, 3, 100)
    y = func(x_t)
    plt.subplot(122).plot(x_t.numpy(), y.numpy(), label="y = 4*x^2")
    plt.grid()
    y_rec = [func(torch.tensor(i)).item() for i in x_rec]
    plt.subplot(122).plot(x_rec, y_rec, '-ro')
    plt.legend()
    plt.show()

# ------------------------------ multi learning rate ------------------------------

# flag = 0
flag = 1
if flag:
    iteration = 100
    num_lr = 10
    lr_min, lr_max = 0.01, 0.2  # .5 .3 .2

    lr_list = np.linspace(lr_min, lr_max, num=num_lr).tolist()
    loss_rec = [[] for l in range(len(lr_list))]
    iter_rec = list()

    for i, lr in enumerate(lr_list):
        x = torch.tensor([2.], requires_grad=True)
        for iter in range(iteration):

            y = func(x)
            y.backward()
            x.data.sub_(lr * x.grad)  # x.data -= x.grad
            x.grad.zero_()

            loss_rec[i].append(y.item())

    for i, loss_r in enumerate(loss_rec):
        plt.plot(range(len(loss_r)), loss_r, label="LR: {}".format(lr_list[i]))
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Loss value')
    plt.show()









