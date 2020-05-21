# -*- coding: utf-8 -*-

import torch
torch.manual_seed(10)


# ====================================== retain_graph ==============================================
# flag = True
flag = False
if flag:
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)
    # y=(x+w)*(w+1)
    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    # 第一次求导，设置 retain_graph=True，保留计算图
    y.backward(retain_graph=True)
    print(w.grad)
    # 第二次求导成功
    y.backward()

# ====================================== grad_tensors ==============================================
# flag = True
flag = False
if flag:
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)

    y0 = torch.mul(a, b)    # y0 = (x+w) * (w+1)
    y1 = torch.add(a, b)    # y1 = (x+w) + (w+1)    dy1/dw = 2

    # 把两个 loss 拼接都到一起
    loss = torch.cat([y0, y1], dim=0)       # [y0, y1]
    # 设置两个 loss 的权重: y0 的权重是 1，y1 的权重是 2
    grad_tensors = torch.tensor([1., 2.])

    loss.backward(gradient=grad_tensors)    # gradient 传入 torch.autograd.backward()中的grad_tensors
    # 最终的 w 的导数由两部分组成。∂y0/∂w * 1 + ∂y1/∂w * 2
    print(w.grad)


# ====================================== autograd.gard ==============================================
# flag = True
flag = False
if flag:

    x = torch.tensor([3.], requires_grad=True)
    y = torch.pow(x, 2)     # y = x**2
    # 如果需要求 2 阶导，需要设置 create_graph=True，让一阶导数 grad_1 也拥有计算图
    grad_1 = torch.autograd.grad(y, x, create_graph=True)   # grad_1 = dy/dx = 2x = 2 * 3 = 6
    print(grad_1)
    # 这里求 2 阶导
    grad_2 = torch.autograd.grad(grad_1[0], x)              # grad_2 = d(dy/dx)/dx = d(2x)/dx = 2
    print(grad_2)


# ====================================== tips: 1 ==============================================
# flag = True
flag = False
if flag:

    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)
    # 进行 4 次反向传播求导，每次最后都没有清零
    for i in range(4):
        a = torch.add(w, x)
        b = torch.add(w, 1)
        y = torch.mul(a, b)
        y.backward()
        print(w.grad)
        # 每次都把梯度清零
        # w.grad.zero_()


# ====================================== tips: 2 ==============================================
# flag = True
flag = False
if flag:

    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)
    # y = (x + w) * (w + 1)
    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    print(a.requires_grad, b.requires_grad, y.requires_grad)


# ====================================== tips: 3 ==============================================
flag = True
# flag = False
if flag:


    print("非 inplace 操作")
    a = torch.ones((1, ))
    print(id(a), a)
    # 非 inplace 操作，内存地址不一样
    a = a + torch.ones((1, ))
    print(id(a), a)

    print("inplace 操作")
    a = torch.ones((1, ))
    print(id(a), a)
    # inplace 操作，内存地址一样
    a += torch.ones((1, ))
    print(id(a), a)


# flag = True
flag = False
if flag:

    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)
    # y = (x + w) * (w + 1)
    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)
    # 在反向传播之前 inplace 改变了 w 的值，再执行 backward() 会报错
    w.add_(1)
    y.backward()





