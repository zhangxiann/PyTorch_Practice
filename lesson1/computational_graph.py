# -*- coding:utf-8 -*-

import torch
w = torch.tensor([1.], requires_grad=True)
x = torch.tensor([2.], requires_grad=True)
# y=(x+w)*(w+1)
a = torch.add(w, x)     # retain_grad()
b = torch.add(w, 1)
y = torch.mul(a, b)
# y 求导
y.backward()
# 打印 w 的梯度，就是 y 对 w 的导数
print(w.grad)

# 查看叶子结点
print("is_leaf:\n", w.is_leaf, x.is_leaf, a.is_leaf, b.is_leaf, y.is_leaf)

# 查看梯度
print("gradient:\n", w.grad, x.grad, a.grad, b.grad, y.grad)

# 查看 grad_fn
print("w.grad_fn = ", w.grad_fn)
print("x.grad_fn = ", x.grad_fn)
print("a.grad_fn = ", a.grad_fn)
print("b.grad_fn = ", b.grad_fn)
print("y.grad_fn = ", y.grad_fn)

