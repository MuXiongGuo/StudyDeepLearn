import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
import random


def synthetic_data(w, b, num_examples): #@save
    """⽣成 y = Xw + b + 噪声。"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i +
        batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


def linreg(X, w, b): #@save
    """线性回归模型。"""
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y): #@save
    """均⽅损失。"""
    return (y_hat - y.reshape(y_hat.shape))**2 / 2


def sgd(params, lr, batch_size): #@save
    """⼩批量随机梯度下降。"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

# 目标
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 10000)
batch_size = 10

# 一些准备
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

# start
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y) # `X`和`y`的⼩批量损失
        # 因为`l`形状是(`batch_size`, 1)，⽽不是⼀个标量。`l`中的所有元素被加到⼀起，
        # 并以此计算关于[`w`, `b`]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size) # 使⽤参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')