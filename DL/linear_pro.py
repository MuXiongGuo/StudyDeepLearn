#  使用框架来实现线性训练
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn


def load_array(data_arrays, batch_size, is_train=True):  # @save
    """构造⼀个PyTorch数据迭代器。"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


# generated data
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

# iter
batch_size = 10
data_iter = load_array((features, labels), batch_size)

# create model
net = nn.Sequential(nn.Linear(2, 1))
# initialize model
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# define loss function
loss = nn.MSELoss()  # L2 norm by default

# define optimization algorithm
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# start training
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    with torch.no_grad():
        l = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {l:f}')
