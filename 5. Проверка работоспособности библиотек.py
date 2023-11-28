# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 20:41:48 2023

@author: sviat
"""

import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Создайте небольшую сеть
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(3, 3)

    def forward(self, x):
        return self.fc1(x)

net = SimpleNet().to(device)
optimizer = optim.SGD(net.parameters(), lr=0.001)

# Создайте случайный тензор и небольшой цикл оптимизации
inputs = torch.randn(8, 3).to(device)
criterion = nn.MSELoss()

for epoch in range(10):
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, inputs)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")