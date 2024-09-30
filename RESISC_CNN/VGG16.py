#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

"""
    构造VGG16卷积神经网络
"""


class vgg_16(nn.Module):

    def __init__(self):
        super(vgg_16, self).__init__()

        # 定义网格结构
        self.net = nn.Sequential(
            # input 3 * 224 * 224
            # C1 卷积层 + ReLU
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # C2 卷积层 + ReLU
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # P2 池化层
            nn.MaxPool2d(kernel_size=2, stride=2),

            # C3 卷积层 + ReLU
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # C4 卷积层 + ReLU
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # P4 池化层
            nn.MaxPool2d(kernel_size=2, stride=2),

            # C5 卷积层 + ReLU
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # C6 卷积层 + ReLU
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # C7 卷积层 + ReLU
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # P7 池化层
            nn.MaxPool2d(kernel_size=2, stride=2),

            # C8 卷积层 + ReLU
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # C9 卷积层 + ReLU
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # C10 卷积层 + ReLU
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # P10 池化层
            nn.MaxPool2d(kernel_size=2, stride=2),

            # C11 卷积层 + ReLU
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # C12 卷积层 + ReLU
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # C13 卷积层 + ReLU
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # P13 池化层
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 铺平
            nn.Flatten(),

            # FC14 全连接层
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(0.8),

            # FC15 全连接层
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.8),

            # FC16 全连接层
            nn.Linear(4096, 45),
        )

    def forward(self, x):
        return self.net(x)