#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

"""
    构造AlexNet卷积神经网络
"""


# 网格基本信息
class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet, self).__init__()

        # 创建网络结构
        self.net = nn.Sequential(
            # 输入数据为1 * 224 * 224的灰度图像
            # C1 卷积层 + ReLU
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2),
            nn.LeakyReLU(),

            # P1 最大值池化层
            nn.MaxPool2d(kernel_size=3, stride=2),

            # C2 卷积层 + ReLU
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(),

            # P2 最大值池化层
            nn.MaxPool2d(kernel_size=3, stride=2),

            # C3 卷积层 + ReLU
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),

            # C4 卷积层 + ReLU
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),

            # C5 卷积层 + ReLU
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),

            # P5 最大值池化层
            nn.MaxPool2d(kernel_size=3, stride=2),

            # 铺平
            nn.Flatten(),

            # FC6 全连接层
            nn.Linear(256*6*6, 4096),
            nn.LeakyReLU(),
            nn.Dropout(0.75),

            # FC7 全连接层
            nn.Linear(4096, 4096),
            nn.LeakyReLU(),
            nn.Dropout(0.75),

            # 输出层
            nn.Linear(4096, 45)

        )

    def forward(self, x):
        return self.net(x)
