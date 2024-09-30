#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn import functional as f

"""
    构造V1版本的GoogleNet
                                           前一层
                   1x1 conv      1x1 conv             1x1 conv        3x3 max pooling
                      |             |                     |                  |
                                 3x3 conv             5x5 conv            1x1 conv 
                                         特征图拼接
"""


class InceptionV1(nn.Module):
    def __init__(self, in_channels, kernel_1, kernel_31, kernel_c3, kernel_51, kernel_c5, kernel_c51):
        """
        :param in_channels: 输入通道数
        :param kernel_1: 1*1卷积层的输出通道数
        :param kernel_31: 3*3卷积层之前的1*1卷积层的输出通道数
        :param kernel_51: 5*5卷积层之前的1*1卷积层的输出通道数
        :param kernel_c3: 3*3卷积层的输出通道数
        :param kernel_c5: 5*5卷积层的输出通道数
        :param kernel_c51: 3*3池化层后的卷积层的输出通道数
        """
        super(InceptionV1, self).__init__()
        # 线路1， 单个1*1卷积层
        self.p1_1 = nn.Conv2d(in_channels=in_channels, out_channels=kernel_1, kernel_size=1, padding=0, stride=1)

        # 线路2，一个1*1卷积层和一个3*3卷积层
        self.p2_1 = nn.Conv2d(in_channels=in_channels, out_channels=kernel_31, kernel_size=1, padding=0, stride=1)
        self.p2_2 = nn.Conv2d(in_channels=kernel_31, out_channels=kernel_c3, kernel_size=3, padding=1, stride=1)

        # 线路3，一个1*1卷积层和一个5*5卷积层
        self.p3_1 = nn.Conv2d(in_channels=in_channels, out_channels=kernel_51, kernel_size=1, padding=0, stride=1)
        self.p3_2 = nn.Conv2d(in_channels=kernel_51, out_channels=kernel_c5, kernel_size=5, padding=2, stride=1)

        # 线路4，一个3*3最大值池化层和一个1*1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
        self.p4_2 = nn.Conv2d(in_channels=in_channels, out_channels=kernel_c51, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        p1 = f.relu(self.p1_1(x))
        p2 = f.relu(self.p2_2(f.relu(self.p2_1(x))))
        p3 = f.relu(self.p3_2(f.relu(self.p3_1(x))))
        p4 = f.relu(self.p4_2(self.p4_1(x)))

        return torch.cat((p1, p2, p3, p4), dim=1)


class GoogleNet_V1(nn.Module):
    def __init__(self):
        super(GoogleNet_V1, self).__init__()
        self.net = nn.Sequential(
            # C1 7*7卷积层 + ReLU
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=3, stride=2),
            nn.LeakyReLU(),

            # P1 最大值池化层
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2),

            # C2 3*3卷积层 + ReLU
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
            nn.LeakyReLU(),

            # P2 最大值池化层
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2),

            # I3 Inception层
            InceptionV1(192, 64, 96, 128, 16, 32, 32),

            # I4 Inception层
            InceptionV1(256, 128, 128, 192, 32, 96, 64),

            # P4 最大值池化层
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2),

            # I5 Inception层
            InceptionV1(480, 192, 96, 208, 16, 48, 64),

            # I6 Inception层
            InceptionV1(512, 160, 112, 224, 24, 64, 64),

            # I7 Inception层
            InceptionV1(512, 128, 128, 256, 24, 64, 64),

            # I8 Inception层
            InceptionV1(512, 112, 144, 288, 32, 64, 64),

            # I9 Inception层
            InceptionV1(528, 256, 160, 320, 32, 128, 128),

            # P9 最大值池化层
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2),

            # I10 Inception层
            InceptionV1(832, 256, 160, 320, 32, 128, 128),

            # I11 Inception层
            InceptionV1(832, 384, 192, 384, 48, 128, 128),

            # P11 平均值池化层
            nn.AvgPool2d(kernel_size=7, padding=0, stride=1),
            nn.Flatten(),

            nn.Dropout(0.5),

            nn.Linear(1024, 45)

        )

    def forward(self, x):
        return self.net(x)

