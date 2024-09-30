#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn import functional as f
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

from CNN_with_Transformer import SENet

"""
    构造V2版本的GoogleNet，相比于V1版本，V2版本针对如下情况进行了改动：
    1)、引入了BN层，对所有隐藏层的输入项进行归一化处理
    2)、利用两个串联的3*3卷积层代替Inception块中所有的5*5卷积层，这一改动可以进一步改进为：
        利用串联的n*1和1*n卷积层代替Inception块中所有的n*n卷积层
                                          前一层                                                                     
                   1x1 conv      1x1 conv               pool              1x1 conv 
                      |             |                     |                  |
                   3x3 conv      3x3 conv             1x1 conv               |
                      |             |                     |                  | 
                   3x3 conv         |                     |                  |                                                                  
                                         特征图拼接
"""


class InceptionV2(nn.Module):
    # V2版本的Inception层
    def __init__(self, _in_channels, pth1, pth2, pth3, pth4):
        """
        :param _in_channels: 输入的通道数
        :param pth1: 路径1，包括一个1*1卷积层的输出通道数
        :param pth2: 路径2，包括一个1*1卷积层的输出通道数，一个3*3卷积层的输出通道数
        :param pth3: 路径3，包括一个1*1卷积层的输出通道数，两个3*3卷积层的输出通道数
        :param pth4: 路径4，一个3*3池化层，一个1*1卷积层的输出通道数
        """
        super(InceptionV2, self).__init__()
        # 路径1，单个1*1卷积层
        self.p1_1 = nn.Conv2d(in_channels=_in_channels, out_channels=pth1, kernel_size=1, padding=0, stride=1)
        self.Bth1 = nn.BatchNorm2d(pth1)

        # 路径2，一个1*1卷积层，一个3*3卷积层
        self.p2_1 = nn.Conv2d(in_channels=_in_channels, out_channels=pth2[0], kernel_size=1, padding=0, stride=1)
        self.Bth2_1 = nn.BatchNorm2d(pth2[0])
        self.p2_2 = nn.Conv2d(in_channels=pth2[0], out_channels=pth2[1], kernel_size=3, padding=1, stride=1)
        self.Bth2_2 = nn.BatchNorm2d(pth2[1])

        # 路径3，一个1*1卷积层，两个3*3卷积层
        self.p3_1 = nn.Conv2d(in_channels=_in_channels, out_channels=pth3[0], kernel_size=1, padding=0, stride=1)
        self.Bth3_1 = nn.BatchNorm2d(pth3[0])
        self.p3_2 = nn.Conv2d(in_channels=pth3[0], out_channels=pth3[1], kernel_size=3, padding=1, stride=1)
        self.Bth3_2 = nn.BatchNorm2d(pth3[1])
        self.p3_3 = nn.Conv2d(in_channels=pth3[1], out_channels=pth3[2], kernel_size=3, padding=1, stride=1)
        self.Bth3_3 = nn.BatchNorm2d(pth3[2])

        # 路径4，一个3*3池化层，一个1*1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
        self.p4_2 = nn.Conv2d(in_channels=_in_channels, out_channels=pth4, kernel_size=1, padding=0, stride=1)
        self.Bth4 = nn.BatchNorm2d(pth4)

        # 添加一个SE层
        channels = pth1 + pth2[1] + pth3[2] + pth4
        self.se = SENet.SEBlock(channel=channels, reduction=12)

    def forward(self, x):
        p1 = f.relu(self.Bth1(self.p1_1(x)))
        p2 = f.relu(self.Bth2_2(self.p2_2(f.relu(self.Bth2_1(self.p2_1(x))))))
        p3 = f.relu(self.Bth3_3(self.p3_3(f.relu(self.Bth3_2(self.p3_2(f.relu(self.Bth3_1(self.p3_1(x)))))))))
        p4 = f.relu(self.Bth4(self.p4_2(self.p4_1(x))))

        return self.se(torch.cat((p1, p2, p3, p4), dim=1))


class GoogleNet_V2(nn.Module):
    # 构建GoogleNet网络的V2版本
    def __init__(self):
        super(GoogleNet_V2, self).__init__()
        self.net = nn.Sequential(
            # C1 7*7卷积层 + ReLU
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            # P1 最大值池化层
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2),

            # C2 3*3卷积层 + ReLU
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(),

            # P2 最大值池化层
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2),

            # I3 Inception层
            InceptionV2(192, 64, (96, 128), (16, 32, 32), 32),

            # I4 Inception层
            InceptionV2(256, 128, (128, 192), (32, 96, 96), 64),

            # P4 最大值池化层
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2),

            # I5 Inception层
            InceptionV2(480, 192, (96, 208), (16, 48, 48), 64),

            # I6 Inception层
            InceptionV2(512, 160, (112, 224), (24, 64, 64), 64),

            # I7 Inception层
            InceptionV2(512, 128, (128, 256), (24, 64, 64), 64),

            # I8 Inception层
            InceptionV2(512, 112, (144, 288), (32, 64, 64), 64),

            # I9 Inception层
            InceptionV2(528, 256, (160, 320), (32, 128, 128), 128),

            # P9 最大值池化层
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2),

            # I10 Inception层
            InceptionV2(832, 256, (160, 320), (32, 128, 128), 128),

            # I11 Inception层
            InceptionV2(832, 384, (192, 384), (48, 128, 128), 128),

            # P11 平均值池化层
            nn.AvgPool2d(kernel_size=7, padding=0, stride=1),

            # 展平
            nn.Flatten(),

            # DropOut正则化
            nn.Dropout(0.8),

            nn.Linear(1024, 45)

        )

    def forward(self, x):
        return self.net(x)


# 制作数据集
def Get_dataset():
    # 创建数据集

    # 数据集转换参数
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224, antialias=None),
        transforms.Normalize(0.1307, 0.3801)
    ])

    # 制作数据集

    train_Data = datasets.FashionMNIST(
        root='E:\\gzr\\Simple-CNN',
        train=True,
        download=True,
        transform=transform
    )

    test_Data = datasets.FashionMNIST(
        root='E:\\gzr\\Simple-CNN',
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_Data, shuffle=True, batch_size=64)
    test_loader = DataLoader(test_Data, shuffle=True, batch_size=64)

    return train_Data, test_Data, train_loader, test_loader
