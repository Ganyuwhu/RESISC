#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn import functional as f
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

"""
    构造V3版本的GoogleNet，相比于V2版本，V3版本的GoogleNet进行了如下改动：
        进行卷积分解，将n*n的卷积核分解成一个1*n的卷积核与一个n*1的卷积核的串联，
    这样加深了神经网络的层数，又减少了需要进行运算的参数数目
"""


class InceptionV3(nn.Module):
    # V3版本的Inception层
    def __init__(self, _in_channels, pth1, pth2, pth3, pth4):
        """
        :param _in_channels: 输入通道数
        :param pth1: 路径1上各卷积核的输出通道数
        :param pth2: 路径2上各卷积核的输出通道数
        :param pth3: 路径3上各卷积核的输出通道数
        :param pth4: 路径4上各卷积核的输出通道数
        """
        super(InceptionV3, self).__init__()
        # 路径1 单个1*1卷积层
        self.p1 = nn.Conv2d(in_channels=_in_channels, out_channels=pth1, kernel_size=1, padding=0, stride=1)
        self.Bth1 = nn.BatchNorm2d(pth1)

        # 路径2 一个1*1卷积层和一个被拆解的3*3卷积层
        self.p2_1 = nn.Conv2d(in_channels=_in_channels, out_channels=pth2[0], kernel_size=1, padding=0, stride=1)
        self.Bth2_1 = nn.BatchNorm2d(pth2[0])
        self.p2_2 = nn.Conv2d(in_channels=pth2[0], out_channels=pth2[1], kernel_size=(1, 3), padding=(0, 1), stride=1)
        self.Bth2_2 = nn.BatchNorm2d(pth2[1])
        self.p2_3 = nn.Conv2d(in_channels=pth2[1], out_channels=pth2[2], kernel_size=(3, 1), padding=(1, 0), stride=1)
        self.Bth2_3 = nn.BatchNorm2d(pth2[2])

        # 路径3 一个1*1卷积层和两个被拆解的3*3卷积层
        self.p3_1 = nn.Conv2d(in_channels=_in_channels, out_channels=pth3[0], kernel_size=1, padding=0, stride=1)
        self.Bth3_1 = nn.BatchNorm2d(pth3[0])
        self.p3_2 = nn.Conv2d(in_channels=pth3[0], out_channels=pth3[1], kernel_size=(1, 3), padding=(0, 1), stride=1)
        self.Bth3_2 = nn.BatchNorm2d(pth3[1])
        self.p3_3 = nn.Conv2d(in_channels=pth3[1], out_channels=pth3[2], kernel_size=(3, 1), padding=(1, 0), stride=1)
        self.Bth3_3 = nn.BatchNorm2d(pth3[2])
        self.p3_4 = nn.Conv2d(in_channels=pth3[2], out_channels=pth3[3], kernel_size=(1, 3), padding=(0, 1), stride=1)
        self.Bth3_4 = nn.BatchNorm2d(pth3[3])
        self.p3_5 = nn.Conv2d(in_channels=pth3[3], out_channels=pth3[4], kernel_size=(3, 1), padding=(1, 0), stride=1)
        self.Bth3_5 = nn.BatchNorm2d(pth3[4])

        # 路径4 一个3*3池化层，一个1*1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
        self.p4_2 = nn.Conv2d(in_channels=_in_channels, out_channels=pth4, kernel_size=1, padding=0, stride=1)
        self.Bth4 = nn.BatchNorm2d(pth4)

    def forward(self, x):
        p1 = f.relu(self.Bth1(self.p1(x)))
        p2 = f.relu(self.Bth2_3(self.p2_3(f.relu(self.Bth2_2(self.p2_2(f.relu(self.Bth2_1(self.p2_1(x)))))))))
        p3 = f.relu(self.Bth3_5(self.p3_5(f.relu(self.Bth3_4(self.p3_4(f.relu(self.Bth3_3(self.p3_3(
            f.relu(self.Bth3_2(self.p3_2(f.relu(self.Bth3_1(self.p3_1(x))))))
        )))))))))
        p4 = f.relu(self.Bth4(self.p4_2(self.p4_1(x))))

        return torch.cat((p1, p2, p3, p4), dim=1)


class GoogleNet_V3(nn.Module):
    # 构建GoogleNet网络的V3版本
    def __init__(self):
        super(GoogleNet_V3, self).__init__()
        self.net = nn.Sequential(
            # C1.1、C1.2 拆分的7*7卷积层 + ReLU
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(1, 7), padding=(0, 3), stride=(1, 2)),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 1), padding=(3, 0), stride=(2, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            # P1 最大值池化层
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2),

            # C2.1、C2.2 拆分的3*3卷积层 + ReLU
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3), padding=(0, 1), stride=(1, 1)),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=(3, 1), padding=(1, 0), stride=(1, 1)),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(),

            # P2 最大值池化层
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2),

            # I3 Inception层
            InceptionV3(192, 64, (96, 96, 128), (16, 32, 32, 32, 32), 32),

            # I4 Inception层
            InceptionV3(256, 128, (128, 128, 192), (32, 32, 96, 96, 96), 64),

            # P4 最大值池化层
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2),

            # I5 Inception层
            InceptionV3(480, 192, (96, 96, 208), (16, 16, 48, 48, 48), 64),

            # I6 Inception层
            InceptionV3(512, 160, (112, 112, 224), (24, 24, 64, 64, 64), 64),

            # I7 Inception层
            InceptionV3(512, 128, (128, 128, 256), (24, 24, 64, 64, 64), 64),

            # I8 Inception层
            InceptionV3(512, 112, (144, 144, 288), (32, 32, 64, 64, 64), 64),

            # I9 Inception层
            InceptionV3(528, 256, (160, 160, 320), (32, 32, 128, 128, 128), 128),

            # P9 最大值池化层
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2),

            # I10 Inception层
            InceptionV3(832, 256, (160, 160, 320), (32, 32, 128, 128, 128), 128),

            # I11 Inception层
            InceptionV3(832, 384, (192, 192, 384), (48, 48, 128, 128, 128), 128),

            # P11 平均值池化层
            nn.AvgPool2d(kernel_size=7, padding=0, stride=1),

            # 展平
            nn.Flatten(),

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
