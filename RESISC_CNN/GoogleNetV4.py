#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn import functional as f
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

"""
    构造残差网络GoogleNetV4的结构
"""


class stem(nn.Module):
    # 构造初始化stem块
    def __init__(self, _in_channels):
        super(stem, self).__init__()

        # 定义stem块内部结构

        # C1 卷积层，输出(32, 149, 149)
        self.p1 = nn.Conv2d(in_channels=_in_channels, out_channels=32, kernel_size=3, padding=0, stride=2)
        self.Bth1 = nn.BatchNorm2d(32)

        # C2 卷积层，输出(32, 147, 147)
        self.p2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=0, stride=1)
        self.Bth2 = nn.BatchNorm2d(32)

        # C3 卷积层，输出(64, 147, 147)
        self.p3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.Bth3 = nn.BatchNorm2d(64)

        # Layer4 分叉
        # P4.1 最大值池化层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, padding=0, stride=2)

        # P4.2 卷积层，输出(96, 73, 73)
        self.p4_2 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, padding=0, stride=2)
        self.Bth4_2 = nn.BatchNorm2d(96)

        # Layer4 额外进行一次cat操作，使得输出变为(160, 73, 73)

        # Layer5 分叉
        # P5.11 卷积层，输出(64, 73, 73)
        self.p5_11 = nn.Conv2d(in_channels=160, out_channels=64, kernel_size=1, padding=0, stride=1)
        self.Bth5_11 = nn.BatchNorm2d(64)

        # P5.12 卷积层，输出(96, 71, 71)
        self.p5_12 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, padding=0, stride=1)
        self.Bth5_12 = nn.BatchNorm2d(96)

        # P5.21 卷积层，输出(64, 73, 73)
        self.p5_21 = nn.Conv2d(in_channels=160, out_channels=64, kernel_size=1, padding=0, stride=1)
        self.Bth5_21 = nn.BatchNorm2d(64)

        # P5.22 卷积层，输出(64, 73, 73)
        self.p5_22 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(7, 1), padding=(3, 0), stride=1)
        self.Bth5_22 = nn.BatchNorm2d(64)

        # P5.23 卷积层，输出(64, 73, 73)
        self.p5_23 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 7), padding=(0, 3), stride=1)
        self.Bth5_23 = nn.BatchNorm2d(64)

        # P5.24 卷积层，输出(96, 71, 71)
        self.p5_24 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, padding=0, stride=1)
        self.Bth5_24 = nn.BatchNorm2d(96)

        # Layer5 额外进行一次cat操作，使输出变为(192, 71, 71)

        # Layer6 分叉
        # P6.1 卷积层，输出(192, 35, 35)
        self.p6_1 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=0, stride=2)
        self.Bth6_1 = nn.BatchNorm2d(192)

        # P6.2 最大值池化层
        self.p6_2 = nn.MaxPool2d(kernel_size=3, padding=0, stride=2)

        # Layer6 额外进行一次cat操作，使输出变为(384, 35, 35)

    def forward(self, x):
        # 进行前三层的前向传播
        x_3 = f.relu(self.Bth3(self.p3(f.relu(self.Bth2(self.p2(f.relu(self.Bth1(self.p1(x)))))))))

        # 第四层的前向传播
        x_41 = self.p4_1(x_3)
        x_42 = f.relu(self.Bth4_2(self.p4_2(x_3)))
        x_4 = torch.cat((x_41, x_42), dim=1)

        # 第五层的前向传播
        x_51 = f.relu(self.Bth5_12(self.p5_12(f.relu(self.Bth5_11(self.p5_11(x_4))))))
        x_52 = f.relu(self.Bth5_24(self.p5_24(f.relu(self.Bth5_23(self.p5_23(
            f.relu(self.Bth5_22(self.p5_22(f.relu(self.Bth5_21(self.p5_21(x_4))))))
        ))))))
        x_5 = torch.cat((x_51, x_52), dim=1)

        # 第六层的前向传播
        x_61 = f.relu(self.Bth6_1(self.p6_1(x_5)))
        x_62 = self.p6_2(x_5)
        x_6 = torch.cat((x_61, x_62), dim=1)

        return x_6


class InceptionA(nn.Module):
    # 构造InceptionA块，GoogleNetV4中所有v1版本的Inception块均不改变维度，因此所有层的stride均为1
    def __init__(self, _in_channels):
        super(InceptionA, self).__init__()

        # 路径1，一个平均值池化层，一个1*1卷积层，输出通道数为96
        self.p1_1 = nn.AvgPool2d(kernel_size=3, padding=1, stride=1)
        self.p1_2 = nn.Conv2d(in_channels=_in_channels, out_channels=96, kernel_size=1, padding=0, stride=1)
        self.Bth1_2 = nn.BatchNorm2d(96)

        # 路径2，一个1*1卷积层，输出通道数为96
        self.p2 = nn.Conv2d(in_channels=_in_channels, out_channels=96, kernel_size=1, padding=0, stride=1)
        self.Bth2 = nn.BatchNorm2d(96)

        # 路径3，一个1*1卷积层，输出通道数为64；一个3*卷积层，输出通道数为96
        self.p3_1 = nn.Conv2d(in_channels=_in_channels, out_channels=64, kernel_size=1, padding=0, stride=1)
        self.Bth3_1 = nn.BatchNorm2d(64)
        self.p3_2 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, padding=1, stride=1)
        self.Bth3_2 = nn.BatchNorm2d(96)

        # 路径4，一个1*1卷积层，输出通道数为64，两个3*3卷积层，输出通道数为96
        self.p4_1 = nn.Conv2d(in_channels=_in_channels, out_channels=64, kernel_size=1, padding=0, stride=1)
        self.Bth4_1 = nn.BatchNorm2d(64)
        self.p4_2 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, padding=1, stride=1)
        self.Bth4_2 = nn.BatchNorm2d(96)
        self.p4_3 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1, stride=1)
        self.Bth4_3 = nn.BatchNorm2d(96)

    def forward(self, x):
        x1 = f.relu(self.Bth1_2(self.p1_2(self.p1_1(x))))
        x2 = f.relu(self.Bth2(self.p2(x)))
        x3 = f.relu(self.Bth3_2(self.p3_2(f.relu(self.Bth3_1(self.p3_1(x))))))
        x4 = f.relu(self.Bth4_3(self.p4_3(f.relu(self.Bth4_2(self.p4_2(f.relu(self.Bth4_1(self.p4_1(x)))))))))
        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x


class InceptionB(nn.Module):
    # 构造InceptionB块的v1版本
    def __init__(self, _in_channels):
        super(InceptionB, self).__init__()

        # 路径1，一个平均值池化层和一个1*1卷积层，输出通道数为128
        self.p1_1 = nn.AvgPool2d(kernel_size=3, padding=1, stride=1)
        self.p1_2 = nn.Conv2d(in_channels=_in_channels, out_channels=128, kernel_size=1, padding=0, stride=1)
        self.Bth1_2 = nn.BatchNorm2d(128)

        # 路径2，一个1*1卷积层，输出通道数为384
        self.p2 = nn.Conv2d(in_channels=_in_channels, out_channels=384, kernel_size=1, padding=0, stride=1)
        self.Bth2 = nn.BatchNorm2d(384)

        # 路径3，一个1*1卷积层，输出通道数为192；一个7*1卷积层，输出通道数为224，一个1*7卷积层，输出通道数为256
        self.p3_1 = nn.Conv2d(in_channels=_in_channels, out_channels=192, kernel_size=1, padding=0, stride=1)
        self.Bth3_1 = nn.BatchNorm2d(192)
        self.p3_2 = nn.Conv2d(in_channels=192, out_channels=224, kernel_size=(7, 1), padding=(3, 0), stride=1)
        self.Bth3_2 = nn.BatchNorm2d(224)
        self.p3_3 = nn.Conv2d(in_channels=224, out_channels=256, kernel_size=(1, 7), padding=(0, 3), stride=1)
        self.Bth3_3 = nn.BatchNorm2d(256)

        # 路径4，一个1*1卷积层，输出通道数为192，一个1*7卷积层，输出通道数为192，一个7*1卷积层，输出通道数为224，一个1*7卷积层，输出通道数
        # 为224，一个7*1卷积层，输出通道数为256
        self.p4_1 = nn.Conv2d(in_channels=_in_channels, out_channels=192, kernel_size=1, padding=0, stride=1)
        self.Bth4_1 = nn.BatchNorm2d(192)
        self.p4_2 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(1, 7), padding=(0, 3), stride=1)
        self.Bth4_2 = nn.BatchNorm2d(192)
        self.p4_3 = nn.Conv2d(in_channels=192, out_channels=224, kernel_size=(7, 1), padding=(3, 0), stride=1)
        self.Bth4_3 = nn.BatchNorm2d(224)
        self.p4_4 = nn.Conv2d(in_channels=224, out_channels=224, kernel_size=(1, 7), padding=(0, 3), stride=1)
        self.Bth4_4 = nn.BatchNorm2d(224)
        self.p4_5 = nn.Conv2d(in_channels=224, out_channels=256, kernel_size=(7, 1), padding=(3, 0), stride=1)
        self.Bth4_5 = nn.BatchNorm2d(256)

    def forward(self, x):
        x_1 = f.relu(self.Bth1_2(self.p1_2(self.p1_1(x))))
        x_2 = f.relu(self.Bth2(self.p2(x)))
        x_3 = f.relu(self.Bth3_3(self.p3_3(f.relu(self.Bth3_2(self.p3_2(f.relu(self.Bth3_1(self.p3_1(x)))))))))
        x_4 = f.relu(self.Bth4_5(self.p4_5(f.relu(self.Bth4_4(self.p4_4(f.relu(self.Bth4_3(self.p4_3(
            f.relu(self.Bth4_2(self.p4_2(f.relu(self.Bth4_1(self.p4_1(x))))))
        )))))))))
        x = torch.cat((x_1, x_2, x_3, x_4), dim=1)

        return x


class InceptionC(nn.Module):
    # 构造InceptionC块的v1版本
    def __init__(self, _in_channels):
        super(InceptionC, self).__init__()

        # 路径1，一个平均值池化层和一个1*1卷积层，输出通道数为256
        self.p1_1 = nn.AvgPool2d(kernel_size=3, padding=1, stride=1)
        self.p1_2 = nn.Conv2d(in_channels=_in_channels, out_channels=256, kernel_size=1, padding=0, stride=1)
        self.Bth1_2 = nn.BatchNorm2d(256)

        # 路径2，一个1*1卷积层，输出通道数为256
        self.p2 = nn.Conv2d(in_channels=_in_channels, out_channels=256, kernel_size=1, padding=0, stride=1)
        self.Bth2 = nn.BatchNorm2d(256)

        # 路径3，一个1*1卷积层，输出通道数为384，然后分叉
        self.p3 = nn.Conv2d(in_channels=_in_channels, out_channels=384, kernel_size=1, padding=0, stride=1)
        self.Bth3 = nn.BatchNorm2d(384)

        # 路径3第一条支线，一个1*3卷积层，输出通道数为256
        self.p3_1 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(1, 3), padding=(0, 1), stride=1)
        self.Bth3_1 = nn.BatchNorm2d(256)

        # 路径3第二条支线，一个3*1卷积层，输出通道数为256
        self.p3_2 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 1), padding=(1, 0), stride=1)
        self.Bth3_2 = nn.BatchNorm2d(256)

        # 路径4
        self.p4_1 = nn.Conv2d(in_channels=_in_channels, out_channels=384, kernel_size=1, padding=0, stride=1)
        self.Bth4_1 = nn.BatchNorm2d(384)
        self.p4_2 = nn.Conv2d(in_channels=384, out_channels=448, kernel_size=(1, 3), padding=(0, 1), stride=1)
        self.Bth4_2 = nn.BatchNorm2d(448)
        self.p4_3 = nn.Conv2d(in_channels=448, out_channels=512, kernel_size=(3, 1), padding=(1, 0), stride=1)
        self.Bth4_3 = nn.BatchNorm2d(512)

        # 路径4第一条支线
        self.p4_4_1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 1), padding=(1, 0), stride=1)
        self.Bth4_4_1 = nn.BatchNorm2d(256)

        # 路径4第二条支线
        self.p4_4_2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 3), padding=(0, 1), stride=1)
        self.Bth4_4_2 = nn.BatchNorm2d(256)

    def forward(self, x):
        x_1 = f.relu(self.Bth1_2(self.p1_2(self.p1_1(x))))
        x_2 = f.relu(self.Bth2(self.p2(x)))
        x_3 = f.relu(self.Bth3(self.p3(x)))
        x_3_1 = f.relu(self.Bth3_1(self.p3_1(x_3)))
        x_3_2 = f.relu(self.Bth3_2(self.p3_2(x_3)))
        x_4 = f.relu(self.Bth4_3(self.p4_3(f.relu(self.Bth4_2(self.p4_2(f.relu(self.Bth4_1(self.p4_1(x)))))))))
        x_4_1 = f.relu(self.Bth4_4_1(self.p4_4_1(x_4)))
        x_4_2 = f.relu(self.Bth4_4_2(self.p4_4_2(x_4)))

        x = torch.cat((x_1, x_2, x_3_1, x_3_2, x_4_1, x_4_2), dim=1)

        return x


class GoogleNet_V4(nn.Module):
    def __init__(self):
        super(GoogleNet_V4, self).__init__()
        self.net = nn.Sequential(
            # S1 stem块
            stem(3),

            # IA2 4个InceptionA块
            InceptionA(384),
            InceptionA(384),
            InceptionA(384),
            InceptionA(384),

            # R3 Reduction层
            nn.Conv2d(in_channels=384, out_channels=1024, kernel_size=3, padding=0, stride=2),

            # IB4 7个InceptionB块
            InceptionB(1024),
            InceptionB(1024),
            InceptionB(1024),
            InceptionB(1024),
            InceptionB(1024),
            InceptionB(1024),
            InceptionB(1024),

            # R5 Reduction层
            nn.Conv2d(in_channels=1024, out_channels=1536, kernel_size=3, padding=0, stride=2),

            # IC6 3个Inception层
            InceptionC(1536),
            InceptionC(1536),
            InceptionC(1536),

            # P7 平均值池化层
            nn.AvgPool2d(kernel_size=8, padding=0, stride=1),

            # 展平
            nn.Flatten(),

            # DropOut正则化
            nn.Dropout(0.8),

            nn.Linear(1536, 45)

        )

    def forward(self, x):
        return self.net(x)


# 制作数据集
def Get_dataset():
    # 创建数据集

    # 数据集转换参数
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(299, antialias=None),
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
