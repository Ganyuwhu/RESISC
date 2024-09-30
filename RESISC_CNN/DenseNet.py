#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn import functional as f
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

"""
    构造密集网络DenseNet的网络结构
"""


class DenseBlock(nn.Module):
    def __init__(self, num_layers, num_features, growth_rate):
        super(DenseBlock, self).__init__()
        self.num_features = num_features
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            bn_1 = nn.BatchNorm2d(num_features + i * growth_rate)
            relu_1 = nn.LeakyReLU()
            conv_1 = nn.Conv2d(in_channels=num_features + i*growth_rate, out_channels=growth_rate, kernel_size=1,
                               padding=0, stride=1)
            bn_2 = nn.BatchNorm2d(growth_rate)
            relu_2 = nn.LeakyReLU()
            conv_3 = nn.Conv2d(in_channels=growth_rate, out_channels=growth_rate, kernel_size=3, padding=1, stride=1)

            self.layers.append(nn.Sequential(
                bn_1, relu_1, conv_1, bn_2, relu_2, conv_3
            ))

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, dim=1))
            features.append(new_feature)

        return torch.cat(features, dim=1)


class DenseNet(nn.Module):
    def __init__(self, growth_rate):
        super(DenseNet, self).__init__()
        self.net = nn.Sequential(
            # C1 7*7卷积层
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=3, stride=2),

            # P1 3*3最大值池化层
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2),

            # D2 DenseBlock块
            DenseBlock(num_layers=6, num_features=64, growth_rate=growth_rate),

            # T3 Transition块
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, padding=0, stride=1),
            nn.AvgPool2d(kernel_size=2, padding=0, stride=2),

            # D4 DenseBlock块
            DenseBlock(num_layers=12, num_features=128, growth_rate=growth_rate),

            # T5 Transition块
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, padding=0, stride=1),
            nn.AvgPool2d(kernel_size=2, padding=0, stride=2),

            # D6 DenseBlock块
            DenseBlock(num_layers=24, num_features=256, growth_rate=growth_rate),

            # T7 Transition块
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, padding=0, stride=1),
            nn.AvgPool2d(kernel_size=2, padding=0, stride=2),

            # D8 DenseBlock块
            DenseBlock(num_layers=16, num_features=512, growth_rate=growth_rate),

            # P8 平均值池化层
            nn.AvgPool2d(kernel_size=7, padding=0, stride=1),

            # 展平
            nn.Flatten(),

            # Dropout
            nn.Dropout(0.8),

            # 全连接层
            nn.Linear(1024, 45)

        )

    def forward(self, x):
        return self.net(x)
