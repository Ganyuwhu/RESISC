#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class Channel_attention(nn.Module):
    # CBAM中的通道注意力模块
    def __init__(self, channel, reduction):
        """
        :param channel: 输入通道数
        :param reduction: 通道下降率，用于生成MLP的隐藏层
        """
        super(Channel_attention, self).__init__()
        # 使用自适应池化层保证输出的维度恒定为1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)

        # MLP层
        self.mlp = nn.Sequential(
            nn.Linear(channel, channel//reduction),
            nn.LeakyReLU(),
            nn.Linear(channel//reduction, channel),
        )

        self.sig = nn.Sigmoid()

    # 前向传播函数
    def forward(self, x):
        # 获取池化层的输出，输出大小为(batch_size, channels, height, width)
        x_avg = self.avgpool(x)
        x_max = self.maxpool(x)

        (batch_size, channels, height, width) = (x_avg.shape[0], x_avg.shape[1], x_avg.shape[2], x_avg.shape[3])

        # 将x_avg和x_max进行翻转以适应全连接层
        x_avg = x_avg.reshape(batch_size, height, width, -1)
        x_max = x_max.reshape(batch_size, height, width, -1)

        x_avg = self.mlp(x_avg)
        x_max = self.mlp(x_max)

        output = self.sig(x_avg + x_max)

        output = output.reshape(batch_size, channels, height, width)

        return output


class Spatial_attention(nn.Module):
    # CBAM中的空间注意力模块
    def __init__(self, kernel_size=7):
        super(Spatial_attention, self).__init__()
        padding = kernel_size // 2

        # 卷积层
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # 沿通道的维度做最大值池化和平均值池化
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)

        # 拼接二者
        x_cat = torch.cat([x_avg, x_max], dim=1)

        x_conv = self.conv(x_cat)

        output = self.sig(x_conv)

        return output


class CBAM(nn.Module):
    def __init__(self, channel, reduction, kernel_size):
        super(CBAM, self).__init__()

        # 创建注意力层
        self.channel_att = Channel_attention(channel, reduction)
        self.spatial_att = Spatial_attention(kernel_size)

    def forward(self, x):
        # 先经过一次通道注意力
        x_channel = self.channel_att(x)
        x = torch.mul(x_channel, x)

        # 将输出经过一次空间注意力
        x_spatial = self.spatial_att(x)
        x = torch.mul(x_spatial, x)

        return x
