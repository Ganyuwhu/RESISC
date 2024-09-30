#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channel, reduction):
        super(SEBlock, self).__init__()

        self.linear1 = nn.Linear(channel, channel // reduction)
        self.linear2 = nn.Linear(channel // reduction, channel)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        # x的输入为(batch_size, channels, height, weight)
        x_sq = torch.mean(x, dim=[2, 3], keepdim=True)

        (batch_size, channels, height, width) = (x_sq.shape[0], x_sq.shape[1], x_sq.shape[2], x_sq.shape[3])

        # 改变张量的形状以适应全连接层
        x_sq = x_sq.reshape(batch_size, height, width, -1)

        x_line1 = self.relu(self.linear1(x_sq))
        weight = self.relu(self.linear2(x_line1))
        weight = weight.reshape(batch_size, channels, height, width)

        # 将输入与weight相乘
        x = torch.mul(x, weight)

        return x

