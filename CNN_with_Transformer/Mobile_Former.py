#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


# Mobile-Former中的自注意力块
class self_attn(nn.Module):
    def __init__(self, heads, HW, token_dims, token_nums):
        """
        :param heads: 自注意力中头的数目
        :param token_dims: 令牌的维度
        :param HW: 图像宽高的乘积
        :param token_nums: 令牌的数量，一般认为等于6
        """
        super(self_attn, self).__init__()
        # 每个头平均拥有tokens
        self.heads = heads
        self.d = token_nums // heads
        self.scale_factor = 1.0 / (self.d**0.5)

        # 获取键、值的全连接层
        self.query_layer = nn.Linear(HW, HW//self.heads, bias=False)
        self.key_layer = nn.Linear(token_nums, self.d, bias=False)
        self.value_layer = nn.Linear(token_nums, self.d, bias=False)

        # 输出的全连接层
        self.fc_out = nn.Linear(self.d * heads, token_nums)

    # 前向传播函数
    def forward(self, x, z):
        """
        :param x: 输入图像，大小为(batch_size, channels, height, width)
        :param z: 令牌，大小为(token_dims, token_nums)
        :return: z'，输出的令牌，大小为(token_dims, token_nums)
        """
        # 将x的后两个维度进行压缩
        bs, C, _, _ = x.shape
        x = x.reshape(bs, C, -1)  # 大小为(batch_size, channels, height*width)

        # 获取查询、键、值
        query = self.query_layer(x)  # 大小为(batch_size, channels, hw/heads)
        key = self.key_layer(z)  # 大小为(token_dims, self.d)
        value = self.value_layer(z)  # 大小为(token_dims, self.d)
