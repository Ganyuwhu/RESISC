#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    加载PyTorch上在ImageNet上预训练的ImageNet模型
"""

import torch
import torch.nn as nn
import torchvision.models as models

dict_alex = "C:\\Users\\user2\\.cache\\torch\\hub\\checkpoints\\alexnet.pth"
dict_vgg = "C:\\Users\\user2\\.cache\\torch\\hub\\checkpoints\\vgg16.pth"
dict_google = "C:\\Users\\user2\\.cache\\torch\\hub\\checkpoints\\googlenet.pth"


def load_alex(feature_num):
    """
    :param feature_num: 数据集的类别数，用来修改最后的全连接层
    :return: 修改后的预训练AlexNet
    """

    # 预训练模型文件
    state_dict = torch.load(dict_alex)

    # 加载一个空模型
    model = models.alexnet()

    # 读取
    model.load_state_dict(state_dict)

    # 将模型转为调试模式
    model.eval()

    # 获取原全连接层的输入维度
    num_ftrs = model.classifier[6].in_features

    # 将最后一个全连接层的输出维度替换成45
    model.classifier[6] = nn.Linear(num_ftrs, feature_num)

    return model


def load_vgg(feature_num):
    """
    :param feature_num: 数据集的类别数，用来修改最后的全连接层
    :return: 修改后的预训练vgg
    """

    # 预训练模型文件
    state_dict = torch.load(dict_vgg)

    # 加载一个空模型
    model = models.vgg16()

    # 读取
    model.load_state_dict(state_dict)

    # 将模型转为调试模式
    model.eval()

    # 获取原全连接层的输入维度
    num_ftrs = model.classifier[6].in_features

    # 将最后一个全连接层的输出维度替换成45
    model.classifier[6] = nn.Linear(num_ftrs, feature_num)

    return model


def load_google(feature_num=45):
    """
    :param feature_num: 数据集的类别数，用来修改最后的全连接层
    :return: 修改后的预训练vgg
    """

    # 预训练模型文件
    state_dict = torch.load(dict_google)

    # 加载一个空模型
    model = models.googlenet()

    # 读取
    model.load_state_dict(state_dict)

    # 将模型转为调试模式
    model.eval()

    # 获取全连接层的输入项
    num_ftrs = model.fc.in_features

    # 改变全连接层的输出项
    model.fc = nn.Linear(num_ftrs, feature_num)

    return model

