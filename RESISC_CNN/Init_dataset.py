#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import random
import shutil
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

"""
    创建RESISC数据集
"""


# 定义数据集类
class RESISC_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        # 初始化数据集
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for class_name in os.listdir(self.root_dir):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for image_name in os.listdir(class_dir):
                    image_path = os.path.join(class_dir, image_name)
                    self.image_paths.append(image_path)
                    self.labels.append(class_name)  # 这里的class_name是由字符串构成的，需要进行转化

        # 将label转化为整型数组
        label_unique = np.unique(self.labels)  # 提取全部类型
        label_dict = {}  # 将类型和一个整数用字典的形式绑定
        label_int = 0
        for label in label_unique:
            label_dict[label] = label_int
            label_int += 1

        for index in range(len(self.labels)):
            self.labels[index] = label_dict[self.labels[index]]

        # 转化为独热编码
        self.labels = to_onehot(self.labels, 45)

    def __len__(self):
        # 返回数据集的长度
        return len(self.image_paths)

    def __getitem__(self, index):
        # 根据索引获取图像
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')  # 转化为RGB格式

        if self.transform:
            image = self.transform(image)  # 利用transform进一步转化格式

        label = self.labels[index]  # 获取图像对应的类别

        return image, label


# transform预处理
def get_transform(image_size=224):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # 调整图像大小
        transforms.ToTensor(),  # 将图像转变为张量形式
    ]
    )

    return transform


# 创建数据集
def get_dataset(path=None, batch_size=32, transform=None):
    train_dataset = RESISC_Dataset(root_dir=path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_dataset, train_loader


# 手动将labels转化为独热编码
def to_onehot(labels, num_classes):
    # 创建一个全零矩阵
    onehot = np.zeros((len(labels), num_classes))

    # 使用索引将labels对应的元素设置为1
    for index in range(len(labels)):
        onehot[index, labels[index]] = 1

    return onehot


# 数据增强--对图片进行翻转
def get_flipped(root_dir, output_dir):
    for class_name in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, class_name)
        if os.path.isdir(class_dir):
            output_path = os.path.join(output_dir, class_name)

            # 创建文件夹
            if not os.path.exists(output_path):
                os.mkdir(output_path)

            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                image = Image.open(image_path)

                # 水平翻转图像
                flipped_image_lr = image.transpose(Image.FLIP_LEFT_RIGHT)

                # 垂直翻转图像
                flipped_image_tb = image.transpose(Image.FLIP_TOP_BOTTOM)

                # 保存翻转后的图像
                flipped_image_lr_path = os.path.join(output_path, 'flipped_lr'+image_name)
                flipped_image_tb_path = os.path.join(output_path, 'flipped_tb'+image_name)

                flipped_image_lr.save(flipped_image_lr_path)
                flipped_image_tb.save(flipped_image_tb_path)


# 移动图片
def move_images(root_dir, des_dir_train, des_dir_test, option=1):
    # 若不存在目标文件夹，则创建它
    if not os.path.isdir(des_dir_train):
        os.mkdir(des_dir_train)

    if not os.path.isdir(des_dir_test):
        os.mkdir(des_dir_test)

    for class_name in os.listdir(root_dir):
        # 获取类别源地址
        class_root_dir = os.path.join(root_dir, class_name)

        # 获取类别根目录。若该目录不存在，则创建它
        class_des_dir_train = os.path.join(des_dir_train, class_name)
        if not os.path.isdir(class_des_dir_train):
            os.mkdir(class_des_dir_train)

        class_des_dir_test = os.path.join(des_dir_test, class_name)
        if not os.path.isdir(class_des_dir_test):
            os.mkdir(class_des_dir_test)

        # 用一个列表储存该类别下所有的图片
        list_images = [image for image in os.listdir(class_root_dir)]

        # 从列表中随机选取70*option张图片作为训练集
        selected_images = random.sample(list_images, 70*option)

        # 移动选中的图片
        for image in selected_images:
            image_root_dir = os.path.join(class_root_dir, image)
            image_des_dir_train = os.path.join(class_des_dir_train, image)
            shutil.move(image_root_dir, image_des_dir_train)

        # 将剩下的图片移动到测试集
        for image in set(list_images) - set(selected_images):
            image_root_dir = os.path.join(class_root_dir, image)
            image_des_dir_test = os.path.join(class_des_dir_test, image)
            shutil.move(image_root_dir, image_des_dir_test)

    print('move successfully!')
