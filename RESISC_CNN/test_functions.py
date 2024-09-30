#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings

import torch.nn
import torch.nn as nn
from RESISC_CNN import testNet
from RESISC_CNN import Init_dataset

from RESISC_CNN import AlexNet
from RESISC_CNN import VGG16
from RESISC_CNN import GoogleNetV1
from RESISC_CNN import GoogleNetV2
from RESISC_CNN import GoogleNetV3
from RESISC_CNN import GoogleNetV4
from RESISC_CNN import DenseNet

from Transformer.Units import Coders

from RESISC_CNN import Pretrained_cnns

# 忽略 NCCL 相关的 UserWarning
warnings.filterwarnings("ignore", category=UserWarning, message=".*NCCL.*")

# 数据库路径
path_train_1 = 'E:\\gzr\\RESISC\\train_1'
path_test_1 = 'E:\\gzr\\RESISC\\test_1'

path_train_2 = 'E:\\gzr\\RESISC\\train_2'
path_test_2 = 'E:\\gzr\\RESISC\\test_2'

path_train_6 = 'E:\\gzr\\RESISC\\train_6'
path_test_6 = 'E:\\gzr\\RESISC\\test_6'

"""
    将各个网络模型的测试函数整合到一个包中，方便管理
"""


# 测试AlexNet
def test_AlexNet(learning_rate=0.001, image_size=224, batch_size=128, scheduler='None'):
    model_alex = AlexNet.AlexNet().to('cuda:0')
    print(model_alex.__class__.__name__, '训练结果：')
    transform = Init_dataset.get_transform(image_size)
    train_data, train_loader = Init_dataset.get_dataset(path_train_1, batch_size, transform)
    test_data, test_loader = Init_dataset.get_dataset(path_test_1, batch_size, transform)
    model_alex, loss_alex = testNet.test(model_alex, train_loader, test_loader,
                                         learning_rate, batch_size=batch_size, scheduler_type=scheduler)

    return model_alex, loss_alex


# 测试GoogleNetV1
def test_GoogleNetV1(learning_rate=0.01, image_size=224, batch_size=128, scheduler='None'):
    model_v1 = GoogleNetV1.GoogleNet_V1().to('cuda:0')
    print(model_v1.__class__.__name__, '训练结果：')
    transform = Init_dataset.get_transform(image_size)
    train_data, train_loader = Init_dataset.get_dataset(path_train_1, batch_size, transform)
    test_data, test_loader = Init_dataset.get_dataset(path_test_1, batch_size, transform)
    model_v1, loss_v1 = testNet.test(model_v1, train_loader, test_loader,
                                     learning_rate, batch_size=batch_size, scheduler_type=scheduler)
    
    return model_v1, loss_v1


# 测试GoogleNetV2
def test_GoogleNetV2(learning_rate=0.001, image_size=224, batch_size=128, scheduler='None'):
    model_v1 = GoogleNetV2.GoogleNet_V2().to('cuda:0')
    print(model_v1.__class__.__name__, '训练结果：')
    transform = Init_dataset.get_transform(image_size)
    train_data, train_loader = Init_dataset.get_dataset(path_train_6, batch_size, transform)
    test_data, test_loader = Init_dataset.get_dataset(path_test_6, batch_size, transform)
    model_v1, loss_v1 = testNet.test(model_v1, train_loader, test_loader,
                                     learning_rate, batch_size=batch_size, scheduler_type=scheduler)

    return model_v1, loss_v1


# 测试GoogleNetV3
def test_GoogleNetV3(learning_rate=0.001, image_size=224, batch_size=128, scheduler='None'):
    model_v1 = GoogleNetV3.GoogleNet_V3().to('cuda:0')
    print(model_v1.__class__.__name__, '训练结果：')
    transform = Init_dataset.get_transform(image_size)
    train_data, train_loader = Init_dataset.get_dataset(path_train_6, batch_size, transform)
    test_data, test_loader = Init_dataset.get_dataset(path_test_6, batch_size, transform)
    model_v1, loss_v1 = testNet.test(model_v1, train_loader, test_loader,
                                     learning_rate, batch_size=batch_size, scheduler_type=scheduler)

    return model_v1, loss_v1


# 测试GoogleNetV4
def test_GoogleNetV4(learning_rate=0.001, image_size=224, batch_size=128, scheduler='None'):
    model_v1 = GoogleNetV4.GoogleNet_V4().to('cuda:0')
    print(model_v1.__class__.__name__, '训练结果：')
    transform = Init_dataset.get_transform(image_size)
    train_data, train_loader = Init_dataset.get_dataset(path_train_6, batch_size, transform)
    test_data, test_loader = Init_dataset.get_dataset(path_test_6, batch_size, transform)
    model_v1, loss_v1 = testNet.test(model_v1, train_loader, test_loader,
                                     learning_rate, batch_size=batch_size, scheduler_type=scheduler)

    return model_v1, loss_v1


# 测试VGG16
def test_VGG(learning_rate=0.001, image_size=224, batch_size=50, scheduler='None'):
    model_vgg = VGG16.vgg_16().to('cuda:0')
    print(model_vgg.__class__.__name__, '训练结果：')
    transform = Init_dataset.get_transform(image_size)
    train_data, train_loader = Init_dataset.get_dataset(path_train_1, batch_size, transform)
    test_data, test_loader = Init_dataset.get_dataset(path_test_1, batch_size, transform)
    model_vgg, loss_vgg = testNet.test(model_vgg, train_loader, test_loader,
                                       learning_rate, batch_size=batch_size, scheduler_type=scheduler)

    return model_vgg, loss_vgg


# 测试DenseNet
def test_DenseNet(learning_rate=0.001, image_size=224, batch_size=64, scheduler='None'):
    model_den = DenseNet.DenseNet(growth_rate=32).to('cuda:0')
    print(model_den.__class__.__name__, '训练结果：')
    transform = Init_dataset.get_transform(image_size)
    train_data, train_loader = Init_dataset.get_dataset(path_train_1, batch_size, transform)
    test_data, test_loader = Init_dataset.get_dataset(path_test_1, batch_size, transform)
    model_den, loss_den = testNet.test(model_den, train_loader, test_loader,
                                       learning_rate, batch_size=batch_size, scheduler_type=scheduler)

    return model_den, loss_den


# 测试预训练AlexNet
def test_preAlex(learning_rate=0.001, image_size=224, batch_size=128):
    model_prealex = Pretrained_cnns.load_alex(feature_num=45).to('cuda:0')
    print(model_prealex.__class__.__name__, '训练结果：')
    transform = Init_dataset.get_transform(image_size)
    train_data, train_loader = Init_dataset.get_dataset(path_train_2, batch_size, transform)
    test_data, test_loader = Init_dataset.get_dataset(path_test_2, batch_size, transform)
    model_ft_alex, loss_alex = testNet.test(model_prealex, train_loader, test_loader, batch_size=batch_size)

    return model_ft_alex, loss_alex


# 测试预训练的vgg16
def test_prevgg(learning_rate=0.001, image_size=224, batch_size=50):
    model_prevgg = Pretrained_cnns.load_vgg(feature_num=45).to('cuda:0')
    print(model_prevgg.__class__.__name__, '训练结果：')
    transform = Init_dataset.get_transform(image_size)
    train_data, train_loader = Init_dataset.get_dataset(path_train_2, batch_size, transform)
    test_data, test_loader = Init_dataset.get_dataset(path_test_2, batch_size, transform)
    model_ft_vgg, loss_vgg = testNet.test(model_prevgg, train_loader, test_loader, batch_size=batch_size)

    return model_ft_vgg, loss_vgg


# 测试预训练的GoogleNet
def test_pregoogle(learning_rate=0.01, image_size=224, batch_size=128):
    model_pregoogle = Pretrained_cnns.load_google(feature_num=45).to('cuda:0')
    print(model_pregoogle.__class__.__name__, '训练结果：')
    transform = Init_dataset.get_transform(image_size)
    train_data, train_loader = Init_dataset.get_dataset(path_train_2, batch_size, transform)
    test_data, test_loader = Init_dataset.get_dataset(path_test_2, batch_size, transform)
    model_ft_google, loss_google = testNet.test(model_pregoogle, train_loader, test_loader, batch_size=batch_size)

    return model_ft_google, loss_google


# 测试Transformer
def test_transformer(learning_rate=0.001, image_size=224, batch_size=64, embed_size=28, patch_size=16,
                     output_size=45):
    model_transformer = Coders.VisionTransformer(N=6, image_size=image_size, embed_size=embed_size,
                                                 patch_size=patch_size, output_size=output_size).to('cuda:0')
    print(model_transformer.__class__.__name__, '训练结果：')
    transform = Init_dataset.get_transform(image_size)
    train_data, train_loader = Init_dataset.get_dataset(path_train_6, batch_size, transform)
    test_data, test_loader = Init_dataset.get_dataset(path_test_6, batch_size, transform)
    model_transformer, loss_transformer = testNet.test(model_transformer, train_loader, test_loader, learning_rate,
                                                       batch_size=batch_size)

    return model_transformer, loss_transformer


# # 测试SE_Google
# def test_SE_Google(learning_rate=0.01, image_size=224, batch_size=128):
#     model_segoogle = SE_Google.SE_GoogleNet_V1().to('cuda:0')
#     print(model_segoogle.__class__.__name__, '训练结果：')
#     transform = Init_dataset.get_transform(image_size)
#     train_data, train_loader = Init_dataset.get_dataset(path_train_6, batch_size, transform)
#     test_data, test_loader = Init_dataset.get_dataset(path_test_6, batch_size, transform)
#     model_segoogle, loss_segoogle = testNet.test(model_segoogle, train_loader, test_loader, batch_size=batch_size)
#
#     return model_segoogle, loss_segoogle
#
#
# # 测试CBAM_Google
# def test_CBAM_Google(learning_rate=0.01, image_size=224, batch_size=128):
#     model_cbam_google = CBAM_Google.CBAM_GoogleNet_V1().to('cuda:0')
#     print(model_cbam_google.__class__.__name__, '训练结果：')
#     transform = Init_dataset.get_transform(image_size)
#     train_data, train_loader = Init_dataset.get_dataset(path_train_6, batch_size, transform)
#     test_data, test_loader = Init_dataset.get_dataset(path_test_6, batch_size, transform)
#     model_cbam_google, loss_cbam_google = testNet.test(model_cbam_google, train_loader, test_loader,
#                                                        batch_size=batch_size)
#
#     return model_cbam_google, loss_cbam_google
