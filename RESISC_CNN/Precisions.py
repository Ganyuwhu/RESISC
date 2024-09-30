#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np
from RESISC_CNN import Init_dataset
from sklearn.metrics import recall_score

np.set_printoptions(threshold=1000000)


def Get_Precision(model, loader, data_type):
    correct = 0
    total = 0
    with torch.no_grad():
        for (x, y) in loader:
            (x, y) = (x.to('cuda:0'), y.to('cuda:0'))
            predict = model(x)
            _, predict = torch.max(predict.data, dim=1)
            predict = torch.Tensor(Init_dataset.to_onehot(predict, 45)).to('cuda:0')

            for i, j in zip(predict, y):
                if (i == j).all():
                    correct += 1
            total += y.size(0)

    accuracy = correct * 100 / total

    if data_type == 'train':
        print(f'对训练集的正确率为:{accuracy}%')

    elif data_type == 'check':
        print(f'对验证集的正确率为:{accuracy}%')

    else:
        print(f'对测试集的正确率为:{accuracy}%')


# def Get_callback(model, loader, data_type):
#     with torch.no_grad():
#         for (x, y) in loader:
#             (x, y) = (x.to('cuda:0'), y.to('cuda:0'))
#             predict = model(x)
#             predict = torch.argmax(predict, dim=1)
#             predict = torch.Tensor(Init_dataset.to_onehot(predict, 45)).to('cuda:0')
#
#     recall = recall_score(y.cpu().numpy(), predict.cpu().numpy(), average='micro')
#
#     if data_type == 'train':
#         print(f'对训练集的召回率为:{recall * 100}%')
#
#     elif data_type == 'check':
#         print(f'对验证集的召回率为:{recall * 100}%')
#
#     else:
#         print(f'对测试集的召回率为:{recall * 100}%')


# def Get_F1score(model, loader, data_type):
#     correct, total = 0, 0
#     with torch.no_grad():
#         for (x, y) in loader:
#             (x, y) = (x.to('cuda:0'), y.to('cuda:0'))
#             predict = model(x)
#             _, predict_ac = torch.max(predict.data, dim=1)
#             predict = torch.Tensor(Init_dataset.to_onehot(predict, 45)).to('cuda:0')
#             for i, j in zip(predict, y):
#                 if (i == j).all():
#                     correct += 1
#             total += y.size(0)
#
#             predict_cb = torch.argmax(predict, dim=1)
#
#     accuracy = correct / total
#     recall = recall_score(y.cpu().numpy(), predict_cb.cpu().numpy(), average='micro')
#     f1_score = 2 / (1/accuracy + 1/recall)
#
#     if data_type == 'train':
#         print(f'对训练集的F1 score为:{f1_score}%')
#
#     elif data_type == 'check':
#         print(f'对验证集的F1 score为:{f1_score}%')
#
#     else:
#         print(f'对测试集的F1 score为:{f1_score}%')
