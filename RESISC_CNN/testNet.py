#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as opt
import time
from matplotlib_inline import backend_inline
from RESISC_CNN import Schedulers
from RESISC_CNN import Precisions

backend_inline.set_matplotlib_formats('svg')


def test(_model, _train_loader, _test_loader, _learning_rate=0.001, final_layer_lr=0.01, last_layer=None,
         _loss_fn=nn.CrossEntropyLoss(), _momentum=0.9, _decay=0.005,
         batch_size=64, epochs=15000, scheduler_type='None', ):
    losses = []

    loss_temp = [0] * batch_size
    epochs = epochs // 150
    lr = _learning_rate

    # 如果传入不为None的last_layer参数，则对最后一层的学习率进行单独修改
    if last_layer is not None:
        # 获取模型的所有参数
        params = list(_model.parameters())

        final_layer_name = ['classifier.6.weight', 'classifier.6.bias']

        param_groups = [
            {'params': [p for n, p in _model.named_parameters() if n not in final_layer_name], 'lr': _learning_rate},
            {'params': [p for n, p in _model.named_parameters() if n in final_layer_name], 'lr': final_layer_lr}
        ]

        optimizer = opt.Adam(params=param_groups, weight_decay=_decay)
    else:
        optimizer = opt.Adam(_model.parameters(), lr=lr, weight_decay=_decay)

    scheduler = Schedulers.init_scheduler(base_lr=lr, Type=scheduler_type)
    print('使用的学习率下降调度器为：', scheduler.__class__.__name__)
    result = 0

    for _epochs in range(epochs):
        loss_epoch = []

        if scheduler_type == 'Origin':
            lr = scheduler(result)
        elif scheduler_type == 'Factor' or 'None':
            lr = scheduler()
        else:
            lr = scheduler(_epochs + 1)

        for (x, y) in _train_loader:
            (x, y) = (x.to('cuda:0'), y.to('cuda:0'))
            optimizer.zero_grad()
            predict = _model(x)
            loss = _loss_fn(predict, y)
            losses.append(loss.item())
            loss_epoch.append(loss.item())
            loss.backward()
            optimizer.step()

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        diff = [(a - b) ** 2 for a, b in zip(loss_epoch, loss_temp)]
        loss_total = sum([a**2 for a in loss_epoch])
        result = sum(diff)

        print(
            f'第 {_epochs + 1} 次训练: loss = {loss_total}, loss_max = {max(loss_epoch)}'
        )

        loss_temp = loss_epoch

    return _model, losses

