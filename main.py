import torch
from torch import nn
import torch.optim as opt
import transformers

from RESISC_CNN import test_functions
from RESISC_CNN import Init_dataset
from RESISC_CNN import AlexNet
from RESISC_CNN import VGG16
from RESISC_CNN import GoogleNetV1
from RESISC_CNN import GoogleNetV2
from RESISC_CNN import GoogleNetV3
from RESISC_CNN import GoogleNetV4
from RESISC_CNN import DenseNet
from RESISC_CNN import Pretrained_cnns
from RESISC_CNN import Precisions

from Transformer.Units import Coders

from CNN_with_Transformer import GoogleV2_cbam
from CNN_with_Transformer import GoogleV2_SE
from CNN_with_Transformer import Mobile_Former

from transformers import ViTFeatureExtractor, ViTForImageClassification

# 按照比例划分数据集
# root_dir = 'E:\\gzr\\RESISC\\NWPU-RESISC45'
# des_dir_train = 'E:\\gzr\\RESISC\\train_6'
# des_dir_test = 'E:\\gzr\\RESISC\\test_6'
# Init_dataset.move_images(root_dir, des_dir_train, des_dir_test, option=6)


path_train_6 = 'E:\\gzr\\RESISC\\train_6'
path_test_6 = 'E:\\gzr\\RESISC\\test_6'
#
transform_goo = Init_dataset.get_transform(224)
transform_tra = Init_dataset.get_transform(224)
#
# _, train_loader_goo = Init_dataset.get_dataset(path_train_6, 128, transform_goo)
# _, test_loader_goo = Init_dataset.get_dataset(path_test_6, 128, transform_goo)
#
_, train_loader_tra = Init_dataset.get_dataset(path_train_6, 64, transform_tra)
_, test_loader_tra = Init_dataset.get_dataset(path_test_6, 64, transform_tra)

# model_GoogleV2 = torch.load('GoogleV2.pth').to('cuda:0')
# model_GoogleV2_cbam = torch.load('GoogleV2_cbam.pth').to('cuda:0')
# model_GoogleV2_se = torch.load('GoogleV2_SE.pth').to('cuda:0')
model_transformer = torch.load('LightVit.pth').to('cuda:0')
#
# print('GoogleV2:')
# Precisions.Get_Precision(model_GoogleV2, train_loader_goo, 'train')
# Precisions.Get_Precision(model_GoogleV2, test_loader_goo, 'test')
#
# print('GoogleV2_cbam:')
# Precisions.Get_Precision(model_GoogleV2_cbam, train_loader_goo, 'train')
# Precisions.Get_Precision(model_GoogleV2_cbam, test_loader_goo, 'test')
#
# print('GoogleV2_se:')
# Precisions.Get_Precision(model_GoogleV2_se, train_loader_goo, 'train')
# Precisions.Get_Precision(model_GoogleV2_se, test_loader_goo, 'test')
#
print('Transformer:')
Precisions.Get_Precision(model_transformer, train_loader_tra, 'train')
Precisions.Get_Precision(model_transformer, test_loader_tra, 'test')

# model_light_vit, loss_light_vit = test_functions.test_transformer()
# torch.save(model_light_vit, 'LightVit.pth')
