#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as f


# 编码器的模板类
class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__()

    def forward(self, X, *args):
        raise NotImplementedError


# 解码器的模板类
class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__()

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, *args):
        raise NotImplementedError


# 编码器-解码器架构
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dnc_X, *args):
        """
        :param enc_X: encoder的输入
        :param dnc_X: decoder的输入
        :param args: 其他变量
        :return: decoder的输出
        """
        # 获取定长编码状态
        enc_outputs = self.encoder(enc_X)

        # 设置解码器状态
        dnc_state = self.decoder.init_state(enc_outputs)

        return self.decoder(dnc_X, dnc_state)


# 缩放点积注意力机制架构
class Attention(nn.Module):
    def __init__(self, d_model, d, **kwargs):
        super(Attention, self).__init__()
        # 使用全连接层定义计算查询、键和值的线性变换
        self.query_layer = nn.Linear(d_model, d, bias=False)
        self.key_layer = nn.Linear(d_model, d, bias=False)
        self.value_layer = nn.Linear(d_model, d, bias=False)

        # 用于进行点积缩放
        self.scale_factor = 1.0 / (d ** 0.5)

    def forward(self, Q, K, V):
        """
        :param Q: 查询
        :param K: 键
        :param V: 值
        :return: output: 输出，attention_weight: 注意力权重矩阵
        """

        # 第一步，生成注意力评分函数，注意这里只对K的倒数第二和倒数第一个维度进行转置
        attention_score = torch.matmul(Q, K.transpose(-2, -1)*self.scale_factor)
        # attention_score的形状为(batch_size, T, T)，最后一个维度表示对应的注意力权重

        # 第二步，通过softmax对最后一个维度获取注意力权重
        attention_weight = f.softmax(attention_score, dim=-1)

        # 第四步，利用注意力权重对值进行加权求和
        output = torch.matmul(attention_weight, V)  # output形状为(batch_size, T, d)

        return output, attention_weight


# 基于缩放点积注意力的多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads, **kwargs):
        """
        :param d_model: 多头注意力对应的隐藏层单元数
        :param heads: 多头注意力中头的数量
        :param kwargs: 其他参数
        """
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.heads = heads
        self.d = d_model // heads
        self.scale_factor = 1.0 / (self.d**0.5)

        # 使用全连接层定义各个线性变换
        self.query_layer = nn.Linear(d_model, self.d)
        self.key_layer = nn.Linear(d_model, self.d)
        self.value_layer = nn.Linear(d_model, self.d)
        self.get_query = nn.ModuleList()
        self.get_key = nn.ModuleList()
        self.get_value = nn.ModuleList()
        for i in range(self.heads):
            self.get_query.append(nn.Linear(self.d, self.d))
            self.get_key.append(nn.Linear(self.d, self.d))
            self.get_value.append(nn.Linear(self.d, self.d))

        self.fc_out = nn.Linear(self.heads*self.d, self.d_model, bias=False)

    def forward(self, Q, K, V, mask=None):

        # 第一步，获取键、值、查询
        batch_size = K.shape[0]
        T = K.shape[1]
        d = K.shape[2]

        # 第二步，将h个头的键、值、查询进行整合
        cat_K = torch.zeros(self.heads, batch_size, T, d)
        cat_V = torch.zeros(self.heads, batch_size, T, d)
        cat_Q = torch.zeros(self.heads, batch_size, T, d)
        for i in range(self.heads):
            cat_K[i] = self.get_key[i](K)
            cat_V[i] = self.get_value[i](K)
            cat_Q[i] = self.get_query[i](K)

        cat_K = cat_K.reshape(batch_size, T, self.heads, d)
        cat_V = cat_V.reshape(batch_size, T, self.heads, d)
        cat_Q = cat_Q.reshape(batch_size, T, self.heads, d)

        # 第三步，利用torch.einsum进行批处理，计算其注意力评分函数
        attention_score = torch.einsum("bqhd, bkhd -> bhqk", [cat_Q, cat_K])
        # 由于一个字母不能在箭头右侧出现两次，因此用q和k代替T

        # 掩蔽操作
        if mask is not None:
            attention_score = attention_score.masked_fill(mask==0, 0)

        # 第四步，利用f.softmax函数计算注意力权重，由于是对每个head单独计算权重，因此需要指定dim=-1
        attention_weight = f.softmax(attention_score*self.scale_factor, dim=-1)

        # 第五步，计算将每个注意力汇聚concat后的输出
        out = torch.einsum("bhqk, bkhd -> bhqd", [attention_weight, cat_V]).to('cuda:0')
        out = out.reshape(batch_size, T, self.heads*d)

        # 第六步，通过全连接层得到输出
        output = self.fc_out(out)

        return output


# 基于三角函数的位置编码
class PositionEncoding(nn.Module):
    def __init__(self, max_len):
        super(PositionEncoding, self).__init__()
        # 创建一个足够大的矩阵，其中元素的值为其行数加1(python意义下)
        p_row = torch.arange(max_len)+1
        p_row = p_row.unsqueeze(1)
        self.max_len = max_len
        self.P = p_row.repeat(1, max_len)

    def forward(self, X):
        # 前向传播函数
        d = X.shape[-1]
        p_pow = torch.pow(10000, 2*((torch.arange(0, self.max_len)+1)//2)/d)
        p_pow = p_pow.unsqueeze(1)
        P = self.P / p_pow

        # 获取位置编码
        PE = torch.zeros(size=P.shape).to('cuda:0')
        PE[:, 0:d:2] = torch.cos(P[:, 0:d:2])
        PE[:, 1:d:2] = torch.sin(P[:, 1:d:2])

        X = X + PE[0:X.shape[-2], 0:X.shape[-1]]

        return X


# 逐位前馈神经网络
class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim):  # hidden_dim应远大于input_dim
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(float(input_dim)/hidden_dim)

    def forward(self, x):
        out = self.fc2(self.dropout(self.relu(self.fc1(x))))
        return out


# Transformer架构中的编码器块
class Transformer_Encoder_Block(nn.Module):
    def __init__(self, embed_size, num_features):
        """
        :param embed_size: 嵌入层的输出大小，同时也是sublayer的输出大小
        :param num_features: 输入图表的数目
        """
        super(Transformer_Encoder_Block, self).__init__()

        # 多头注意力 + 归一化
        self.multi_head = MultiHeadAttention(d_model=embed_size, heads=4)
        self.norm1 = nn.BatchNorm1d(num_features)
        # 逐位前馈 + 归一化
        self.ffn_layer = FFN(input_dim=embed_size, hidden_dim=5*embed_size)
        self.norm2 = nn.BatchNorm1d(num_features)

    def forward(self, x):
        # 0、 获取查询、键、值
        query = self.multi_head.query_layer(x)
        key = self.multi_head.key_layer(x)
        value = self.multi_head.value_layer(x)

        # 1、多头注意力+残差连接+归一化
        x_multi = self.multi_head(Q=query, K=key, V=value)
        x = x + x_multi
        x = self.norm1(x)

        # 2、逐位前馈+残差连接+归一化
        x_ffn = self.ffn_layer(x)
        x = x + x_ffn
        x = self.norm2(x)

        return x


# Transformer架构下的解码器
class Transformer_Decoder_Block(nn.Module):
    def __init__(self, embed_size, num_features):
        """
        :param embed_size: 嵌入层的输出大小，同时也是sublayer的输出大小
        :param num_features: 输入图表的数目
        """
        super(Transformer_Decoder_Block, self).__init__()
        # 掩蔽多头注意力 + 归一化
        self.mask_multi = MultiHeadAttention(d_model=embed_size, heads=4, mask=True)
        self.norm1 = nn.BatchNorm1d(num_features)
        # 编码器-解码器注意力 + 归一化
        self.ed_multi = MultiHeadAttention(d_model=embed_size, heads=4)
        self.norm2 = nn.BatchNorm1d(num_features)
        # 逐位前馈 + 归一化
        self.ffn_layer = FFN(input_dim=embed_size, hidden_dim=3*embed_size)
        self.norm3 = nn.BatchNorm1d(num_features)

    def forward(self, x, enc_output):
        """
        :param x: 输入项
        :param enc_output: 编码器的输出项
        :return:
        """
        # 0、 获取查询、键、值
        query = self.mask_multi.query_layer(x)
        key = self.mask_multi.key_layer(x)
        value = self.mask_multi.value_layer(x)

        # 1、掩蔽多头注意力 + 残差连接 + 归一化
        x_mask_multi = self.mask_multi(Q=query, K=key, V=value)
        x = x + x_mask_multi
        x = self.norm1(x)

        # 2、编码器-解码器注意力 + 残差连接 + 归一化
        # 键、值来自整个编码器的输出
        key = self.ed_multi.key_layer(enc_output)
        value = self.ed_multi.value_layer(enc_output)
        # 查询来自上一个解码器子层的输出
        query = self.ed_multi.query_layer(x)

        x_ed_multi = self.ed_multi(Q=query, K=key, V=value)
        x = x + x_ed_multi
        x = self.norm2(x)

        # 3、逐位前馈 + 归一化
        x_ffn = self.ffn_layer(x)
        x = x + x_ffn
        x = self.norm3(x)

        return x


# Transformer的编码器
class Transformer_Encoder(nn.Module):
    def __init__(self, N, vocab_size, embed_size, num_features):
        """
        :param N: 编码器中编码器块的个数
        :param vocab_size: 词元大小，即输入向量大小大小
        :param embed_size: 嵌入层的输出大小
        :param num_features: 输入通道数
        """
        super(Transformer_Encoder, self).__init__()
        self.num_blocks = N
        # 编码器前的嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # 位置编码
        self.pos_coding = PositionEncoding(1000)

        # 编码器块
        self.blocks = nn.Sequential()
        for i in range(N):
            self.blocks.add_module("block"+str(i),
                                   Transformer_Encoder_Block(embed_size, num_features))

    def forward(self, x):
        x = self.pos_coding(self.embedding(x))
        for i, block in enumerate(self.blocks):
            x = block(x)

        return x


# Transformer的解码器
class Transformer_Decoder(nn.Module):
    def __init__(self, N, vocab_size, embed_size, num_features):
        """
        :param N: 解码器块的数目
        :param vocab_size: 词表大小
        :param embed_size: 嵌入层输出大小
        :param num_features: 输入通道数
        """
        super(Transformer_Decoder, self).__init__()
        self.num_blocks = N

        # 编码器前的嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # 位置编码
        self.pos_coding = PositionEncoding(1000)

        # 解码器块
        self.blocks = nn.Sequential()
        for i in range(N):
            self.blocks.add_module("block"+str(i),
                                   Transformer_Decoder_Block(embed_size, num_features))

    def forward(self, dnc_input, enc_output):
        x = self.pos_coding(self.embedding(dnc_input))
        for i, block in enumerate(self.blocks):
            x = block(x, enc_output)

        return x


# Transformer架构
class Transformer(nn.Module):
    def __init__(self, N, vocab_size, embed_size, num_features, output_size):
        """
        :param N: 解码器块的数目
        :param vocab_size: 词表大小
        :param embed_size: 嵌入层输出大小
        :param num_features: 输入通道数
        :param output_size: 全连接层输出数
        """
        super(Transformer, self).__init__()
        self.encoder = Transformer_Encoder(N, vocab_size, embed_size, num_features)
        self.decoder = Transformer_Decoder(N, vocab_size, embed_size, num_features)
        self.fc_out = nn.Linear(embed_size, output_size)

    def forward(self, enc_input):
        # 首先计算在编码器中的输出
        x_encoder = self.encoder(enc_input)

        # 将编码器中的输出和解码器的输入结合
        result = self.decoder(x_encoder, x_encoder)

        # 经过全连接层得到输出
        output = self.fc_out(result)

        return output


# ViT的图像块划分
class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, embed_size):
        """
        :param image_size: 原图像的分辨率
        :param patch_size: 图像块的分辨率
        :param embed_size: 图像的RGB通道数
        """
        super(PatchEmbedding, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        # 单个图像需要划分的图像块序列所含的图像块个数
        self.num_patches = (image_size//patch_size)**2
        # 将图像划分为图像块，由于是不重叠的划分，因此可以使用卷积层进行划分操作
        self.proj = nn.Linear(3 * patch_size**2, embed_size)

    def forward(self, x):
        # 将图像划分为互不重叠的图像块
        x = x.unfold(dimension=3, size=self.patch_size, step=self.patch_size)
        # (batch_size, channels, image_size, num_patches**0.5, patch_size)
        x = x.unfold(dimension=2, size=self.patch_size, step=self.patch_size)
        # (batch_size, channels, num_patches**0.5, num_patches**0.5, patch_size, patch_size)

        # 将图像块展平
        batch_size = x.shape[0]
        x = x.reshape(batch_size, self.num_patches, -1)

        # 通过全连接层将向量转化为符合Embedding层的形状
        x = self.proj(x)

        return x


# ViT的EnCoder
class ViT_Encoder(nn.Module):
    def __init__(self, N, image_size, patch_size, embed_size):
        """
        :param N: 编码器中编码器块的个数
        :param image_size: 输入图片大小
        :param embed_size: 嵌入层的输出大小
        :param patch_size: 图片切片大小
        """
        super(ViT_Encoder, self).__init__()
        num_patches = (image_size//patch_size) ** 2
        num_features = 3 * (image_size**2)//num_patches

        self.num_blocks = N
        # 编码器前的嵌入层
        self.embedding = PatchEmbedding(image_size, patch_size, embed_size)

        # 位置编码
        self.pos_coding = PositionEncoding(1000)

        # 编码器块
        self.blocks = nn.Sequential()
        for i in range(N):
            self.blocks.add_module("block"+str(i),
                                   Transformer_Encoder_Block(embed_size, num_patches))

    def forward(self, x):
        x = self.pos_coding(self.embedding(x))
        for i, block in enumerate(self.blocks):
            x = block(x)

        return x


# ViT的Decoder
class ViT_Decoder(nn.Module):
    def __init__(self, N, image_size, patch_size, embed_size):
        """
        :param N: 解码器块的数目
        :param image_size: 词表大小
        :param embed_size: 嵌入层输出大小
        :param patch_size: 输入通道数
        """
        super(ViT_Decoder, self).__init__()
        num_patches = (image_size//patch_size) ** 2
        num_features = 3 * (image_size**2)//num_patches

        self.num_blocks = N

        # 编码器前的嵌入层
        self.embedding = PatchEmbedding(image_size, patch_size, embed_size)

        # 位置编码
        self.pos_coding = PositionEncoding(1000)

        # 解码器块
        self.blocks = nn.Sequential()
        for i in range(N):
            self.blocks.add_module("block"+str(i),
                                   Transformer_Decoder_Block(embed_size, num_patches))

    def forward(self, x, enc_output):
        x = self.pos_coding(self.embedding(x))
        for i, block in enumerate(self.blocks):
            x = block(x, enc_output)

        return x


# ViT块
class VisionTransformer(nn.Module):
    def __init__(self, N, image_size, embed_size, patch_size, output_size):
        """
        :param N: 解码器块的数目
        :param image_size: 图像大小
        :param embed_size: 嵌入层输出大小
        :param patch_size: 输入通道数
        :param output_size: 全连接层输出数
        """
        super(VisionTransformer, self).__init__()
        self.encoder = ViT_Encoder(N, image_size, patch_size, embed_size)
        self.decoder = ViT_Decoder(N, image_size, patch_size, embed_size)

        num_patches = (image_size//patch_size) ** 2
        num_features = 3 * (image_size**2)//num_patches

        self.fc_out = nn.Linear(embed_size*num_patches, output_size)

    def forward(self, x_encoder):
        # 首先计算在编码器中的输出
        x_enc_out = self.encoder(x_encoder)

        # 将编码器中的输出和解码器的输入结合
        result = self.decoder(x_encoder, x_enc_out)
        result = result.reshape(result.size(0), -1)

        # 全连接层和dropout
        output = self.fc_out(result)

        return output


# 符合原论文的ViT架构，原论文的ViT仅包含Encoder部分
class Light_ViT(nn.Module):
    def __init__(self, N, image_size, embed_size, patch_size, output_size):
        super(Light_ViT, self).__init__()

        self.encoder = ViT_Encoder(N, image_size, patch_size, embed_size)
        num_patches = (image_size//patch_size) ** 2
        num_features = 3 * (image_size**2)//num_patches

        self.fc_out = nn.Linear(embed_size*num_patches, output_size)

    def forward(self, x_encoder):
        # 首先计算encoder的输出
        x_enc_out = self.encoder(x_encoder)

        # flatten最后两个维度
        x_reshape = x_enc_out.reshape(x_enc_out.size(0), -1)

        # 通过全连接层
        x_output = self.fc_out(x_reshape)

        return x_output

