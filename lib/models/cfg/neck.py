from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging
import numpy as np
from os.path import join

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from lib.models.DCNv2.dcn_v2 import DCN

import matplotlib.pyplot as plt
BN_MOMENTUM = 0.1

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 自适应最大池化

        # 两个卷积层用于从池化后的特征中学习注意力权重
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)  # 第一个卷积层，降维
        self.relu1 = nn.ReLU()  # ReLU激活函数
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)  # 第二个卷积层，升维
        self.sigmoid = nn.Sigmoid()  # Sigmoid函数生成最终的注意力权重

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))  # 对平均池化的特征进行处理
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))  # 对最大池化的特征进行处理
        out = avg_out + max_out  # 将两种池化的特征加权和作为输出
        return self.sigmoid(out)  # 使用sigmoid激活函数计算注意力权重


# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'  # 核心大小只能是3或7
        padding = 3 if kernel_size == 7 else 1  # 根据核心大小设置填充

        # 卷积层用于从连接的平均池化和最大池化特征图中学习空间注意力权重
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()  # Sigmoid函数生成最终的注意力权重

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 对输入特征图执行平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 对输入特征图执行最大池化
        x = torch.cat([avg_out, max_out], dim=1)  # 将两种池化的特征图连接起来
        x = self.conv1(x)  # 通过卷积层处理连接后的特征图
        return self.sigmoid(x)  # 使用sigmoid激活函数计算注意力权重
    
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)  # 通道注意力实例
        self.sa = SpatialAttention(kernel_size)  # 空间注意力实例

    def forward(self, x):
        out = x * self.ca(x)  # 使用通道注意力加权输入特征图
        result = out * self.sa(out)  # 使用空间注意力进一步加权特征图
        return result  # 返回最终的特征图

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.conv = DCN(chi, cho, kernel_size=(3,3), stride=1, padding=1, dilation=1, deformable_groups=1)
    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x
class Upsample(nn.Module):
    def __init__(self,in_channel,up_f,o):
        super().__init__()
        self.proj = DeformConv(in_channel,o)
        self.node = DeformConv(o,o)
        self.up_f = up_f
        f = up_f
        self.up = nn.ConvTranspose2d(o, o, f * 2, stride=f, 
                                    padding=f // 2, output_padding=0,
                                    groups=o, bias=False)
    def forward(self,x):
        x = self.proj(x)
        x = self.up(x)
        x = self.node(x)
        return x

class GatedFusionConv(nn.Module):
    def __init__(self, c_rgb, c_flow, c_fused):
        super(GatedFusionConv, self).__init__()
        self.rgb_conv = nn.Conv2d(c_rgb, c_fused, kernel_size=1)
        self.flow_conv = nn.Conv2d(c_flow, c_fused, kernel_size=1)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_fused * 2, c_fused // 4, 1),
            nn.ReLU(),
            nn.Conv2d(c_fused // 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, f_rgb, f_flow):
        f_rgb = self.rgb_conv(f_rgb)  # B x C x H x W
        f_flow = self.flow_conv(f_flow)
        fusion = torch.cat([f_rgb, f_flow], dim=1)
        alpha = self.gate(fusion)  # B x 1 x 1 x 1
        fused = alpha * f_rgb + (1 - alpha) * f_flow
        return fused

class AddFusion2D(nn.Module):
    def __init__(self, o,  channels, up_f=2):
        super(AddFusion2D, self).__init__()
    
        for i in range(1, len(channels)):
            # print(len(channels))

            c = channels[i]
            o = channels[i-1]
            proj = DeformConv(c, o)
            node = DeformConv(o, o)
            f = up_f
            up = nn.ConvTranspose2d(o, o, f * 2, stride=f, 
                                    padding=f // 2, output_padding=0,
                                    groups=4, bias=False)

            cbam = CBAM(c)
            fill_up_weights(up)
            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)
            setattr(self, 'cbam_' + str(i), cbam)
        # 添加第0层的CBAM
        cbam = CBAM(channels[0])
        setattr(self, 'cbam_' + "0", cbam)

    def forward(self, layers, startp, endp):
        #去掉 cbam or 用dla_up
        for i in range(endp-1, startp,-1):
            
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            
            node = getattr(self, 'node_' + str(i - startp))
            
            layers[i-1] = node(layers[i] + layers[i-1])
            cbam =  getattr(self, 'cbam_' + str(i - startp-1))
            layers[i-1] = cbam(layers[i-1])
        
        return layers[0]

class ConcatFusion(nn.Module):
    def __init__(self, o,channels, up_f=2):
        super(ConcatFusion, self).__init__()

        self.out_conv = nn.Conv2d(sum(channels), o,(1,1))
        for i in range(1, len(channels)):
            # print(len(channels))
            c = sum(channels[i:])
            f = int(up_f)
            proj = nn.Conv2d(c, c,(1,1))
            node = nn.Conv2d(c, c,(1,1))
     
            up = nn.ConvTranspose2d(c, c, f * 2, stride=f, 
                                    padding=f // 2, output_padding=0,
                                    groups=o, bias=False)
            cbam = CBAM(c)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)
            setattr(self, 'cbam_' + str(i), cbam)

    def forward(self, layers, startp, endp):
        
        for i in range(endp-1, startp, -1):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            node = getattr(self, 'node_' + str(i - startp))
            cbam =  getattr(self, 'cbam_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))

            
            layers[i] = node(layers[i])
           
            layers[i-1] = torch.cat((layers[i-1], layers[i]), 1)
            layers[i-1] = cbam(layers[i-1])

        return self.out_conv(layers[0])
    
class AddFusion3D(nn.Module):
    def __init__(self,o, channels, up_f=2):
        super(AddFusion3D, self).__init__()
       

        for i in range(1, len(channels)):
            # print(len(channels))
        
            c = channels[i]
            o = channels[i-1]
            proj = nn.Conv3d(c, o,(1,1,1))
            node = nn.Conv3d(o, o,(1,1,1))
            f = up_f
            up = nn.ConvTranspose3d(o, o, (1,f*2,f * 2) ,stride=(1,f,f), 
                                    padding=(0,f//2,f // 2), output_padding=0,
                                    groups=o, bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)
            


    def forward(self, layers, startp, endp):
        
        for i in range(endp-1, startp,-1):
            
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            
            node = getattr(self, 'node_' + str(i - startp))
           
            layers[i-1] = node(layers[i] + layers[i-1])
        
        return layers[0]

class CS_CrossAttention(nn.Module):
    def __init__(self, channel):
        super(CS_CrossAttention, self).__init__()
        # 空间注意力分支
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )

        # 通道注意力分支（SE结构）
        self.channel_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // 8, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // 8, channel, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # Cross Attention 融合（使用 1x1卷积 + 简化QK注意机制）
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(channel*2, channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )

    def forward(self, x):

        # ——空间注意力通路——
        max_pool_spatial, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool_spatial = torch.mean(x, dim=1, keepdim=True)
        spatial_attn_input = torch.cat([max_pool_spatial, avg_pool_spatial], dim=1)
        spatial_weight = self.spatial_conv(spatial_attn_input)  # (B, 1, H, W)
        spatial_weight = torch.sigmoid(spatial_weight)

        # ——通道注意力通路——
        channel_weight = self.channel_mlp(x)  # (B, C, 1, 1)

        # ——Cross Attention 融合——
        # 使用空间注意力图扩展通道维度，与原始 x 一起拼接后进行融合
        spatial_up = spatial_weight.expand_as(x)  # (B, C, H, W)
        fusion_input = torch.cat([x * spatial_up, x * channel_weight], dim=1)  # (B, 2C, H, W)
        fusion_output = self.fuse_conv(fusion_input)  # (B, C, H, W)

        return fusion_output

