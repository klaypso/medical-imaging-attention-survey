
# Imports
from collections import OrderedDict
import re
from typing import Union, List, Dict, cast, Tuple

# PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from torchvision.models import ResNet, VGG
import torch.utils.checkpoint as cp



# BasicConv Module from https://github.com/Jongchan/attention-module
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=False) if relu else None


    def forward(self, x):
        x = self.conv(x)
        
        if self.bn is not None:
            x = self.bn(x)
        
        if self.relu is not None:
            x = self.relu(x)
        
        
        return x



# ChannelGate Module from https://github.com/Jongchan/attention-module
class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        
        self.gate_channels = gate_channels

        self.linear1 = nn.Linear(gate_channels, gate_channels // reduction_ratio)
        self.linear2 = nn.Linear(gate_channels // reduction_ratio, gate_channels)

        relus = dict()
        for idx, _ in enumerate(pool_types):
            relus[idx] = nn.ReLU(inplace=False)
        
        self.relus = relus


        self.pool_types = pool_types
    
    
    def forward(self, x):
        
        channel_att_sum = None
        
        for idx, pool_type in enumerate(self.pool_types):
            if pool_type=='avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = torch.reshape(avg_pool, (avg_pool.size(0), -1))
                channel_att_raw = self.linear1(channel_att_raw)
                channel_att_raw = self.relus[idx](channel_att_raw)
                channel_att_raw = self.linear2(channel_att_raw)
            
            elif pool_type=='max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = torch.reshape(max_pool, (max_pool.size(0), -1))
                channel_att_raw = self.linear1(channel_att_raw)
                channel_att_raw = self.relus[idx](channel_att_raw)
                channel_att_raw = self.linear2(channel_att_raw)

            
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))

                pass
            
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
    
                pass


            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw


        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)


        return x * scale



# logsumexp_2d from https://github.com/Jongchan/attention-module
def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    
    
    return outputs



# ChannelPool Module from https://github.com/Jongchan/attention-module
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)



# SpatialGate Module from https://github.com/Jongchan/attention-module
class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    
    
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        
        
        return x * scale



# CBAM Module from https://github.com/Jongchan/attention-module
class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    
    
    def forward(self, x):
        x_out = self.ChannelGate(x)
        
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        
        
        return x_out
