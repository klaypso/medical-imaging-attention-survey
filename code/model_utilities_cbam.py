
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



# ResNet Functions and Classes
# Helper Function (from: https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html#resnet18)
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> torch.nn.Conv2d:
    
    """3x3 convolution with padding"""

    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)



# ResNet BasicBlock w/ changes from https://github.com/Jongchan/attention-module
class CBAMBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=True):
        super(CBAMBasicBlock, self).__init__()
        
        self.conv1 = conv3x3(in_planes=inplanes, out_planes=planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(in_planes=planes, out_planes=planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride


        if use_cbam:
            self.cbam = CBAM(planes, 16)
        else:
            self.cbam = None
        

        # For further use
        self.use_cbam = use_cbam


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)


        if self.use_cbam:
            out = self.cbam(out)

        out += residual
        out = self.relu2(out)


        return out



# ResNet Bottleneck Module w/changes from https://github.com/Jongchan/attention-module
class CBAMBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=True):
        super(CBAMBottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)
        self.relu3 = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

        if use_cbam:
            self.cbam = CBAM(planes * 4, 16)
        else:
            self.cbam = None

        # For further use
        self.use_cbam = use_cbam


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        

        if self.use_cbam:
            out = self.cbam(out)


        out += residual
        out = self.relu3(out)


        return out



# ResNet Module w/ changes from https://github.com/Jongchan/attention-module
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNet, self).__init__()


        # We use the network_type == "ImageNet" model config
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(7)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)


        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        nn.init.kaiming_normal_(self.fc.weight)
        
        for key in self.state_dict():
            if key.split('.')[-1]=="weight":
                if "conv" in key:
                    nn.init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if "bn" in key:
                    if "SpatialGate" in key:
                        self.state_dict()[key][...] = 0
                    else:
                        self.state_dict()[key][...] = 1
            
            elif key.split(".")[-1]=='bias':
                self.state_dict()[key][...] = 0


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )


        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_cbam=True))
        self.inplanes = planes * block.expansion


        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_cbam=True))


        return nn.Sequential(*layers)


    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)


        x = self.maxpool(x)


        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)


        x = self.avgpool(x)


        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        
        return x



# Model: CBAMResNet50 (adapted from: https://github.com/Jongchan/attention-module)
class CBAMResNet50(torch.nn.Module):
    def __init__(self, channels, height, width, nr_classes):
        super(CBAMResNet50, self).__init__()

        # Init variables
        self.channels = channels
        self.height = height
        self.width = width
        self.nr_classes = nr_classes


        # Init modules
        # Get model
        model = ResNet(CBAMBottleneck, [3, 4, 6, 3], 1000)


        # Create our new models
        # We start by converting this model into a new one withouht the FC layers
        self.cbam_resnet50 = torch.nn.Sequential(*(list(model.children())[:-1]))

        # FC-Layers
        # Compute in_features
        _in_features = torch.rand(1, self.channels, self.height, self.width)
        aux_model = torch.nn.Sequential(*(list(model.children())[:-1]))
        aux_model.eval()
        _in_features = aux_model(_in_features)
        _in_features = _in_features.size(0) * _in_features.size(1) * _in_features.size(2) * _in_features.size(3)

        # Create FC1 Layer for classification
        self.fc1 = torch.nn.Linear(in_features=_in_features, out_features=self.nr_classes)


        return
    

    def forward(self, inputs):
        # Compute CBAM features
        features = self.cbam_resnet50(inputs)

        # Reshape features
        features = torch.reshape(features, (features.size(0), -1))

        # FC1-Layer
        outputs = self.fc1(features)


        return outputs



# VGG-16 Functions and Classes
# Helper Function: Create VGG layers with CBAM Layer Blocks (adapted from: https://pytorch.org/vision/stable/_modules/torchvision/models/vgg.html#vgg16)
def make_layers_cbam(cfg: List[Union[str, int]], batch_norm: bool = False) -> torch.nn.Sequential:
    layers: List[torch.nn.Module] = []
    in_channels = 3
    
    for v in cfg:
        if v == 'M':
            # layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]

            # Replace a MaxPool2d with an Attention Layer
            layers += [CBAM(in_channels)]
        
        else:
            v = cast(int, v)
            conv2d = torch.nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            
            if batch_norm:
                layers += [conv2d, torch.nn.BatchNorm2d(v), torch.nn.ReLU(inplace=False)]
            
            else:
                layers += [conv2d, torch.nn.ReLU(inplace=False)]
                # layers += [conv2d, se_layer]
            
            in_channels = v

    
    return torch.nn.Sequential(*layers)



# Model: CBAMVGG-16 (adapted from: https://pytorch.org/vision/stable/_modules/torchvision/models/vgg.html#vgg16)
class CBAMVGG16(torch.nn.Module):
    def __init__(self, channels, height, width, nr_classes, pretrained=False):
        super(CBAMVGG16, self).__init__()

        # Init variables
        self.channels = channels
        self.height = height
        self.width = width
        self.nr_classes = nr_classes
        self.weights_url = 'https://download.pytorch.org/models/vgg16-397923af.pth'
        self.cfgs: Dict[str, List[Union[str, int]]] = {
            'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],