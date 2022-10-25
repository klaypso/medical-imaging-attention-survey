# Imports
from typing import Type, Any, Callable, Union, List, Optional

# PyTorch Imports
import torch
from torch import Tensor
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import torchvision



# Create PyTorch Models
# Model: DenseNet 121 (Baseline)
class DenseNet121(torch.nn.Module):
    def __init__(self, channels, height, width, nr_classes):
        super(DenseNet121, self).__init__()

        # Init variables
        self.channels = channels
        self.height = height
        self.width = width
        self.nr_classes = nr_classes


        # Init modules
        # Backbone to extract features
        self.densenet121 = torchvision.models.densenet121(pretrained=True).features

        # FC-Layers
        # Compute in_features
        _in_features = torch.rand(1, self.channels, self.height, self.width)
        _in_features = self.densenet121(_in_features)
        _in_features = _in_features.size(0) * _in_features.size(1) * _in_features.size(2) * _in_features.size(3)

        # Create FC1 Layer for classification
        self.fc1 = torch.nn.Linear(in_features=_in_features, out_features=self.nr_classes)


        return
    

    def forward(self, inputs):
        # Compute Backbone features
        features = self.densenet121(inputs)

        # Reshape features
        features = torch.reshape(features, (features.size(0), -1))

        # FC1-Layer
        outputs = self.fc1(features)


        return outputs



# Helper Function: ResNet Conv3x3 (https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)



# Helper Function: ResNet Conv1x1 (https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)
def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



# Helper Class: ResNet BasicBlock (https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)
class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None, groups: int = 1, base_width: int = 64, dilation: int = 1, norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        
        
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        
        
        # Both self.conv1 and self.downsample layers downsample the input when stride !