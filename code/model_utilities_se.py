# Imports
from collections import OrderedDict
import re
from typing import Union, List, Dict, cast, Tuple

# PyTorch Imports
import torch
from torch.hub import load_state_dict_from_url
from torchvision.models import ResNet, VGG
import torch.utils.checkpoint as cp



# ResNet-50 Functions and Classes
# Helper Function (from: https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html#resnet18)
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> torch.nn.Conv2d:
    
    """3x3 convolution with padding"""

    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


# Helper Function (from: https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html#resnet18)
def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> torch.nn.Conv2d:
    
    ""