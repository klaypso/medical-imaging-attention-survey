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
    