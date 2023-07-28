# Source: https://github.com/hila-chefer/Transformer-Explainability

# PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['forward_hook', 'Clone', 'Add', 'Cat', 'ReLU', 'GELU', 'Dropout', 'BatchNorm2d', 'Linear', 'MaxPool2d',
           'AdaptiveAvgPool2d', 'AvgPool2d', 'Conv2d', 'Sequential', 'safe_divide', 'einsum', 'Softmax', 'IndexSelect',
           'LayerNorm', 'AddEye']


def safe_divide(a, b):
    den = b.clamp(min=1e-9) + b.clamp(max=1e-9)
    den = den + den.eq(0).type(den.type()) * 1e-9
    return a / den * b.ne(0).type(b.type())


def forward_hook(self, input, output):
    