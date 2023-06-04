# Source: https://github.com/hila-chefer/Transformer-Explainability
# Vision Transformer (ViT) in PyTorch
# Hacked together by / Copyright 2020 Ross Wightman

# PyTorch Imports
import torch
import torch.nn as nn
from einops import rearrange

# Project Imports
from layers_lrp import *
from helpers import load_pretrained
from weight_init import trunc_normal_
from layer_helpers import to_2tuple



# Function: Get configuration
def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
       