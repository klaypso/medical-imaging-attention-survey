
# Source: https://github.com/hila-chefer/Transformer-Explainability
# Vision Transformer (ViT) in PyTorch Hacked together by / Copyright 2020 Ross Wightman

# PyTorch Imports
import torch
import torch.nn as nn
from einops import rearrange

# Project Imports
from .layers_ours import *
from .helpers import load_pretrained
from .weight_init import trunc_normal_
from .layer_helpers import to_2tuple



# Function: Generate configuration for models
def _cfg(url='', num_classes=1000, input_size=(3, 224, 224), pool_size=None, crop_pct=.9, interpolation='bicubic', first_conv='patch_embed.proj', classifier='head', **kwargs):
    
    
    # return {
    #     'url': url,
    #     'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
    #     'crop_pct': .9, 'interpolation': 'bicubic',
    #     'first_conv': 'patch_embed.proj', 'classifier': 'head',
    #     **kwargs
    # }

    cfg_dict = {
        'url': url,
        'num_classes': num_classes,
        'input_size': input_size,
        'pool_size': pool_size,
        'crop_pct': crop_pct,
        'interpolation': interpolation,
        'first_conv': first_conv,
        'classifier': classifier,
        **kwargs
    }


    return cfg_dict



# Function: Default configurations for models
def get_default_cfgs():

    default_cfgs = {
        # patch models
        'vit_small_patch16_224': _cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',),
        'vit_base_patch16_224': _cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth', mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),),
        'vit_large_patch16_224': _cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth', mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    }

    return default_cfgs