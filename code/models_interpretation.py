# Imports
import os
import argparse
from collections import OrderedDict
import numpy as np

# PyTorch Imports
import torch
from torch.utils.data import DataLoader
import torchvision

# Project Imports
from data_utilities import APTOSDataset, ISIC2020Dataset, MIMICXRDataset
from model_utilities_baseline import DenseNet121, ResNet50
from model_utilities_cbam import CBAMDenseNet121, CBAMResNet50
from model_utilities_xai import generate_post_hoc_xmap, gen_transformer_att
from model_utilities_se import SEDenseNet121, SEResNet50
from transformers import DeiTFeatureExtractor
from transformer_explainability_utils.ViT_LRP import deit_tiny_patch16_224 as DeiT_Tiny 



# Fix Random Seeds
random_seed = 42
torch.manual_seed(random_seed)
