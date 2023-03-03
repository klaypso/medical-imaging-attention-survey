# Imports
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import cv2

# PyTorch Imports
import torch
import torch.nn as nn

# Captum Imports
from captum.attr import DeepLift, LRP
from captum.attr._utils.custom_modules import Addition_Module
from captum.attr._utils.lrp_rules import EpsilonRule, PropagationRule

# Transformer xAI Imports
from transformer_explainability_utils.ViT_explanation_generator import LRP as DeiT_LRP
from transformer_explainability_utils.ViT_explanation_generator import Baselines as DeiT_Baselines

# Project Imports
from model_utilities_cbam import ChannelPool


# Fix Random Seeds
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(rand