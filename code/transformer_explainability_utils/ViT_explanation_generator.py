# Source: https://github.com/hila-chefer/Transformer-Explainability

# Imports
import numpy as np
from numpy import *

# PyTorch Imports
import torch



# Function: Compute rollout between attention layers
def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration- code adapted from https://github.com/samiraabnar/attention_flow
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens,