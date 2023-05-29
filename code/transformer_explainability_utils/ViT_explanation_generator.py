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
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    matrices_aug = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
                          for i in range(len(all_layer_matrices))]
    joint_attention = matrices_aug[start_layer]
    
    for i in range(start_layer+1, len(matrices_aug)):
        joint_attention = matrices_aug[i].bmm(joint_attention)
    
    
    return joint_attention



# Class: DeiT-LRP
class LRP:
    def __init__(self, model, device):
        
        # Put model into evaluation mode
        self.model = model
        self.model.eval()

        # Select device (GPU or CPU)
        self.device= device


    # Method: Generate LRP attribution
    def generate_attribution(self, input_img, index=None, method="transformer_attribution", is_ablation=False, start_layer=0):
        output = self.model(input_img)
        kwargs = {"alpha": 1}
 