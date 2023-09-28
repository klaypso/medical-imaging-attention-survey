# Source: https://github.com/hila-chefer/Transformer-Explainability

# Imports
import math
import warnings

# PyTorch Imports
import torch



# Function
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumul