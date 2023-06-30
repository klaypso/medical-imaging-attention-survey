# Source: https://github.com/hila-chefer/Transformer-Explainability
# Model creation / weight loading / state_dict helpers
# Hacked together by / Copyright 2020 Ross Wightman

# Imports
import logging
import os
import math
from collections import OrderedDict
from copy import deepcopy
from typing import Callable

# PyTorch Imports
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

_logger = logging.getLogger(__name__)



# Function: load_state_dict
def load_state_dict(checkpoint_path, use_ema=False):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = 'state_dict'
        if isinstance(checkpoint, dict):
            if use_ema and 'state_dict_ema' in checkpoint:
                state_dict_key = 'state_dict_ema'
        if state_dict_key and state_dict_key in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                # strip `module.` prefix
        