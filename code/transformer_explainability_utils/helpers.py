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

_logger = logging.getLo