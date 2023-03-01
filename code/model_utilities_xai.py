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
from transfor