# Source: https://github.com/hila-chefer/Transformer-Explainability
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>

# Imports
import cv2
import numpy as np
import subprocess

# PyTorch Imports
import torch
import torchvision.transforms as transforms



# Class: NormalizeInverse
class NormalizeInverse(transforms.Normalize):
    # Undo normalization on images

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7