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
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super(NormalizeInverse, self).__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super(NormalizeInverse, self).__call__(tensor.clone())



# Function: Create folder
def create_folder(folder_name):
    try:
        subprocess.call(['mkdir', '-p', folder_name])
    except OSError:
        None



# Function: Save saliency map on image
def save_saliency_map(image, saliency_map, filename):
    """
    Save saliency map on image.

    Args:
        image: Tensor of size (3,H,W)
        saliency_map: Tensor of size (1,H,W)
        filename: string with complete path and file extension

    """

    image = image.data.cpu().numpy()
    saliency_map = saliency_map.data.cpu().numpy()

    saliency_map = saliency_map - saliency_map.min()
    saliency_map = saliency_map / saliency_map.max()
    saliency_map = s