# Imports
import os
import _pickle as cPickle
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image

# Sklearn Imports
from sklearn.model_selection import train_test_split

# PyTorch Imports
import torch
from torch.utils.data import Dataset



# General
# Function: Resize images
def resize_images(datapath, newpath, newheight=512):
    
    # Create new directories (if necessary)
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    

    # Go through data directory and generate new (resized) images
    for f in tqdm(os.listdir(datapath)):
        if(f.endswith(".jpg") or f