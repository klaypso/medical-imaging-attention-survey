# Imports
import os
import argparse
import numpy as np
from tqdm import tqdm
import datetime
from torchinfo import summary

# Sklearn Imports
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

# PyTorch Imports
import torch
from torch.utils.data import DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter


# Fix Random Seeds
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)


# Project Imports
from model_utilities_baseline import DenseNet121, ResNet50
from model_utilities_se import SEDenseNet121, SEResNet50
from model_utilities_cbam import CBAMDenseNet121, CBAMResNet50
from data_utilities import APTOSDataset, ISIC2020Dataset, MIMICXRDataset
from transformers import DeiTFeatureExtractor
from transformer_explainability_utils.ViT_LRP import deit_tiny_patch16_224 as DeiT_Tiny


# Command Line Interface
# Create the parser
parser = argparse.ArgumentParser()

# Add the arguments
# Data directory
parser.add_argument('--data_dir', type=str, required=True, help="Directory of the data set.")

# Data set
parser.add_argument('--dataset', type=str, required=True, choices=["APTOS2019", "ISIC2020", "MIMICCXR"], help="Data set: APTOS2019, ISIC2020, MIMICCXR.")

# Model
parser.add_argument('--model', type=str, required=True, choices=["DenseNet121", "ResNet50", "SEDenseNet121", "SEResNet50", "CBAMDenseNet121", "CBAMResNet50", "DeiT-T-LRP"], help='Model Name: DenseNet121, ResNet50, SEDenseNet121, SEResNet50, CBAMDenseNet121, CBAMResNet50, DeiT-T-LRP.')

# Low Data Regimen
parser.add_argument('--low_data_regimen', action="store_true", help="Activate the low data regimen training.")
parser.add_argument('--perc_train', type=float, default=1, help="Percentage of training data to be used during training."