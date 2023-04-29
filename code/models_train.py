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
parser.add_argument('--perc_train', type=float, default=1, help="Percentage of training data to be used during training.")

# Batch size
parser.add_argument('--batchsize', type=int, default=4, help="Batch-size for training and validation")

# Image size
parser.add_argument('--imgsize', type=int, default=224, help="Size of the image after transforms")

# Resize
parser.add_argument('--resize', type=str, choices=["direct_resize", "resizeshortest_randomcrop"], default="direct_resize", help="Resize data transformation")

# Class Weights
parser.add_argument("--classweights", action="store_true", help="Weight loss with class imbalance")

# Number of epochs
parser.add_argument('--epochs', type=int, default=300, help="Number of training epochs")

# Learning rate
parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")

# Output directory
parser.add_argument("--outdir", type=str, default="results", help="Output directory")

# Number of workers
parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for dataloader")

# GPU ID
parser.add_argument("--gpu_id", type=int, default=0, help="The index of the GPU")

# Save frequency
parser.add_argument("--save_freq", type=int, default=10, help="Frequency (in number of epochs) to save the model")

# Resume training
parser.add_argument("--resume", action="store_true", help="Resume training")
parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint from which to resume training")

# Number of layers (ViT)
parser.add_argument("--nr_layers", type=int, default=12, help="Number of hidden layers (only for ViT)")


# Parse the arguments
args = parser.parse_args()


# Resume training
if args.resume:
    assert args.ckpt is not None, "Please specify the model checkpoint when resume is True"

resume = args.resume

# Training checkpoint
ckpt = args.ckpt


# Data directory
data_dir = args.data_dir

# Dataset
dataset = args.dataset

# Results Directory
outdir = args.outdir

# Number of workers (threads)
workers = args.num_workers

# Number of training epochs
EPOCHS = args.epochs

# Learning rate
LEARNING_RATE = args.lr

# Batch size
BATCH_SIZE = args.batchsize

# Image size (after transforms)
IMG_SIZE = args.imgsize

# Save frquency
save_freq = args.save_freq

# Number of layers of the Visual Transformer
nr_layers = args.nr_layers

# Resize (data transforms)
resize_opt = args.resize
model = args.model
model_name = model.lower()

# Low data regimen
low_data_regimen = args.low_data_regimen
perc_train = args.perc_train



# Timestamp (to save results)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
outdir = os.path.join(outdir, dataset.lower(), model_name, timestamp)
if not os.path.isdir(outdir):
    os.makedirs(outdir)


# Save training parameters
with open(os.path.join(outdir, "train_params.txt"), "w") as f:
    f.write(str(args))



# Load data
# Mean and STD to Normalize the inputs into pretrained models
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# Input Data Dimensions
img_nr_channels = 3
img_height = IMG_SIZE
img_width = IMG_SIZE

# Feature