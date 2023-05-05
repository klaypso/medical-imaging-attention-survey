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

# Feature extractor (for Transformers)
feature_extractor = None


# Train Transforms
train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(IMG_SIZE if resize_opt == 'resizeshortest_randomcrop' else (IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.RandomCrop(IMG_SIZE if resize_opt == 'resizeshortest_randomcrop' else (IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.05, 0.1), scale=(0.95, 1.05), shear=0, resample=0, fillcolor=(0, 0, 0)),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=feature_extractor.image_mean if feature_extractor else MEAN, std=feature_extractor.image_std if feature_extractor else STD)
])


# Validation Transforms
val_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(IMG_SIZE if resize_opt == 'resizeshortest_randomcrop' else (IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.RandomCrop(IMG_SIZE if resize_opt == 'resizeshortest_randomcrop' else (IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=feature_extractor.image_mean if feature_extractor else MEAN, std=feature_extractor.image_std if feature_extractor else STD)
])


# APTOS2019
if dataset == "APTOS":
    # Datasets
    train_set = APTOSDataset(base_data_path=data_dir, split="Train", resized=True, low_data_regimen=low_data_regimen, perc_train=perc_train, transform=train_transforms)
    val_set = APTOSDataset(base_data_path=data_dir, split="Validation", resized=True, low_data_regimen=low_data_regimen, perc_train=perc_train, transform=val_transforms)


# ISIC2020
elif dataset == "ISIC2020":
    # Datasets
    train_set = ISIC2020Dataset(base_data_path=data_dir, split='Train', random_seed=random_seed, resized=True, low_data_regimen=low_data_regimen, perc_train=perc_train, transform=train_transforms)
    val_set = ISIC2020Dataset(base_data_path=data_dir, split='Validation', random_seed=random_seed, resized=True, low_data_regimen=low_data_regimen, perc_train=perc_train, transform=val_transforms)


# MIMICXR
elif dataset == "MIMICCXR":
    # Directories
    train_dir = os.path.join(data_dir, "Train_images_AP_resized")
    val_dir = os.path.join(data_dir, "Val_images_AP_resized")
    test_dir = os.path.join(data_dir, "Test_images_AP_resized")

    # Datasets
    train_set = MIMICXRDataset(base_data_path=train_dir, pickle_path=os.path.join(train_dir, "Annotations.pickle"), resized=True, low_data_regimen=low_data_regimen, perc_train=perc_train, transform=train_transforms)
    val_set = MIMICXRDataset(base_data_path=val_dir, pickle_path=os.path.join(val_dir, "Annotations.pickle"), transform=val_transforms)



# Results and Weights
weights_dir = os.path.join(outdir, "weights")
if not os.path.isdir(weights_dir):
    os.makedirs(weights_dir)


# History Files
history_dir = os.path.join(outdir, "history")
if not os.path.isdir(history_dir):
    os.makedirs(history_dir)


# Tensorboard
tbwritter = SummaryWriter(log_dir=os.path.join(outdir, "tensorboard"), flush_secs=30)


# Choose GPU
DEVICE = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# Number of classes for models
nr_classes = train_set.nr_classes


# DenseNet121
if model == "DenseNet121":
    model = DenseNet121(channels=img_nr_channels, height=img_height, width=img_width, nr_classes=nr_classes)

# ResNet50
elif model == "ResNet50":
    model = ResNet50(channels=img_nr_channels, height=img_height, width=img_width, nr_classes=nr_classes)

# SEDenseNet121
elif model == "SEDenseNet121":
    model = SEDenseNet121(channels=img_nr_channels, height=img_height, width=img_width, nr_classes=nr_classes)

# SEResNet50
elif model == "SEResNet50":
    model = SEResNet50(channels=img_nr_channels, height=img_height, width=img_width, nr_classes=nr_classes)

# CBAMDenseNet121
elif model == "CBAMDenseNet121":
    model = CBAMDenseNet121(channels=img_nr_channels, height=img_height, width=img_width, nr_classes=nr_classes)

# CBAMResNet50
elif model == "CBAMResNet50":
    model = CBAMResNet50(channels=img_nr_channels, 