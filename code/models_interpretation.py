# Imports
import os
import argparse
from collections import OrderedDict
import numpy as np

# PyTorch Imports
import torch
from torch.utils.data import DataLoader
import torchvision

# Project Imports
from data_utilities import APTOSDataset, ISIC2020Dataset, MIMICXRDataset
from model_utilities_baseline import DenseNet121, ResNet50
from model_utilities_cbam import CBAMDenseNet121, CBAMResNet50
from model_utilities_xai import generate_post_hoc_xmap, gen_transformer_att
from model_utilities_se import SEDenseNet121, SEResNet50
from transformers import DeiTFeatureExtractor
from transformer_explainability_utils.ViT_LRP import deit_tiny_patch16_224 as DeiT_Tiny 



# Fix Random Seeds
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)



# Command Line Interface
# Create the parser
parser = argparse.ArgumentParser()

# Add the arguments
# Data dir
parser.add_argument('--data_dir', type=str, default="data", help="Main data directory (e.g., 'data/')")

# Data set
parser.add_argument('--dataset', type=str, required=True, choices=["APTOS", "ISIC2020", "MIMICCXR"], help="Data set: APTOS, ISIC2020, MIMICCXR.")

# Data split
parser.add_argument('--split', type=str, required=True, choices=["Train", "Validation", "Test"], help="Data split: Train, Validation or Test")

# Model
parser.add_argument('--model', type=str, required=True, choices=["DenseNet121", "ResNet50", "SEDenseNet121", "SEResNet50", "CBAMDenseNet121", "CBAMResNet50", "DeiT-T-LRP"], help='Model Name: DenseNet121, ResNet50, SEDenseNet121, SEResNet50, CBAMDenseNet121, CBAMResNet50, DeiT-T-LRP.')

# Model checkpoint
parser.add_argument("--modelckpt", type=str, required=True, help="Directory where model is stored")

# Batch size
parser.add_argument('--batchsize', type=int, default=4, help="Batch-size for training and validation")

# Image size
parser.add_argument('--imgsize', type=int, default=224, help="Size of the image after transforms")

# Resize
parser.add_argument('--resize', type=str, choices=["direct_resize", "resizeshortest_randomcrop"], default="direct_resize", help="Resize data transformation")

# Number of workers
parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for dataloader")

# GPU ID
parser.add_argument("--gpu_id", type=int, default=0, help="The index of the GPU")

# Number of layers (ViT)
parser.add_argument("--nr_layers", type=int, default=12, help="Number of hidden layers (only for ViT)")


# Parse the arguments
args = parser.parse_args()


# Data directory
data_dir = args.data_dir

# Dataset
dataset = args.dataset

# Data split
data_split = args.split

# Model Directory
modelckpt = args.modelckpt

# Number of workers (threads)
workers = args.num_workers

# GPU ID
gpu_id = args.gpu_id

# Batch size
BATCH_SIZE = args.batchsize

# Image size (after transforms)
IMG_SIZE = args.imgsize

# Number of layers of the Visual Transformer
nr_layers = args.nr_layers

# Resize (data transforms)
resize_opt = args.resize



# Load data
# Mean and STD to Normalize the inputs into pretrained models
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# Input Data Dimensions
img_nr_channels = 3
img_height = IMG_SIZE
img_width = IMG_SIZE


# Get the right model from the CLI
model = args.model
model_name = model.lower()
feature_extractor = None


# Evaluation Transforms
eval_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(IMG_SIZE if resize_opt == 'resizeshortest_randomcrop' else (IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.RandomCrop(IMG_SIZE if resize_opt == 'resizeshortest_randomcrop' else (IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=feature_extractor.image_mean if feature_extractor else MEAN, std=feature_extractor.image_std if feature_extractor else STD)
])



# APTOS2019
if dataset == "APTOS":
    # Directories
    dataset_dir = os.path.join(data_dir, "APTOS2019")

    # Evaluation set
    eval_set = APTOSDataset(base_data_path=dataset_dir, split=data_split, transform=eval_transforms)

    # Weights directory
    weights_dir = os.path.join(modelckpt, "weights")

    # Post-hoc explanations
    xai_maps_dir = os.path.join(modelckpt, "xai_maps")
    if not(os.path.isdir(xai_maps_dir)):
        os.makedirs(xai_maps_dir)


# ISIC2020
elif dataset == "ISIC2020":
    # Directories
    dataset_dir = os.path.join(data_dir, "ISIC2020/jpeg/train")

    # Evaluation set
    eval_set = ISIC2020Dataset(base_data_path=dataset_dir, split=data_split, random_seed=random_seed, transform=eval_transforms)

    # Weights directory
    weights_dir = os.path.join(modelckpt, "weights")

    # Post-hoc explanations
    xai_maps_dir = os.path.join(modelckpt, "xai_maps")
    if not(os.path.isdir(xai_maps_dir)):
        os.makedirs(xai_maps_dir)


# MIMICXR
elif dataset == "MIMICCXR":
    # Directories
    dataset_dir = os.path.join(data_dir, "MedIA")

    if data_split == "Train":    
        eval_dir = os.path.join(dataset_dir, "Train_images_AP_resized")
    
    elif data_split == "Validation":
        eval_dir = os.path.join(dataset_dir, "Val_images_AP_resized")
    
    elif data_split == "Test":
        eval_dir = os.path.join(dataset_dir, "Test_images_AP_resized")
    

    # Evaluation set
    eval_set = MIMICXRDataset(base_data_path=eval_dir, pickle_path=os.path.join(eval_dir, "Annotations.pickle"), transform=eval_transforms)


    # Weights directory
    weights_dir = os.path.join(modelckpt, "weights")

    # Post-hoc explanations
    xai_maps_dir = os.path.join(modelckpt, "xai_maps")
    if not(os.path.isdir(xai_maps_dir)):
        os.makedirs(xai_maps_dir)



# Choose GPU
DEVICE = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"


# Dataloaders
eval_loader = DataLoader(dataset=eval_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=workers)

# Get number of classes
nr_classes = eval_set.nr_classes


# DenseNet12