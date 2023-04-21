
# Imports
import os
import argparse
from tqdm import tqdm
import numpy as np
from collections import OrderedDict

# Sklearn Import
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# PyTorch Imports
import torch
from torch.utils.data import DataLoader
import torchvision


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
parser.add_argument('--dataset', type=str, required=True, choices=["ISIC2020", "MIMICCXR", "APTOS"], help="Data set: ISIC2020, MIMICCXR, APTOS")

# Data split
parser.add_argument('--split', type=str, required=True, choices=["Train", "Validation", "Test"], help="Data split: Train, Validation or Test")

# Model
parser.add_argument('--model', type=str, required=True, choices=["DenseNet121", "ResNet50", "SEDenseNet121", "SEResNet50", "CBAMDenseNet121", "CBAMResNet50", "DeiT-T-LRP"], help='Model Name: DenseNet121, ResNet50, SEDenseNet121, SEResNet50, CBAMDenseNet121, CBAMResNet50, DeiT-T-LRP.')

# Low Data Regimen
parser.add_argument('--low_data_regimen', action="store_true", help="Activate the low data regimen training.")
parser.add_argument('--perc_train', type=float, default=1, help="Percentage of training data to be used during training.")

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

# Batch size
BATCH_SIZE = args.batchsize

# Image size (after transforms)
IMG_SIZE = args.imgsize

# Number of layers of the Visual Transformer
nr_layers = args.nr_layers

# Resize (data transforms)
resize_opt = args.resize

# Low data regimen
low_data_regimen = args.low_data_regimen
perc_train = args.perc_train

# Weights directory
weights_dir = os.path.join(modelckpt, "weights")


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


# APTOS
if dataset == "APTOS":
    # Create evaluation dataset
    eval_set = APTOSDataset(base_data_path=data_dir, split=data_split, resized=True, low_data_regimen=low_data_regimen, perc_train=perc_train, transform=eval_transforms)


# ISIC2020
elif dataset == "ISIC2020":
    # Create evaluation dataset
    eval_set = ISIC2020Dataset(base_data_path=data_dir, split=data_split, random_seed=random_seed, resized=None, low_data_regimen=low_data_regimen, perc_train=perc_train, transform=eval_transforms)


# MIMICXR
elif dataset == "MIMICCXR":
    # Get data splits
    if data_split == "Train":
        eval_dir = os.path.join(data_dir, "Train_images_AP_resized")
    
    elif data_split == "Validation":
        eval_dir = os.path.join(data_dir, "Val_images_AP_resized")
    
    elif data_split == "Test":
        eval_dir = os.path.join(data_dir, "Test_images_AP_resized")


    # Create evaluation dataset
    if data_split == "Train":
        eval_set = MIMICXRDataset(base_data_path=eval_dir, pickle_path=os.path.join(eval_dir, "Annotations.pickle"), resized=None, low_data_regimen=low_data_regimen, perc_train=perc_train, transform=eval_transforms)
    else:
        eval_set = MIMICXRDataset(base_data_path=eval_dir, pickle_path=os.path.join(eval_dir, "Annotations.pickle"), transform=eval_transforms)



# Get number of classes for model
nr_classes = eval_set.nr_classes

# Create DataLoader
eval_loader = DataLoader(dataset=eval_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=workers)


# Choose GPU
DEVICE = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")



# DenseNet-121
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
    model = CBAMResNet50(channels=img_nr_channels, height=img_height, width=img_width, nr_classes=nr_classes)

# DeiT-Tiny (compatible with LRP)
elif model == "DeiT-T-LRP":
    model = DeiT_Tiny(pretrained=True, num_classes=nr_classes, input_size=(3, IMG_SIZE, IMG_SIZE), url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth")
    feature_extractor = DeiTFeatureExtractor.from_pretrained("facebook/deit-tiny-patch16-224")



# Load model weights
model_file = os.path.join(weights_dir, f"{model_name}_{dataset.lower()}_best.pt")
checkpoint = torch.load(model_file, map_location=DEVICE)



# We need to add an exception to prevent some errors from the attention mechanisms that were already trained
# Case without any error
try:
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    print("Loaded model from " + model_file)

# Case related to CBAM blocks
except:

    # Debug print
    print("Fixing key values with old trained CBAM models")

    # Get missing keys
    missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=False) 

    if len(missing) == len(unexpected):
        
        # Method to remap the new state_dict keys (https://gist.github.com/the-bass/0bf8aaa302f9ba0d26798b11e4dd73e3)
        state_dict = checkpoint['model_state_dict']