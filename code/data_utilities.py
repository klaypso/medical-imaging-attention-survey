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
        if(f.endswith(".jpg") or f.endswith('.png')):
            img = Image.open(os.path.join(datapath, f))
            w, h = img.size
            ratio = w / h
            new_w = int(np.ceil(newheight * ratio))
            new_img = img.resize((new_w, newheight), Image.ANTIALIAS)
            new_img.save(os.path.join(newpath, f))


    return



# MIMIC-CXR: Dataset Class
class MIMICXRDataset(Dataset):
    def __init__(self, base_data_path, pickle_path, random_seed=42, resized=None, low_data_regimen=None, perc_train=None, transform=None):
        """
        Args:
            base_data_path (string): Data directory.
            pickle_path (string): Path for pickle with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        # Init variables
        images_paths, images_labels, nr_classes = self.mimic_map_images_and_labels(base_data_path, pickle_path)

        # Activate low data regimen training
        if low_data_regimen:
            assert perc_train > 0.0 and perc_train <= 0.50, f"Invalid perc_train '{perc_train}'. Please be sure that perc_train > 0 and perc_train <= 50"


            # Get the data percentage
            images_paths, _, images_labels, _ = train_test_split(images_paths, images_labels, train_size=perc_train, stratify=images_labels, random_state=random_seed)

            print(f"Low data regimen.\n% of train data: {perc_train}")


        # Attribute variables
        self.images_paths = images_paths
        self.images_labels = images_labels
        self.nr_classes = nr_classes
        self.transform = transform


        return


    # MIMIC-CXR: Get labels and paths from pickle
    def mimic_map_images_and_labels(self, base_data_path, pickle_path):
        # Open pickle file
        with open(pickle_path, "rb") as fp:
            pickle_data = cPickle.load(fp)

        # Split Images and Labels
        images_path = list()
        labels = list()

        # Go through pickle file
        for path, clf in zip(pickle_data[:, 0], pickle_data[:, 1]):
            images_path.append(os.path.join(base_data_path, path+".jpg"))
            labels.append(int(clf))
        

        # Assign variables to class variables
        images_paths = images_path
        images_labels = labels
        
        # Nr of Classes
        nr_classes = len(np.unique(images_labels))


        return images_paths, images_labels, nr_classes


    # Method: __len__
    def __len__(self):
        return len(self.images_paths)



    # Method: __getitem__
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        

        # Get images
        img_name = self.images_paths[idx]
        image = Image.open(img_name)

        # Get labels
        label = self.images_labels[idx]

        # Apply transformation
        if self.transform:
            image = self.transform(image)

        return image, label



# ISIC2020: Dataset Class
class ISIC2020Dataset(Dataset):
    def __init__(self, base_data_path, split, random_seed=42, resized=None, low_data_regimen=None, perc_train=None, transform=None):
        """
        Args:
            base_data_path (string): Data directory.
            split (string): "train", "val", "