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
            split (string): "train", "val", "test" splits.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        
        # Assure we have the right string in the split argument
        assert split in ["Train", "Validation", "Test"], "Please provide a valid split (i.e., 'Train', 'Validation' or 'Test')"


        # Get correct paths
        data_dir, csv_path, nr_classes = self.isic_get_data_paths(base_data_path=base_data_path, resized=resized)

        # Aux variables to obtain the correct data splits
        # Read CSV file with label information       
        csv_df = pd.read_csv(csv_path)
        # print(f"The dataframe has: {len(csv_df)} records.")
        
        # Get the IDs of the Patients
        patient_ids = csv_df.copy()["patient_id"]
        
        # Get the unique patient ids
        unique_patient_ids = np.unique(patient_ids.values)


        # Split into train, validation and test according to the IDs of the Patients
        # First we split into train and test (60%, 20%, 20%)
        train_ids, test_ids, _, _ = train_test_split(unique_patient_ids, np.zeros_like(unique_patient_ids), test_size=0.20, random_state=random_seed)
        train_ids, val_ids, _, _ = train_test_split(train_ids, np.zeros_like(train_ids), test_size=0.25, random_state=random_seed)


        # Now, we get the data
        if split == "Train":
            # Get the right sampled dataframe
            tr_pids_mask = csv_df.copy().patient_id.isin(train_ids)
            self.dataframe = csv_df.copy()[tr_pids_mask]
            
            # Get the image names
            image_names = self.dataframe.copy()["image_name"].values

            # Get the image labels
            images_labels = self.dataframe.copy()["target"].values


            # Activate low data regimen training
            if low_data_regimen:
                assert perc_train > 0.0 and perc_train <= 0.50, f"Invalid perc_train '{perc_train}'. Please be sure that perc_train > 0 and perc_train <= 50"


                # Get the data percentage
                image_names, _, images_labels, _ = train_test_split(image_names, images_labels, train_size=perc_train, stratify=images_l