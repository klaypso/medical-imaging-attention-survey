# Medical Imaging: A survey towards better algorithms

## About
This is an implementation of the paper [_'A survey on attention mechanisms for medical applications: are we moving towards better algorithms?'_](https://ieeexplore.ieee.org/document/9889720). 

## Abstract
The paper extensively reviews the use of attention mechanisms in machine learning (including Transformers) for several medical applications, and proposes a critical analysis of the claims and potentialities of attention mechanisms presented in the literature through an experimental case study on medical image classification with three different use cases.

## Clone this repository
To clone this repository, open a Terminal window and type:
```bash
$ git clone git@github.com:klaypso/medical-imaging-attention-survey.git
```
Then go to the repository's main directory:
```bash
$ cd medical-imaging-attention-survey
```

## Dependencies
### Install the necessary Python packages
We advise you to create a virtual Python environment first (Python 3.7). To install the necessary Python packages run:
```bash
$ pip install -r requirements.txt
```

## Data
To know more about the data used in this paper, please send an e-mail to [**tiago.f.goncalves@inesctec.pt**](mailto:tiago.f.goncalves@inesctec.pt).

## Usage
### Train Models
To train the models:
```bash
$ python code/models_train.py {command line arguments}
```
This script accepts the following command line arguments:
```
--data_dir: Directory of the data set
--dataset: Data set {APTOS, ISIC2020, MIMICCXR}
--model: Model Name {DenseNet121, ResNet50, SEDenseNet121, SEResNet50, CBAMDenseNet121, CBAMResNet50, DeiT-T-LRP}
--low_data_regimen: Activate the low data regimen training
--perc_train: Percentage of training data to be used during training
--batchsize: Batch-size for training and validation
--imgsize: Size of the image after transforms
--resize: Resize data transformation {direct_resize,resizeshortest_randomcrop}
--classweights: Weight loss with class imbalance
--epochs: Number of training epochs
--lr: Learning rate
--outdir: Output directory
--num_workers: Number of workers for dataloader
--gpu_id: The index of the GPU
--save_freq: Frequency (in number of epochs) to save the model
--resume: Resume training
--ckpt: Checkpoint from which to resume training
--nr_layers: Number of hidden layers (only for ViT)
```

### Test Models
To test the models:
```bash
$ python code/models_test.py {command line arguments}
```
This script accepts the following command line arguments:
```
--data_dir: Directory of the data set
--dataset: Data set {APTOS, ISIC2020, MIMICCXR}
--split: Data split {Train,Validation,Test}
--model: Model Name {DenseNet121, ResNet50, SEDenseNet121, SEResNet50, CBAMDenseNet121, CBAMResNet50, DeiT-T-LRP}
--low_data_regimen: Activate the low data regimen training
--perc_train: Percentage of training data to be used during training
--modelckpt: Directory where model is stored
--batchsize: Batch-size for training and validation
--imgsize: Size of the image after transforms
--resize: Resize data transformation {direct_resize,resizeshortest_randomcrop}
--num_workers: Number of workers for dataloader
--gpu_id: The index of the GPU
--nr_layers: Number of hidden layers (only for ViT)
```

### Generate Post-hoc Explanations (Saliency Maps)
To generate post-hoc explanations (saliency maps):
```bash
$ python code/models_interpretation.py {command line arguments}
```
This script accepts the following command line arguments:
```
--data_dir: Directory of the data set
--dataset: Data set {APTOS, ISIC2020, MIMICCXR}
--split: Data split {Train, Validation, Test}
--model: Model Name {DenseNet121, ResNet50, SEDenseNet121, SEResNet50, CBAMDenseNet121, CBAMResNet50, DeiT-T-LRP}
--modelckpt: Directory where model is stored
--batchsize: Batch-size for training and validation
--imgsize: Size of the image after transforms
--resize: Resize data transformation {direct_resize,resizeshortest_randomcrop}
--num_workers: Number of workers for dataloader
--gpu_id: The index of the GPU
--nr_layers: Number of hidden layers (only for ViT)
``