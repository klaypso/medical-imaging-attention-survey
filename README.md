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
```

### Generate Figures from Post-hoc Explanations (Saliency Maps)
To generate figures from post-hoc explanations (saliency maps):
```bash
$ python code/generate_xai_figures.py {command line arguments}
```
This script accepts the following command line arguments:
```
--modelckpt: Directory where model is stored
--saliency_maps: Saliency maps {ALL, DEEPLIFT, LRP}
--alpha_overlay: Alpha parameter for overlayed saliency maps.
```

## Citation
If you use this repository in your research work, please cite this paper:
```bibtex
@ARTICLE{gonccalves2022survey,  
  author={Gonçalves, Tiago and Rio-Torto, Isabel and Teixeira, Luís F. and Cardoso, Jaime S.},  
  journal={IEEE Access},   
  title={A Survey on Attention Mechanisms for Medical Applications: are we Moving Toward Better Algorithms?},   
  year={2022},  
  volume={10},  
  number={},  
  pages={98909-98935},  
  abstract={The increasing popularity of attention mechanisms in deep learning algorithms for computer vision and natural language processing made these models attractive to other research domains. In healthcare, there is a strong need for tools that may improve the routines of the clinicians and the patients. Naturally, the use of attention-based algorithms for medical applications occurred smoothly. However, being healthcare a domain that depends on high-stake decisions, the scientific community must ponder if these high-performing algorithms fit the needs of medical applications. With this motto, this paper extensively reviews the use of attention mechanisms in machine learning methods (including Transformers) for several medical applications based on the types of tasks that may integrate several works pipelines of the medical domain. This work distinguishes itself from its predecessors by proposing a critical analysis of the claims and potentialities of attention mechanisms presented in the literature through an experimental case study on medical image classification with three different use cases. These experiments focus on the integrating process of attention mechanisms into established deep learning architectures, the analysis of their predictive power, and a visual assessment of their saliency maps generated by post-hoc explanation methods. This paper concludes with a critical analysis of the claims and potentialities presented in the literature about attention mechanisms and proposes future research lines in medical applications that may benefit from these frameworks.},  
  keywords={},  
  doi={10.1109/ACCESS.2022.3206449},  
  ISSN={2169-3536},  
  month={},
}
```

## Credits and Acknowledgments
### Squeeze-and-Excitation (SE) Networks
This model and associated [**code**](https://github.com/moskomule/senet.pytorch) are related to the paper [_'Squeeze-and-Excitation Networks'_](https://ieeexplore.ieee.org/abstract/document/8759331) by Jie Hu, Li Shen, Samuel Albanie, Gang Sun and Enhua Wu.

### Convolutional Block Attention Module (CBAM)
This model and associated [**code**](https://github.com/Jongchan/attention-module) are related to the paper [_'CBAM: Convolutional Block Attention Module'_](https://openaccess.thecvf.com/content_ECCV_2018/html/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.html) by Sanghyun Woo, Jongchan Park, Joon-Young Lee and In So Kweon.

### Transformer Explainability
This framework and associated [**code**](https://github.com/hila-chefer/Transformer-Explainability) are related to the paper [_'Transformer Interpretability Beyond Attention Visualization'_](https://arxiv.org/abs/2012.09838) by Hila Chefer, Shir Gur and Lior Wolf.
