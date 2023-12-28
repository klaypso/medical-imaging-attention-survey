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
--dat