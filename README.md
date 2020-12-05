# SafeUAVNet pretrained on Ruralscapes Dataset 

Training and testing code for SafeUAVNet accompanying Ruralscapes Dataset and SegProp Code

This repo is part of the code release for the paper Semantics through Time: Semi-supervised Segmentation of Aerial Videos with Iterative Label Propagation, accepted as oral presentation at ACCV 2020.

Paper: https://arxiv.org/abs/2010.01910

SegProp Code and Ruralscapes Dataset are available on our project page: https://sites.google.com/site/aerialimageunderstanding/semantics-through-time-semi-supervised-segmentation-of-aerial-videos

### Setup

Create conda environment

```sh
$ conda env create -f environment.yml
```

### Inference
Pretrained weights can be found here: https://drive.google.com/drive/folders/1tdoq9I0IxEc5QotNCyddGf3f32g9F2Pq?usp=sharing

Example script for testing SafeUAVNet:

```sh
$ python3 test_qualitative_evaluation.py
```

### Training (optional)

Example script for training SafeUAVNet:

```sh
$ python3 train_baseline_ground_truth_only.py
```
