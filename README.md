This repository contains an implementation of the following paper:

**V. K. Singh et al., "FCA-Net: Adversarial Learning for Skin Lesion Segmentation Based on Multi-Scale Features and Factorized Channel Attention," in IEEE Access, vol. 7, pp. 130552-130565, 2019, doi: 10.1109/ACCESS.2019.2940418.**

# FCA-Net

![skin_new_model](https://user-images.githubusercontent.com/18607766/59305103-d7357300-8c99-11e9-923a-9c09ef49a210.png)

## Prerequisites

+ Linux
+ Python with numpy
+ NVIDIA GPU + CUDA 8.0 + CuDNNv5.1
+ pytorch 4.0/4.1
+ torchvision

## Getting Started


**+ Clone this repo:**

    cd Skin-Project

**+ Get dataset**

    unzip dataset/skin.zip

**+ Train the model:**

    python train.py --dataset skin --nEpochs 200 --cuda

**+ Test the model:**

    python test.py --dataset skin --model checkpoint/skin/netG_model_epoch_100.pth --cuda

**+ Citation:**

@ARTICLE{8832175,
  author={V. K. {Singh} and M. {Abdel-Nasser} and H. A. {Rashwan} and F. {Akram} and N. {Pandey} and A. {Lalande} and B. {Presles} and S. {Romani} and D. {Puig}},
  journal={IEEE Access}, 
  title={FCA-Net: Adversarial Learning for Skin Lesion Segmentation Based on Multi-Scale Features and Factorized Channel Attention}, 
  year={2019},
  volume={7},
  number={},
  pages={130552-130565},
  doi={10.1109/ACCESS.2019.2940418}}
