# SR-ITM-GAN
This repository is an official PyTorch implementation of the paper "SR-ITM-GAN: Learning 4k UHD HDR with a Generative Adversarial Network" [[Paper]]() [[Axiv]]()
## Authors
Huimin Zeng, Xinliang Zhang, Yubo Wang and Zhibin Yu

## Introduction
## Environments
* CUDA 9.0 & cuDNN 7.0
* Python 3.6
* Pytorch >= 1.1
* opencv
* lmdb
* pyyaml


## Code
Clone this repository using the following command:
```bash
git clone https://github.com/ZeldaM1/SR-ITM-GAN.git
cd SR-ITM-GAN
```
## File Strcture
```
SR-ITM-GAN
├── code
        ├── data
        ├── data_scripts
        ├── metrics
        ├── models
        ├── options
        ├── scripts
        ├── utils
        ├── train.py
        └── test.py
├── experiments
    ├── ST-ITM-GAN
        ├── models
        ├── training_state
        ├── val_images
        └── training.log
├── option
└── tb_logger
```
## Data Preparation
```
python /data_scripts/create_lmdb.py
```
## Train
```
# Step 1:  modify your training datasets in /options/train/train.yml
# Step 2: run the script:
cd codes
python train.py -opt /options/train/train.yml
```
## Test
```
cd codes
python test.py -opt /options/test/test.yml
```
## Citation
## We will realease the code soon
