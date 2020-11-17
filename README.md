# SR-ITM-GAN
This repository is an official PyTorch implementation of the paper "SR-ITM-GAN: Learning 4k UHD HDR with a Generative Adversarial Network" [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9212411) [[Axiv]]()

The uncompressed pdf version could be found [here](https://drive.google.com/drive/folders/1fuiBLHvYz9wH_TbtdSuA7szClLu4v_-v?usp=sharing)
## Authors
Huimin Zeng, Xinliang Zhang, Yubo Wang and Zhibin Yu

## Introduction
<p align="center">
<img src="./img/archi.png" width="80%"/>
</p>
Architecture of our generator.
<p></p>

<p align="center">
<img src="./img/all.png" width="90%"/>
</p>
From top to bottom, we display basic results of Pix2Pix, CycleGAN, Deep-SR-ITM, ESRGAN, our SR-ITM-GAN and HDR GT.
<p></p>

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
└── tb_logger
```
## Data Preparation
* Download our dataset here [Google Drive](https://drive.google.com/drive/folders/1fuiBLHvYz9wH_TbtdSuA7szClLu4v_-v?usp=sharing), [Baidu Cloud](https://pan.baidu.com/s/1KVK1_r3Z5GQC6AtOcOEUlA)(extract code: ue6f)
* Extract images with following FFMPEG(datasets are saved as .mp4 format for convenience):
```
ffmpeg -i /path/to/video/dataset.mp4 -q:v 2  /path/to/extract/location/%d.png
```
* Pack RGB dataset into lmdb format for accelerating.
```
python ./data_scripts/create_lmdb.py
```
## Train
```
# Step 1:  modify your training datasets in /options/train/train.yml
# Step 2: run the script:
cd codes
python train.py -opt ./options/train/train.yml
```
## Test
```
cd codes
python test.py -opt ./options/test/test.yml
```
## Citation
If you find the Repository useful, please give us a star.:blush:

Using our dataset or code, please cite the following:
```
@ARTICLE{9212411,
  author={H. {Zeng} and X. {Zhang} and Z. {Yu} and Y. {Wang}},
  journal={IEEE Access}, 
  title={SR-ITM-GAN: Learning 4K UHD HDR With a Generative Adversarial Network}, 
  year={2020},
  volume={8},
  number={},
  pages={182815-182827},
  doi={10.1109/ACCESS.2020.3028584}}
```
## Contact
Feel free to contact ```cenghuimin@stu.ouc.edu.cn``` if there's any problem.
