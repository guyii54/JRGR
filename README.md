# Closing the Loop: Joint Rain Generation and Removal via Disentangled Image Translation

Torch implementation for Closing the Loop: Joint Rain Generation and Removal via Disentangled Image Translation

# Prerequisites
- Linux or Windows
- Python 3.6
- CPU or NVIDIA GPU + CUDA CuDNN


# Get Started

## Installation
- clone this repo


- Install Pytorch 1.6.0 and other dependencies (e.g., torchvision, visdom and dominate). The requirment of main dependencies are listed in requirements.txt


## Dataset
Our collected dataset RealRain can be downlowed here.

## JRGR train/test
- Download our dataset or make your own dataset, the dataset should have the following structure:
  ```
  -train
    -Ot
    -Os
    -Bs
  -test
    -Ot
    -Os
    -Bs
  ```
The directory Ot, Os, Bs save the real rainy images, synthetic rainy images and the corresponding backgrounds of synthetic rainy images. 

If you have the ground truth of real rainy images and you want to visualize them in the results, you can add Bt directory in the dataset and add the config --Bt_access 1 during training and testing.

- Train the model
  ```
  # Proposed training strategy: pre-train and joint train
  python train.py --dataroot ../Real_Overcast --dataset_mode rain --model unetderain --name RO_UnetDerain
  python train.py --dataroot ../Real_Overcast --dataset_mode rain --unet_load_path ./checkpoints/RO_UnetDerain --model raincycle --name RO_JRGR
  # Sencondary training strategy: directly joint train
  python train.py --dataroot ../Real_Overcast --dataset_mode rain --model raincycle --name RO_JRGR --init 0
  ```

- Test the model
  ```
  python test.py --dataroot ../Real_Overcast --dataset_mode rain --model raincycle --name RO_JRGR --init 0
  ```
  The test results will be saved to a html file here: ./results/RO_JRGR/latest_test/index.html.

# Citation

Our code is inspired by [Cycle GAN](https://github.com/junyanz/CycleGAN)