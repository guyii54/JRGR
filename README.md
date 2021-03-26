# Closing the Loop: Joint Rain Generation and Removal via Disentangled Image Translation

Torch implementation Closing the Loop: Joint Rain Generation and Removal via Disentangled Image Translation in CVPR 2021. [[paper]](https://arxiv.org/abs/2103.13660)


# Demo

## Rain Removal and Generation
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://github.com/guyii54/JRGR/blob/master/Demo_Derain.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Real Rain Removal</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://github.com/guyii54/JRGR/blob/master/Demo_Generation.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Real Rain Generation</div>
</center>

## Network Architecture
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://github.com/guyii54/JRGR/blob/master/Architecture.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Architecture</div>
</center>


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
https://github.com/guyii54/Real-Rainy-Image-Datasets

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
  python train.py --dataroot ./dataset/cityscape --dataset_mode rain --model unetderain --name UnetDerain
  python train.py --dataroot ./dataset/cityscape --dataset_mode rain --unet_load_path ./checkpoints/UnetDerain --model raincycle --name JRGR --init_derain 1,3
  # Sencondary training strategy: directly joint train
  python train.py --dataroot ./dataset/cityscape --dataset_mode rain --model raincycle --name JRGR --init_derain 0
  ```

- Test the model
  ```
  python test.py --dataroot ./dataset/cityscape --dataset_mode rain --model raincycle --name JRGR
  ```
  The test results will be saved to a html file here: ./results/RO_JRGR/latest_test/index.html.

# Citation

Our code is inspired by [Cycle GAN](https://github.com/junyanz/CycleGAN).
