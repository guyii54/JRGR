import os

os.system('python train.py --dataroot /home/solanliu/yeyuntong/RainCycleGAN_dataset/cityRH --dataset_mode rain --singleDNCNN_load_path ./checkpoints/UnetDerain --model raincycle --batch_size 8 --gpu_ids 0,1,2,3 --lambda_MSE 50 --lambda_GAN 4 --lambda_Cycle 40 --name loss_50_4_40')
os.system('python train.py --dataroot /home/solanliu/yeyuntong/RainCycleGAN_dataset/cityRH --dataset_mode rain --singleDNCNN_load_path ./checkpoints/UnetDerain --model raincycle --batch_size 8 --gpu_ids 0,1,2,3 --lambda_MSE 30 --lambda_GAN 4 --lambda_Cycle 40 --name loss_30_4_40')
