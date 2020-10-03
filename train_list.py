import os


os.system('python train.py --dataroot /home/solanliu/yeyuntong/RainCycleGAN_dataset/cityRH --dataset_mode rain --singleDNCNN_load_path ./checkpoints/UnetDerain --model raincycle --batch_size 8 --gpu_ids 0,1,2,3 --lr 0.0001 --lambda_MSE 40 --lambda_GAN 4 --lambda_Cycle 40 --name lr_split_1e-4')
# os.system('python train.py --dataroot /home/solanliu/yeyuntong/RainCycleGAN_dataset/cityRH --dataset_mode rain --singleDNCNN_load_path ./checkpoints/UnetDerain --model raincycle --batch_size 8 --gpu_ids 0,1,2,3 --lr 0.00005 --lambda_MSE 30 --lambda_GAN 4 --lambda_Cycle 40 --name lr5e-5_loss_50_4_40')
