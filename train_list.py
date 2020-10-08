import os

# 2020.10.6.
# os.system('python train.py --dataroot /home/solanliu/yeyuntong/RainCycleGAN_dataset/light_cityRH --dataset_mode rain --model unetderain --batch_size 8 --gpu_ids 0,1,2,3 --lr 0.0001 --name light_UnetDerain')
#
# os.system('python train.py --dataroot /home/solanliu/yeyuntong/RainCycleGAN_dataset/light_cityRH --dataset_mode rain --singleDNCNN_load_path ./checkpoints/light_UnetDerain --model raincycle --batch_size 8 --gpu_ids 0,1,2,3 --lr 0.0001 --lambda_MSE 50 --lambda_GAN 4 --lambda_Cycle 40 --name lrs1e-4')

# os.system('python train.py --dataroot /home/solanliu/yeyuntong/RainCycleGAN_dataset/cityRH --dataset_mode rain --singleDNCNN_load_path ./checkpoints/UnetDerain --model raincycle --batch_size 8 --gpu_ids 0,1,2,3 --lr 0.0001 --lambda_MSE 80 --lambda_GAN 4 --lambda_Cycle 40 --name lr1e-4_loss_80_4_40')
#


# 2020.10.6.
os.system('python train.py --dataroot G:\Dataset\RainCycleGAN_dataset\middle_cityRH --dataset_mode rain --model cycle_gan --batch_size 1 --gpu_ids 0 --lr 0.0001 --name middle_cyclegan')
os.system('python train.py --dataroot /home/solanliu/yeyuntong/RainCycleGAN_dataset/light_cityRH --dataset_mode rain --model unetderain --batch_size 8 --gpu_ids 0,1,2,3 --lr 0.0001 --name light_UnetDerain')


# ubuntu
os.system('python train.py --dataroot /home/solanliu/yeyuntong/RainCycleGAN_dataset/light_cityRH --dataset_mode rain --singleDNCNN_load_path ./checkpoints/light_UnetDerain --model raincycle --batch_size 8 --gpu_ids 0,1,2,3 --lr 0.0001 --lambda_MSE 50 --lambda_GAN 4 --lambda_Cycle 40 --name lrs1e-4')

os.system('python train.py --dataroot /home/solanliu/yeyuntong/RainCycleGAN_dataset/middle_cityRH --dataset_mode rain --init_derain 0 --model raincycle --batch_size 8 --gpu_ids 0,1,2,3 --lr 0.0001 --lambda_MSE 50 --lambda_GAN 4 --lambda_Cycle 40 --name m_lrs1e-4_Init0')

# os.system('python train.py --dataroot /home/solanliu/yeyuntong/RainCycleGAN_dataset/middle_cityRH --dataset_mode rain --singleDNCNN_load_path ./checkpoints/middle_UnetDerain --init_derain 1 --model raincycle --batch_size 8 --gpu_ids 0,1,2,3 --lr 0.0001 --lambda_MSE 50 --lambda_GAN 4 --lambda_Cycle 40 --name m_lrs1e-4_Init1')
#
#
# os.system('python train.py --dataroot /home/solanliu/yeyuntong/RainCycleGAN_dataset/cityRH --dataset_mode rain --model unetderain --batch_size 8 --gpu_ids 0,1,2,3 --lr 0.0001 --name heavy_UnetDerain')
#
#
# os.system('python train.py --dataroot /home/solanliu/yeyuntong/RainCycleGAN_dataset/cityRH --dataset_mode rain --init_derain 0 --model raincycle --batch_size 8 --gpu_ids 0,1,2,3 --lr 0.0001 --lambda_MSE 80 --lambda_GAN 4 --lambda_Cycle 40 --name m_lrs1e-4')

# os.system('python train.py --dataroot /home/solanliu/yeyuntong/RainCycleGAN_dataset/middle_cityRH --dataset_mode rain --singleDNCNN_load_path ./checkpoints/middle_UnetDerain --model raincycle --batch_size 8 --gpu_ids 0,1,2,3 --lr 0.0001 --lambda_MSE 50 --lambda_GAN 4 --lambda_Cycle 40 --name m_lrs1e-4')


# os.system('python train.py --dataroot /home/solanliu/yeyuntong/RainCycleGAN_dataset/cityRH --dataset_mode rain --singleDNCNN_load_path ./checkpoints/UnetDerain --model raincycle --batch_size 8 --gpu_ids 0,1,2,3 --lr 0.0001 --lambda_MSE 80 --lambda_GAN 4 --lambda_Cycle 40 --name lr1e-4_loss_80_4_40')
#