import os


os.system('python train.py --dataroot /home/solanliu/yeyuntong/RainCycleGAN_dataset/light_cityRH --dataset_mode rain --model unetderain --batch_size 8 --gpu_ids 0,1,2,3 --lr 0.0001 --name light_UnetDerain')

os.system('python train.py --dataroot /home/solanliu/yeyuntong/RainCycleGAN_dataset/light_cityRH --dataset_mode rain --singleDNCNN_load_path ./checkpoints/light_UnetDerain --model raincycle --batch_size 8 --gpu_ids 0,1,2,3 --lr 0.0001 --lambda_MSE 50 --lambda_GAN 4 --lambda_Cycle 40 --name lrs1e-4')

# os.system('python train.py --dataroot /home/solanliu/yeyuntong/RainCycleGAN_dataset/cityRH --dataset_mode rain --singleDNCNN_load_path ./checkpoints/UnetDerain --model raincycle --batch_size 8 --gpu_ids 0,1,2,3 --lr 0.0001 --lambda_MSE 80 --lambda_GAN 4 --lambda_Cycle 40 --name lr1e-4_loss_80_4_40')
#