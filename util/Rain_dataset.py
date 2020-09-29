import os
import cv2
import random
import numpy as np


Rain_dataset_path = r'D:\Low level for Real\datasets'

dataset_name = r'Rain_haze'


def Rain_haze_pair_pic_and_crop_for_dncnn():
    rain_path = r'D:\Low level for Real\datasets\Rain_haze\train\in'
    gt_path = r'D:\Low level for Real\datasets\Rain_haze\train\gt'
    streak_path = r'D:\Low level for Real\datasets\Rain_haze\train\streak'

    rain_save_path = r'D:\Low level for Real\DnCNN\data\rain\source\test\rainy'
    gt_save_path = r'D:\Low level for Real\DnCNN\data\rain\source\test\clean'
    streak_save_path = r'D:\Low level for Real\DnCNN\data\rain\source\test\streak'
    print('reading:', rain_path)
    img_index = range(1, 600)
    s_index = ['s80', 's85', 's90', 's95', 's100']
    a_index = ['a04', 'a05', 'a06']
    count = 0
    pic_per_index = 2
    for n_index in img_index:
        n_index = '%04d' % n_index
        if count > 400:
            break
        for j in range(0,pic_per_index):
            s_n = random.randint(0, 4)
            a_n = random.randint(0, 2)
            img_name = 'im_' + n_index + '_' + s_index[s_n] + '_' + a_index[a_n] + '.png'
            # print(os.path.join(precise_dataset_path,img_name))
            rainy = cv2.imread(os.path.join(rain_path, img_name))
            clean = cv2.imread(os.path.join(gt_path, 'im_'+n_index+'.png'))
            streak = cv2.imread(os.path.join(streak_path, img_name))
            rainy = rainy[0:480, 0:480, :]
            clean = clean[0:480, 0:480, :]
            streak = streak[0:480, 0:480, :]
            rainy = cv2.resize(rainy, (600, 600))
            clean = cv2.resize(clean, (600, 600))
            streak = cv2.resize(streak, (600, 600))
            # cv2.imshow('img',img)
            # cv2.waitKey()
            save_name = '%04d.jpg' % count
            count += 1
            cv2.imwrite(os.path.join(rain_save_path, save_name), rainy)
            cv2.imwrite(os.path.join(gt_save_path, save_name), clean)
            cv2.imwrite(os.path.join(streak_save_path, save_name), streak)

def Rain_haze_data_pic_and_crop():
    precise_dataset_path = os.path.join(Rain_dataset_path,dataset_name,'train\in')
    save_path = r'D:\Low level for Real\cycle GAN\pytorch-CycleGAN-and-pix2pix\datasets\realrainyyt2rainhaze\trainA'
    print('reading:',precise_dataset_path)
    img_index = range(1,600)
    s_index = ['s80','s85','s90','s95','s100']
    a_index = ['a04','a05','a06']
    count = 0
    for n_index in img_index:
        n_index = '%04d'% n_index
        s_n = random.randint(0,4)
        a_n = random.randint(0,2)
        img_name = 'im_'+ n_index + '_' + s_index[s_n] + '_' + a_index[a_n] + '.png'
        # print(os.path.join(precise_dataset_path,img_name))
        img = cv2.imread(os.path.join(precise_dataset_path,img_name))
        img = img[0:480,0:480,:]
        img = cv2.resize(img,(600,600))
        # cv2.imshow('img',img)
        # cv2.waitKey()
        save_name =  '%04d.jpg' % count
        count += 1
        cv2.imwrite(os.path.join(save_path,save_name),img)

        # s_n = random.randint(0, 4)
        # a_n = random.randint(0, 2)
        # img_name = 'im_' + n_index + '_' + s_index[s_n] + '_' + a_index[a_n] + '.png'
        # # print(os.path.join(precise_dataset_path, img_name))
        # img = cv2.imread(os.path.join(precise_dataset_path, img_name))
        # img = img[0:480, 0:480, :]
        # img = cv2.resize(img, (600, 600))
        # # cv2.imshow('img',img)
        # # cv2.waitKey()
        # save_name = '%04d.jpg' % count
        # count += 1
        # cv2.imwrite(os.path.join(save_path, save_name), img)


def RID_data_pick_and_crop():
    precise_dataset_path = r'D:\Low level for Real\datasets\RID_Dataset\test'
    save_path = r'D:\Low level for Real\cycle GAN\pytorch-CycleGAN-and-pix2pix\datasets\rain\testB'
    print('reading:',precise_dataset_path)
    img_list = os.listdir(precise_dataset_path)
    count = 0
    for img_name in img_list:
        img = cv2.imread(os.path.join(precise_dataset_path,img_name))
        height,width,_ = img.shape
        square_size = min(height,width)
        img = img[0:square_size,0:square_size,:]
        img = cv2.resize(img,(600,600))
        save_name = '%04d.jpg' % count
        count += 1
        cv2.imwrite(os.path.join(save_path,save_name),img)

def clean_data_pick_in_rain100():
    out_path = r'D:\Low level for Real\cycle GAN\pytorch-CycleGAN-and-pix2pix\datasets\rain2clean\trainB'
    precise_dataset_path = r'D:\Low level for Real\datasets\JORDER_Rain100\rain_data_test_Light\norain'
    img_list = os.listdir(precise_dataset_path)
    count = 1
    for img_name in img_list:
        img = cv2.imread(os.path.join(precise_dataset_path, img_name))
        height, width, _ = img.shape
        square_size = min(height, width)
        img = img[0:square_size, 0:square_size, :]
        img = cv2.resize(img, (600, 600))
        save_name = '%04d.jpg' % count
        count += 1
        if count > 274:
            break
        cv2.imwrite(os.path.join(out_path, save_name), img)

def rain_yyt_crop_resize():
    out_path = r'D:\Low level for Real\cycle GAN\pytorch-CycleGAN-and-pix2pix\datasets\realrainyyt2rainhaze\trainB'
    precise_dataset_path = r'D:\Low level for Real\cycle GAN\pytorch-CycleGAN-and-pix2pix\datasets\realrainyyt2rainhaze\trainB'
    img_list = os.listdir(precise_dataset_path)
    count = 0
    for img_name in img_list:
        img = cv2.imread(os.path.join(precise_dataset_path, img_name))
        height, width, _ = img.shape
        square_size = min(height, width)
        img = img[0:square_size, 0:square_size, :]
        img = cv2.resize(img, (600, 600))
        save_name = '%04d.jpg' % count
        count += 1
        cv2.imwrite(os.path.join(out_path, save_name), img)


if __name__ == '__main__':
    # Rain_haze_data_pic_and_crop()
    # RID_data_pick_and_crop()
    # clean_data_pick_in_rain100()
    # rain_yyt_crop_resize()
    Rain_haze_pair_pic_and_crop_for_dncnn()