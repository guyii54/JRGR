import cv2
import os
import fnmatch


def pick_frame_in_batch(video_path,save_path):
    video_names = os.listdir(video_path)
    video_names = fnmatch.filter(video_names,'*.mp4')
    frame_intern = 100
    # sequence_path = r'D:\Low level for Real\datasets\Yuchangfeng\sequences'
    # precise_save_path = os.path.join(sequence_path,video_name[:-4])
    # print(precise_save_path)
    count = 0
    for video_name in video_names:
        print('reading %s'%video_name)
        video_read = cv2.VideoCapture(os.path.join(video_path, video_name))
        while True:
            ret,img = video_read.read()
            if ret == True:
                if count % frame_intern == 0:
                    cv2.imwrite(os.path.join(save_path,'%04d.jpg'%(count/frame_intern)),img)
                    # cv2.imshow('img',img)
                    # cv2.waitKey()
                count += 1
                if count/frame_intern > 85:
                    break
            else:
                break


def rename_batch(rename_path):
    os.chdir(rename_path)
    video_names = os.listdir(rename_path)
    video_names = fnmatch.filter(video_names,'*.mp4')
    count = 1
    for video_name in video_names:
        os.rename(video_name,'sequence_%02d.mp4'%count)
        count += 1

if __name__ == '__main__':
    # rename_path = r'D:\Low level for Real\datasets\Yeyuntong\rain'
    # rename_batch(rename_path)
    video_path = r'D:\Low level for Real\datasets\Yeyuntong\rain\tree'
    save_path = r'D:\Low level for Real\cycle GAN\pytorch-CycleGAN-and-pix2pix\datasets\realrainyyt2rainhaze\testB'
    pick_frame_in_batch(video_path,save_path)