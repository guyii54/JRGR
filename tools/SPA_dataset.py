import os
import fnmatch
import shutil
import random
import numpy as np

SPA_root = r'E:\Dataset\Spatial_Real_Dataset'
target_root = r'E:\Dataset\RainCycleGAN_dataset\SPA'

def train_file_pick():
    train_file_txt = os.path.join(SPA_root,'Training','real_world','real_world.txt')
    test_file_txt = os.path.join(SPA_root,'Testing','real_test_1000.txt')
    train_num = 5000
    # split train in Bs and Bt
    Bs_list = []
    Bt_list = []
    Ot_list = []
    # one gt - one rainy
    total_train_list = []
    count = 0
    with open(train_file_txt, 'r') as f:
        for line in f.readlines():
            flag = 0
            gt_name = line.strip().split(' ')[-1]
            for train in total_train_list:
                train_gt_name = train.split(' ')[-1]
                if train_gt_name == gt_name:
                    flag = 1
            if flag == 0:
                total_train_list.append(line.strip())
                count += 1
                print(flag, count)
            else:
                print(flag, count)
            if count >= train_num*2:
                break
    print('length:',len(total_train_list))
    with open(os.path.join(SPA_root,'Training','real_world', 'real_world_no_repetitive.txt'), 'w') as f2:
        for train in total_train_list:
            f2.write('%s\n'%train)
    # random.shuffle(total_train_list)
    # for i in range(0,train_num):
    #     Ot_file_name = total_train_list[i].split(' ')[0]
    #     Bt_file_name = total_train_list[i].split(' ')[1]
    #     little_name = Ot_file_name.split('/')[]
    #     shutil.copy(os.path.join(SPA_root,'train',Ot_file_name), os.path.join(target_root,'train'))





def Copying_train_real_data():
    # os.makedirs(os.path.join(target_root,'train','Ot'))
    # os.makedirs(os.path.join(target_root,'train','Bt'))
    count = 0
    with open(os.path.join(SPA_root,'Training','real_world', 'real_world_no_repetitive.txt'), 'r') as f:
        for line in f.readlines():
            line = line.strip()
            Ot_file_name = line.split(' ')[0]
            Ot_file_name = Ot_file_name.replace('/','\\')
            Bt_file_name = line.split(' ')[1]
            Bt_file_name = Bt_file_name.replace('/','\\')
            little_name = Bt_file_name.split('\\')[-1]
            shutil.copy(os.path.join(r'E:\Dataset\Spatial_Real_Dataset\Training\real_world',Ot_file_name[1:]), os.path.join(target_root,'train','Ot',little_name))
            shutil.copy(os.path.join(r'E:\Dataset\Spatial_Real_Dataset\Training\real_world',Bt_file_name[1:]), os.path.join(target_root,'train','Bt',little_name))
            count+=1
            print(count)
            if count>=5000:
                break

def Copying_Syn_data_Bs():
    # os.makedirs(os.path.join(target_root,'train','Bs'))
    # os.makedirs(os.path.join(target_root,'train','Os_r'))
    count = 0
    with open(os.path.join(SPA_root, 'Training', 'real_world', 'real_world_no_repetitive.txt'), 'r') as f:
        for line in f.readlines():
            count+=1
            if count<=5000:
                continue
            line = line.strip()
            Os_file_name = line.split(' ')[0]
            Os_file_name = Os_file_name.replace('/', '\\')
            Bs_file_name = line.split(' ')[1]
            Bs_file_name = Bs_file_name.replace('/', '\\')
            little_name = Bs_file_name.split('\\')[-1]
            shutil.copy(os.path.join(r'E:\Dataset\Spatial_Real_Dataset\Training\real_world', Os_file_name[1:]),
                        os.path.join(target_root, 'train', 'Os_r', little_name))
            shutil.copy(os.path.join(r'E:\Dataset\Spatial_Real_Dataset\Training\real_world', Bs_file_name[1:]),
                        os.path.join(target_root, 'train', 'Bs', little_name))
            print(count)

def writing_test_bs_txt():
    syn_bs_dir = r'E:\Dataset\RainCycleGAN_dataset\SPA\test\Bs'
    txt_file = r'E:\Dataset\Spatial_Real_Dataset\Testing\test_Bs.txt'
    img_list = os.listdir(syn_bs_dir)
    with open(txt_file, 'w') as f:
        for img_name in img_list:
            f.write('%s\n' % img_name)


def write_syn_txt():
    syn_bs_dir = r'E:\Dataset\RainCycleGAN_dataset\SPA\train\Bs'
    txt_file = r'E:\Dataset\Spatial_Real_Dataset\Training\real_world\Bs.txt'
    img_list = os.listdir(syn_bs_dir)
    with open(txt_file, 'w') as f:
        for img_name in img_list:
            f.write('%s\n'%img_name)

# def debug_spa_gen1():
#     img_file = r'E:\Dataset\Spatial_Real_Dataset\Training\real_world\Os\rain\simu_leftImg8bit\017_12_5.png'
#     import cv2
#     img = cv2.imread(img_file)
#     summ = 0
#     for i in range(img.shape[0]):
#         for j in range(img.shape[1]):
#             pixel = img[i,j,0]
#             if pixel != 0:
#                 print(i,j)

def cut_str_gt_in_Bt_in_test():
    Bt_dir = r'E:\Dataset\RainCycleGAN_dataset\SPA_House\test\picked_gt'
    os.chdir(Bt_dir)
    Bt_list = os.listdir(Bt_dir)
    for Bt_name in Bt_list:
        Bt_new_name = Bt_name.replace('gt','')
        print(Bt_new_name)
        os.rename(Bt_name, Bt_new_name)

def Cut_datas():
    dir_to_cut = r'E:\Dataset\RainCycleGAN_dataset\SPA_House\test\Bs'
    final_dir = r'E:\Dataset\RainCycleGAN_dataset\SPA_House\test\Os'
    img_list = os.listdir(dir_to_cut)
    count = 0
    for img_name in img_list:
        if not os.path.exists(os.path.join(final_dir, img_name)):
            os.remove(os.path.join(dir_to_cut,img_name))
            # print(img_name)
            pass
        else:
            count+=1
    print(count)


def Pick_SPA_House():
    SPA_root = r'E:\Dataset\RainCycleGAN_dataset\SPA'
    SPA_house_root = r'E:\Dataset\RainCycleGAN_dataset\SPA_House'
    Os_list = os.listdir(os.path.join(SPA_house_root,'train','Os'))
    Ot_list = os.listdir(os.path.join(SPA_house_root,'train','Ot'))
    for Os_name in Os_list:
        shutil.copy(os.path.join(SPA_root,'train','Bs',Os_name),os.path.join(os.path.join(SPA_house_root), 'train', 'Bs', Os_name))
    for Ot_name in Ot_list:
        shutil.copy(os.path.join(SPA_root,'train','Bt',Ot_name),os.path.join(os.path.join(SPA_house_root), 'train', 'Bt', Ot_name))


def process_test_file():
    test_picked_path = r'E:\Dataset\RainCycleGAN_dataset\SPA_House\test\picked_gt'
    Ot_source_path = r'E:\Dataset\Spatial_Real_Dataset\Training\real_world\Os\rain\simu_leftImg8bit'
    Ot_target_path = r'E:\Dataset\RainCycleGAN_dataset\SPA_House\test\Os'
    img_list = os.listdir(test_picked_path)
    # txt_file = r'E:\Dataset\RainCycleGAN_dataset\SPA_House\test\test.txt'
    # with open(txt_file, 'w') as f:
    for img_name in img_list:
        print(os.path.join(Ot_target_path, img_name))
        shutil.copy(os.path.join(Ot_source_path, img_name), os.path.join(Ot_target_path, img_name))


if __name__ =='__main__':
    # Copying_Syn_data_Bs()
    # writing_test_bs_txt()
    Cut_datas()
    # Pick_SPA_House()
    # process_test_file()
    # cut_str_gt_in_Bt_in_test()
    # debug_spa_gen1()