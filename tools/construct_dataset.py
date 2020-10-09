import os
import random
import shutil

target_root = r'G:\Dataset\RainCycleGAN_dataset\middle_cityRH'
train_root = os.path.join(target_root,'train')
test_root = os.path.join(target_root, 'test')
Ot_src_dir = r'G:\Dataset\2020IJCV\weather_datasets\weather_cityscapes\leftImg8bit\train\rain\50mm_resize_600x300'
Os_src_dir = r'G:\Dataset\RainHazeGenerater\output\hazerain_level2\haze_rain\simu_leftImg8bit\train'
B_src_dir = r'G:\Dataset\RainHazeGenerater\data\cityscape\leftImg8bit\train_Resize_600x300'
city_train_file = r'G:\Dataset\RainHazeGenerater\data\cityscape\cityscapeC.txt'

# os.makedirs(target_root)
# os.makedirs(test_root)

total_list = []
with open(city_train_file, 'r') as f:
    for line in f.readlines():
        total_list.append(line.strip())

random.shuffle(total_list)
train_num = 1400
test_num = len(total_list) - 2*train_num

# for Ot_name in total_list[0:train_num]:
#     little_name = Ot_name.split('\\')[-1]
#     shutil.copy(os.path.join(Ot_src_dir,little_name), os.path.join(target_root,'Ot',little_name))
#     shutil.copy(os.path.join(B_src_dir,little_name), os.path.join(target_root, 'Bt',little_name))
# print('Ot and Bt over.')
#
#
# for Os_name in total_list[train_num:train_num*2]:
#     little_name = Os_name.split('\\')[-1]
#     shutil.copy(os.path.join(Os_src_dir, Os_name), os.path.join(target_root,'Os',little_name))
#     shutil.copy(os.path.join(B_src_dir,little_name), os.path.join(target_root, 'Bs',little_name))
#
# print('Os and Bs over.')

# Testing
count = 0
for Ot_name in total_list:
    little_name = Ot_name.split('\\')[-1]
    if not os.path.exists(os.path.join(train_root, 'Os',little_name)):
        if not os.path.exists(os.path.join(train_root, 'Ot',little_name)):
            count += 1
            shutil.copy(os.path.join(Ot_src_dir,little_name), os.path.join(test_root,'Ot',little_name))
            shutil.copy(os.path.join(B_src_dir,little_name), os.path.join(test_root, 'Bt',little_name))
            shutil.copy(os.path.join(Os_src_dir, Ot_name), os.path.join(test_root, 'Os', little_name))
            shutil.copy(os.path.join(B_src_dir, little_name), os.path.join(test_root, 'Bs', little_name))

print(count)
