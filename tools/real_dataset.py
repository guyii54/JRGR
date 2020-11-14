import os
import cv2
import fnmatch
import random



def write_data_file():
    data_root = r'E:\Dataset\RainCycleGAN_dataset\Real_Overcast\train\Bs'
    data_file = r'Bs_list.txt'
    # dir_list = os.listdir(data_root)
    img_list = os.listdir(data_root)
    print(len(img_list))
    # img_list = fnmatch.filter(img_list, '*.jpg')
    with open(data_file, 'w') as f:
        # for dir in dir_list:
        for name in img_list:
            img_file = name
            print(img_file)
            f.write('%s\n'%img_file)

def random_crop(img, tosize=256):
    img_c = img.copy()
    height, width, _ = img_c.shape
    half_size = int(tosize/2)
    height_center = random.randint(half_size,height-half_size)
    width_center = random.randint(half_size,width-half_size)
    img_crop = img_c[height_center-half_size:height_center+half_size, width_center-half_size:width_center+half_size,:]
    return img_crop

def dual_random_crop(img1, img2, tosize=256):
    img1_c = img1.copy()
    img2_c = img2.copy()
    height, width, _ = img1_c.shape
    half_size = int(tosize/2)
    height_center = random.randint(half_size,height-half_size)
    width_center = random.randint(half_size,width-half_size)
    img1_crop = img1_c[height_center-half_size:height_center+half_size, width_center-half_size:width_center+half_size,:]
    img2_crop = img2_c[height_center-half_size:height_center+half_size, width_center-half_size:width_center+half_size,:]
    return img1_crop, img2_crop


def see_delete_file():
    from os.path import join as path_join
    see_path = r'E:\Dataset\Real_Rain\Random_cropped'
    f = open(path_join(see_path,'last_count.txt'), 'r+')
    img_list = os.listdir(see_path)
    img_list = fnmatch.filter(img_list, '*.png')
    start = f.readline().strip()
    # start = img_list[0]
    count = 0
    begin_flag = 0
    for img_name in img_list:
        count += 1
        if img_name==start:
            begin_flag = 1
        if begin_flag == 1:
            img = cv2.imread(path_join(see_path,img_name))
            cv2.imshow('img',img)
            key = cv2.waitKey()
            if key == ord('d'):
                os.remove(path_join(see_path,img_name))
            elif key==27:
                f.write('%s'%img_name)
                f.close
                break
            else:
                pass
        else:
            pass
        
        print('%d/%d'%(count,len(img_list)))


def crop_big_photo():
    data_root = r'E:\Dataset\Real_Rain\selected_image'
    # data_file = r'yuntong_capture.txt'
    data_file = None
    save_root = r'E:\Dataset\Real_Rain\Random_cropped'
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    img_list = []
    if data_file is None:
        img_list = os.listdir(data_root)
        img_list = fnmatch.filter(img_list, '*.jpg')
    else:
        with open(data_file, 'r') as f:
            for line in f.readlines():
                img_list.append(line.strip())
    # crop how many pathes in one picture
    crop_times = 100
    total_count = 0
    for img_name in img_list:
        img_file = os.path.join(data_root, img_name)
        img = cv2.imread(img_file)
        for count in range(0,crop_times):
            img_crop = random_crop(img, 100)
            total_count += 1
            save_name = '%05d.png'% total_count
            cv2.imwrite(os.path.join(save_root,save_name),img_crop)
            # cv2.imshow('raw',img)
            # cv2.imshow('img_crop',img_crop)
            # cv2.waitKey()

def pick_half_photo():
    from os.path import join as join
    image_path = r'E:\Dataset\Rain800\selected_image'
    clean_saved_path = r'E:\Dataset\Rain800\Bs'
    rainy_saved_path = r'E:\Dataset\Rain800\Os'
    img_list = os.listdir(image_path)
    img_list = fnmatch.filter(img_list,'*.jpg')
    for img_name in img_list:
        img = cv2.imread(join(image_path,img_name))
        width = img.shape[1]
        clean_image = img[:,0:int(width/2),:]
        rainy_image = img[:,int(width/2):width,:]
        cv2.imwrite(join(clean_saved_path,img_name),clean_image)
        cv2.imwrite(join(rainy_saved_path,img_name),rainy_image)

def dual_crop_big_photo():
    Os_root = r'E:\Dataset\Rain800\Os'
    Bs_root = r'E:\Dataset\Rain800\Bs'
    # data_file = r'yuntong_capture.txt'
    data_file = None
    coped_Os_save_root = r'E:\Dataset\Rain800\croped_Os'
    coped_Bs_save_root = r'E:\Dataset\Rain800\croped_Bs'
    if not os.path.exists(coped_Os_save_root):
        os.makedirs(coped_Os_save_root)
    if not os.path.exists(coped_Bs_save_root):
        os.makedirs(coped_Bs_save_root)
    img_list = []
    if data_file is None:
        img_list = os.listdir(Os_root)
        img_list = fnmatch.filter(img_list, '*.jpg')
    else:
        with open(data_file, 'r') as f:
            for line in f.readlines():
                img_list.append(line.strip())
    # crop how many pathes in one picture
    crop_times = 100
    total_count = 0
    for img_name in img_list:
        Os_file = os.path.join(Os_root, img_name)
        Bs_file = os.path.join(Bs_root, img_name) 
        Os_img = cv2.imread(Os_file)
        Bs_img = cv2.imread(Bs_file)
        for count in range(0,crop_times):
            Os_crop, Bs_crop = dual_random_crop(Os_img, Bs_img, 100)
            total_count += 1
            save_name = '%05d.png'% total_count
            cv2.imwrite(os.path.join(coped_Os_save_root,save_name),Os_crop)
            cv2.imwrite(os.path.join(coped_Bs_save_root,save_name),Bs_crop)
            # cv2.imshow('raw',img)
            # cv2.imshow('img_crop',img_crop)
            # cv2.waitKey()

def crop_to_square():
    check_dir = r'E:\Dataset\RainCycleGAN_dataset\Real_Raw\test\Ot'
    img_list = os.listdir(check_dir)
    for img_name in img_list:
        img = cv2.imread(os.path.join(check_dir, img_name))
        img_crop = img.copy()
        height, width, _ = img.shape
        center_h = int(height/2)
        center_w = int(width/2)
        square_half = int(min(height, width)/2)
        print(square_half)
        # img_crop = img_crop[center_w-square_half:center_w+square_half, center_h-square_half:center_h+square_half,:]
        img_crop = img_crop[center_h-square_half:center_h+square_half, center_w-square_half:center_w+square_half,:]
        cv2.imshow('img',img_crop)
        cv2.waitKey()


def resize_to_square():
    check_dir = r'E:\Dataset\RainCycleGAN_dataset\Real_Raw\test\Ot_raw'
    save_dir = r'E:\Dataset\RainCycleGAN_dataset\Real_Raw\test\Ot'
    img_list = os.listdir(check_dir)
    for img_name in img_list:
        img = cv2.imread(os.path.join(check_dir, img_name))
        img_crop = img.copy()
        height, width, _ = img.shape
        square = min(height, width)
        img_crop = cv2.resize(img_crop, (square,square))
        # img_crop = img_crop[center_w-square_half:center_w+square_half, center_h-square_half:center_h+square_half,:]
        # img_crop = img_crop[center_h-square_half:center_h+square_half, center_w-square_half:center_w+square_half,:]
        # cv2.imshow('img',img_crop)
        # cv2.waitKey()
        cv2.imwrite(os.path.join(save_dir,img_name),img_crop)


if __name__ == '__main__':
    # write_data_file()
    # ---------------random_crop---------------
    # crop_big_photo()


    # #----------------see and detele-------------------------
    # see_delete_file()

    # pick_half_photo()
    # dual_crop_big_photo()
    write_data_file()
