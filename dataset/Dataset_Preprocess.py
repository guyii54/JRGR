import numpy as np
import cv2
import os
from os.path import join
import shutil

def getWindows(raw_image, windows_size, stride):
    h, w, _ = raw_image.shape
    patchList = []
    for i in range(0, h-windows_size, stride):
        for j in range(0, w-windows_size, stride):
            patch = raw_image[i:i+windows_size, j:j+windows_size, :]
            patchList.append(patch)
    return patchList


def dataSetCrops(dataroot, saveroot):
    # dataroot = r''
    # saveroot = r''
    window_size = 256
    stride = 256

    img_list = os.listdir(dataroot)
    for img_name in img_list:
        img = cv2.imread(join(dataroot, img_name))
        patches = getWindows(img, window_size, stride)
        for index, patch in enumerate(patches):
            save_name = join(saveroot, 'p_'+img_name.replace('.JPG','')+'_%03d'%index + '.JPG')
            print(save_name)
            cv2.imwrite(save_name, patch)

def select(dataroot):
    save_dir = r'G:\Dataset\PAMIRain\ALL\selected_patches\clean'
    checkpoints = r'G:\Dataset\PAMIRain\ALL\selected_patches\clean\check.txt'
    img_list_file = r'G:\Dataset\PAMIRain\ALL\patches\clean_img_list.npy'
    # dataroot = r'G:\Dataset\PAMIRain\ALL\rainbuilding'
    if not os.path.exists(img_list_file):
        img_list = os.listdir(dataroot)
        print(len(img_list))
        np.save(img_list_file, img_list)

    os.makedirs(save_dir, exist_ok=True)
    img_list = np.load(img_list_file,allow_pickle=True)

    for name in img_list:
        img = cv2.imread(join(dataroot,name))
        cv2.imshow('img',img)
        key = cv2.waitKey()
        if(key == ord('s')):
            cv2.imwrite(join(save_dir,name),img)
            with open(checkpoints, 'w') as f:
                f.writelines(name)
        else:
            continue

def selectNoSelect():
    oneDir = r'G:\Dataset\PAMIRain\6_25图\Raw21_5\街景建筑'
    secDir = r'G:\Dataset\PAMIRain\ALL\rainbuilding'
    saveDir = r'G:\Dataset\PAMIRain\ALL\cleanscape3'
    one_list = os.listdir(oneDir)
    for name in one_list:
        sec_name = join(secDir, name)
        if not os.path.exists(sec_name):
            shutil.copy(join(oneDir, name), join(saveDir,name))

if __name__ == '__main__':
    # ====== crop ============
    dataroot = r'G:\Dataset\PAMIRain\ALL\cleanscape2'
    saveroot = r'G:\Dataset\PAMIRain\ALL\patches\cleanscape2'
    dataSetCrops(dataroot, saveroot)


    #======= select ==========
    # dataroot = r'G:\Dataset\PAMIRain\ALL\patches\cleanscape'
    # # saveroot = r'G:\Dataset\PAMIRain\ALL\patches\rainbuilding'
    # # os.makedirs(saveroot, exist_ok=True)
    # # dataSetCrops(dataroot,saveroot)
    # select(dataroot)

    # =======select No Select =========
    # selectNoSelect()