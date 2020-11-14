import os
import cv2
import shutil
import numpy as np

target = 'Ot'
methods = ['Rain','DDN','JORDER-E','RESCAN','DAF','SSIR','Syn2Real','CycleGAN','JRRG']
# txt_file = r'E:\Projects\Compare_Methods\Cityscape_Results\For_compare\compare_list.txt'
txt_file = r'E:\Projects\Compare_Methods\SPA_Results\For_compare\compare_list.txt'


def write_compare_list():
    file_dir = r'E:\Projects\Compare_Methods\Cityscape_Results\For_compare\DDN\Ot_Pred'
    img_list = os.listdir(file_dir)
    with open(txt_file, 'w') as f:
        for img_name in img_list:
            f.write('%s\n'%img_name)

def pick_results_in_results():
    target_path = r'E:\Projects\Compare_Methods\SPA_Results\For_compare\JRRG\Ot_pred'
    source_path = r'E:\Projects\RainCycleGAN_cross\results\SPASR_UnetDerain\test_latest\images'
    with open(txt_file, 'r') as f:
        for line in f.readlines():
            little_name = line.strip().replace('.png','')
            shutil.copy(os.path.join(source_path,little_name+'_pred_Bs.png'), os.path.join(target_path,little_name+'.png'))

def resize_all():
    resize_path = r'E:\Dataset\Captured\SONY Cpature\20200915\resize_1000_600'
    img_list = os.listdir(resize_path)
    for img_name in img_list:
        img_file = os.path.join(resize_path,img_name)
        img = cv2.imread(img_file)
        img = cv2.resize(img,(1000,600))
        cv2.imwrite(img_file, img)
    # with open(txt_file, 'r') as f:
    #     for line in f.readlines():
    #         img_file = os.path.join(resize_path,line.strip())
    #         img = cv2.imread(img_file)
    #         img = cv2.resize(img,(400,400))
    #         cv2.imwrite(img_file, img)

def rename_JORDERE():
    results_path = r'E:\Projects\Compare_Methods\Cityscape_Results\For_compare\JORDER-E\Ot_Pred'
    img_list = os.listdir(results_path)
    for img_name in img_list:
        targe_name = img_name.replace('_x2_SR','')
        os.rename(os.path.join(results_path,img_name),os.path.join(results_path,targe_name))

def rename_RESCAN():
    results_path = r'E:\Projects\Compare_Methods\Cityscape_Results\For_compare\RESCAN\Ot_Pred'
    img_list = os.listdir(results_path)
    for img_name in img_list:
        targe_name = img_name.replace('_3.png','.png')
        os.rename(os.path.join(results_path,img_name),os.path.join(results_path,targe_name))


def check_RESCAN_result():
    results_path = r'E:\Projects\Compare_Methods\SPA_Results\For_compare\RESCAN\Ot_Pred'
    count = 0
    with open(txt_file, 'r') as f:
        for line in f.readlines():
            if not os.path.exists(os.path.join(results_path,line.strip())):
                little_name = line.strip().replace('_3.png','')
                little_name = little_name + '.png'
                target_name = line.strip()
                os.rename(os.path.join(results_path,little_name), os.path.join(results_path,target_name))
                print(line)
        print(count)



def visualize_comparison():
    compare_path = r'E:\Projects\Compare_Methods\SPA_Results\For_compare'
    with open(txt_file, 'r') as f:
        for line in f.readlines():
            img_list = []
            for i in range(len(methods)):
                img = cv2.imread(os.path.join(compare_path,methods[i],'Ot_pred',line.strip()))
                # img = cv2.resize(img,(300,150))
                if img is None:
                    print(methods[i],line)
                    break
                img_list.append(img)
                # print(img.shape)
            # cv2.imshow('img',img_list[0])
            # cv2.waitKey()
            print(len(img_list))
            row1 = np.concatenate((img_list[0],img_list[1],img_list[2],img_list[3],img_list[3]),axis=1)
            # print(row1.shape)
            row2 = np.concatenate((img_list[4],img_list[5],img_list[6],img_list[7],img_list[8]),axis=1)
            # print(row2.shape)
            show = np.concatenate((row1,row2), axis=0)
            print(methods, line)
            cv2.imshow('row1',show)
            key = cv2.waitKey()
            if key == ord('1'):
                break

def save_picked_case():
    compare_path = r'E:\Projects\Compare_Methods\SPA_Results\For_compare'
    save_path = r'E:\Research\2021 CVPR\Figures\Figure10-11\case'
    picked_name = ['023_9_2','098_3_2']
    for img_name in picked_name:
        for i in range(len(methods)):
            img = cv2.imread(os.path.join(compare_path, methods[i], 'Ot_pred', img_name+'.png'))
            save_name = '%s_'%methods[i]+ img_name+'.png'
            cv2.imwrite(os.path.join(save_path, save_name),img)





if __name__ == '__main__':
    # write_compare_list()
    resize_all()
    # pick_results_in_results()
    # rename_RESCAN()
    # rename_JORDERE()
    # check_RESCAN_result()

    # visualize_comparison()
    # save_picked_case()