import cv2
import os
from skimage.measure import compare_psnr, compare_ssim
import fnmatch
import numpy as np

ubuntu = 0
windows = 1
platform = ubuntu

def crop_and_resize_square(img, tosize = 400):
    height, width = img.shape[0], img.shape[1]
    length = min(height, width)
    if len(img.shape) == 3:
        img_crop = img[0:length, 0:length, :]
    else:
        img_crop = img[0:length, 0:length]
    img_resize = cv2.resize(img_crop, (tosize, tosize))
    return img_resize


def Dataset_PSNR():
    Dataset_path = r'G:\Dataset\RainCycleGAN_dataset\cityRH'
    Train_Bs_list = os.listdir(os.path.join(Dataset_path, 'train','B_s'))
    Train_Os_list = os.listdir(os.path.join(Dataset_path, 'train','O_s'))
    Train_Bt_list = os.listdir(os.path.join(Dataset_path, 'train','B_t'))
    Train_Ot_list = os.listdir(os.path.join(Dataset_path, 'train','O_t'))
    Test_Bs_list = os.listdir(os.path.join(Dataset_path, 'test', 'B_s'))
    Test_Os_list = os.listdir(os.path.join(Dataset_path, 'test', 'O_s'))
    Test_Ot_list = os.listdir(os.path.join(Dataset_path, 'test', 'O_t'))
    Test_Bt_list = os.listdir(os.path.join(Dataset_path, 'test', 'B_t'))
    aver_psnr_train = 0
    aver_psnr_test = 0
    # count = 1
    # for train_name in Train_Bs_list:
    #     bs_file = cv2.imread(os.path.join(Dataset_path,'train','B_s',train_name))
    #     os_file = cv2.imread(os.path.join(Dataset_path,'train','O_s',train_name))
    #     tmp_psnr_bs = compare_psnr(bs_file, os_file)
    #     print('train %d/1400 tmp_psnr:'%count, tmp_psnr_bs)
    #     aver_psnr_train = (aver_psnr_train * count + tmp_psnr_bs) / (count + 1)
    #     count += 1
    count = 1
    for test_name in Test_Bt_list:
        bs_file = cv2.imread(os.path.join(Dataset_path, 'test', 'B_t', test_name))
        bs_file = crop_and_resize_square(bs_file, tosize=256)
        os_file = cv2.imread(os.path.join(Dataset_path, 'test', 'O_t', test_name))
        os_file = crop_and_resize_square(os_file, tosize=256)
        tmp_psnr_bs = compare_psnr(bs_file, os_file)
        print('test %d/175 tmp_psnr:'%count, tmp_psnr_bs)
        aver_psnr_test = (aver_psnr_test * count + tmp_psnr_bs) / (count + 1)
        count += 1
    print('train psnr: ', aver_psnr_train)
    print('test psnr: ', aver_psnr_test)


def Result_PSNR(experiment_name):
    if platform:
        result_path = r'G:\Projects\RainCycleGAN_cross\results'
        # result_path = r'D:\LowLevelforReal\RainCycleGAN_cross\results'
    else:
        # result_path = r'/media/solanliu/YYT2T/Projects/RainCycleGAN_cross/results'
        result_path = r'/home/solanliu/yeyuntong/RainCycleGAN_cross/results'
    image_path = os.path.join(result_path, experiment_name, 'test_latest', 'images')
    eval_file = os.path.join(result_path, experiment_name, 'test_latest','eval_file.txt')
    aver_psnr_bs = 0.
    aver_psnr_bt = 0.
    bs_list = os.listdir(image_path)
    bs_list = fnmatch.filter(bs_list, '*_pred_pred_Bs.png')
    count = 1
    for name in bs_list:
        little_name = name.replace('pred_pred_Bs.png', '')
        # print(little_name)
        pred_Bs_file = little_name + 'pred_Bs.png'
        Bs_file = little_name + 'Bs.png'
        pred_Bs = cv2.imread(os.path.join(image_path, pred_Bs_file))
        Bs = cv2.imread(os.path.join(image_path, Bs_file))
        try:
            tmp_psnr_bs = compare_psnr(pred_Bs, Bs)
            print('tmp_psnr:', tmp_psnr_bs)
            aver_psnr_bs = (aver_psnr_bs * count + tmp_psnr_bs) / (count + 1)
            count += 1
        except:
            print('pass:',pred_Bs_file)
            pass

    count = 0
    for name in bs_list:
        little_name = name.replace('pred_pred_Bs.png', '')
        # print(little_name)
        pred_Bt_file = little_name + 'pred_Bt.png'
        Bt_file = little_name + 'Bt.png'
        pred_Bt = cv2.imread(os.path.join(image_path, pred_Bt_file))
        Bt= cv2.imread(os.path.join(image_path, Bt_file))
        try:
            tmp_psnr_bt = compare_psnr(pred_Bt, Bt)
            print('tmp_psnr:', tmp_psnr_bt)
            aver_psnr_bt = (aver_psnr_bt * count + tmp_psnr_bt) / (count + 1)
            count += 1
        except:
            print('pass:', pred_Bt_file)
            pass
    print('average Bs PSNR: ', aver_psnr_bs)
    print('average Bt PSNR: ', aver_psnr_bt)
    with open(eval_file, 'w') as f:
        f.write('average Bs PSNR: %f\n'% aver_psnr_bs)
        f.write('average Bt PSNR: %f\n'% aver_psnr_bt)



def Result_SSIM(experiment_name):
    if platform:
        result_path = r'G:\Projects\RainCycleGAN_cross\results'
        # result_path = r'D:\LowLevelforReal\RainCycleGAN_cross\results'
    else:
        # result_path = r'/media/solanliu/YYT2T/Projects/RainCycleGAN_cross/results'
        result_path = r'/home/solanliu/yeyuntong/RainCycleGAN_cross/results'
    image_path = os.path.join(result_path, experiment_name, 'test_latest', 'images')
    eval_file = os.path.join(result_path, experiment_name, 'test_latest', 'eval_file.txt')
    aver_ssim_bs = 0.
    aver_ssim_bt = 0.
    test_list = os.listdir(image_path)
    test_list = fnmatch.filter(test_list, '*_pred_pred_Bs.png')
    count = 1
    for name in test_list:
        little_name = name.replace('pred_pred_Bs.png', '')
        # print(little_name)
        pred_Bs_file = little_name + 'pred_Bs.png'
        Bs_file = little_name + 'Bs.png'
        pred_Bs = cv2.imread(os.path.join(image_path, pred_Bs_file))
        Bs = cv2.imread(os.path.join(image_path, Bs_file))
        try:
            tmp_ssim_bs = compare_ssim(pred_Bs, Bs, multichannel=True)
            print('tmp_ssim:', tmp_ssim_bs)
            aver_ssim_bs = (aver_ssim_bs * count + tmp_ssim_bs) / (count + 1)
            count += 1
        except:
            pass

    count = 0
    for name in test_list:
        little_name = name.replace('pred_pred_Bs.png', '')
        # print(little_name)
        pred_Bt_file = little_name + 'pred_Bt.png'
        Bt_file = little_name + 'Bt.png'
        pred_Bt = cv2.imread(os.path.join(image_path, pred_Bt_file))
        Bt= cv2.imread(os.path.join(image_path, Bt_file))
        try:
            tmp_ssim_bt = compare_ssim(pred_Bt, Bt, multichannel=True)
            print('tmp_ssim:', tmp_ssim_bt)
            aver_ssim_bt = (aver_ssim_bt * count + tmp_ssim_bt) / (count + 1)
            count += 1
        except:
            pass
    print(result_path,':')
    print('average Bs ssim: ', aver_ssim_bs)
    print('average Bt ssim: ', aver_ssim_bt)
    with open(eval_file, 'w') as f:
        f.write('average Bs ssim: %f\n'% aver_ssim_bs)
        f.write('average Bt ssim: %f\n'% aver_ssim_bt)


# this is a wrong process, because the output has been clip, so we cannot directly get pred_Rs and pred_Rt by sub, we should output the Rs and Rt in test procedure.
# def See_rain(experiment_name, Syn):
#     if platform:
#         # result_path = r'G:\Projects\RainCycleGAN_cross\results'
#         result_path = r'D:\LowLevelforReal\RainCycleGAN_cross\results'
#     else:
#         result_path = r'/media/solanliu/YYT2T/Projects/RainCycleGAN_cross/results'
#     image_path = os.path.join(result_path, experiment_name, 'test_latest', 'images')
#     test_list = os.listdir(image_path)
#     test_list = fnmatch.filter(test_list, '*_pred_pred_Bs.png')
#     save_path = os.path.join(result_path, experiment_name, 'test_latest', 'Rs'if Syn else 'Rt')
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#     if Syn:
#         for name in test_list:
#             little_name = name.replace('pred_pred_Bs.png', '')
#             # print(little_name)
#             pred_Bs_file = little_name + 'pred_Bs.png'
#             pred_pred_Bs_file = little_name + 'pred_pred_Bs.png'
#             Os_file = little_name + 'Os.png'
#             pred_Bs = cv2.imread(os.path.join(image_path, pred_Bs_file))
#             pred_pred_Bs = cv2.imread(os.path.join(image_path, pred_pred_Bs_file))
#             Os = cv2.imread(os.path.join(image_path, Os_file))
#             pred_Rs = Os - pred_Bs
#             pred_pred_Rs = Os - pred_pred_Bs
#             np.clip(pred_Rs, 0, 255, pred_Rs)
#             np.clip(pred_pred_Rs, 0, 255, pred_pred_Rs)
#             cv2.imwrite(os.path.join(save_path,little_name+'pred_Rs.png'),pred_Rs)
#             cv2.imwrite(os.path.join(save_path,little_name+'pred_pred_Rs.png'),pred_pred_Rs)
#     else:
#         for name in test_list:
#             little_name = name.replace('pred_pred_Bs.png', '')
#             # print(little_name)
#             pred_Bt_file = little_name + 'pred_Bt.png'
#             pred_pred_Bt_file = little_name + 'pred_pred_Bt.png'
#             Ot_file = little_name + 'Ot.png'
#
#             pred_Bt = cv2.imread(os.path.join(image_path, pred_Bt_file))
#             pred_pred_Bt = cv2.imread(os.path.join(image_path, pred_pred_Bt_file))
#             Ot = cv2.imread(os.path.join(image_path, Ot_file))
#             pred_Rt = Ot - pred_Bt
#             pred_pred_Rt = Ot - pred_pred_Bt
#             pred_Rst = cv2.imread(os.path.join(image_path,little_name+'pred_Ot.png')) - cv2.imread(os.path.join(image_path,little_name+'pred_Bs.png'))
#             np.clip(pred_Rt,0,255,pred_Rt)
#             np.clip(pred_pred_Rt,0,255,pred_pred_Rt)
#             np.clip(pred_Rst, 0, 255, pred_Rst)
#             cv2.imwrite(os.path.join(save_path,little_name+'pred_Rt.png'),pred_Rt)
#             cv2.imwrite(os.path.join(save_path,little_name+'pred_pred_Rt.png'),pred_pred_Rt)
#             cv2.imwrite(os.path.join(save_path,little_name+'pred_Rst.png'),pred_Rst)
#




if __name__ == '__main__':
    Result_PSNR('lr_split_1e-4_50_4_40')
    # Result_SSIM()
    # Dataset_PSNR()