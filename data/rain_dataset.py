"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
from data.base_dataset import BaseDataset, get_transform
# from data.image_folder import make_dataset
from PIL import Image
import os
import torchvision.transforms as transforms
import random
import cv2

windows = 1
ubuntu = 0

platform = windows

class RainDataset(BaseDataset):
    '''
    dataset for RainCycleGAN
    Input: O_t, O_s, B_s, and may unaligned B_t

    The data structure should be:
    ---data_root
        ---O_t
        ---O_s
        ---B_s
    '''

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        # get the image paths of your dataset;
        # self.image_paths = []  # You can call sorted(make_dataset(self.root, opt.max_dataset_size)) to get all the image paths under the directory self.root
        BaseDataset.__init__(self, opt)
        self.Bt_access = opt.Bt_access
        self.data_root = opt.dataroot
        self.O_t_path = os.path.join(self.data_root,opt.phase,'Ot')
        self.O_s_path = os.path.join(self.data_root,opt.phase,'Os')
        self.B_s_path = os.path.join(self.data_root,opt.phase,'Bs')
        if self.Bt_access:
            self.B_t_path = os.path.join(self.data_root,opt.phase,'Bt')
            self.B_t_name_list = os.listdir(self.B_t_path)
        self.O_t_name_list = os.listdir(self.O_t_path)
        self.O_s_name_list = os.listdir(self.O_s_path)
        random.shuffle(self.O_t_name_list)
        random.shuffle(self.O_s_name_list)
        if self.Bt_access:
            random.shuffle(self.B_t_name_list)

        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        self.transform = transforms.Compose([
            # transforms.ToPILImage(),
            # transforms.RandomCrop(opt.crop_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        self.O_t_file = os.path.join(self.O_t_path, self.O_t_name_list[index])
        self.O_s_file = os.path.join(self.O_s_path, self.O_s_name_list[index])
        self.B_s_file = os.path.join(self.B_s_path, self.O_s_name_list[index])
        if self.Bt_access:
            self.B_t_file = os.path.join(self.B_t_path, self.O_t_name_list[index])

        # needs to be a string
        # O_t = Image.open(self.O_t_file).convert('RGB')
        # O_s = Image.open(self.O_s_file).convert('RGB')
        # B_s = Image.open(self.B_s_file).convert('RGB')
        O_t = cv2.imread(self.O_t_file)
        O_s = cv2.imread(self.O_s_file)
        B_s = cv2.imread(self.B_s_file)
        O_t = cv2.cvtColor(O_t, cv2.COLOR_BGR2RGB)
        O_s = cv2.cvtColor(O_s, cv2.COLOR_BGR2RGB)
        B_s = cv2.cvtColor(B_s, cv2.COLOR_BGR2RGB)
        if self.Bt_access:
            B_t = cv2.imread(self.B_t_file)
            B_t = cv2.cvtColor(B_t, cv2.COLOR_BGR2RGB)


        # crop O_s and B_s
        img_size = min(O_s.shape[0], O_s.shape[1])
        crop_size = self.opt.crop_size
        crop_x_loc = random.randint(0, img_size-crop_size)
        crop_y_loc = random.randint(0, img_size-crop_size)
        O_s = O_s[crop_x_loc:crop_x_loc + crop_size, crop_y_loc:crop_y_loc + crop_size, :]
        B_s = B_s[crop_x_loc:crop_x_loc + crop_size, crop_y_loc:crop_y_loc + crop_size, :]

        # crop O_t and B_t
        img_size = min(O_t.shape[0], O_t.shape[1])
        crop_size = self.opt.crop_size
        crop_x_loc = random.randint(0, img_size-crop_size)
        crop_y_loc = random.randint(0, img_size-crop_size)
        O_t = O_t[crop_x_loc:crop_x_loc + crop_size, crop_y_loc:crop_y_loc + crop_size, :]
        if self.Bt_access:
            B_t = B_t[crop_x_loc:crop_x_loc + crop_size, crop_y_loc:crop_y_loc + crop_size, :]


        # needs to be a tensor
        O_t = self.transform(O_t)
        O_s = self.transform(O_s)
        B_s = self.transform(B_s)
        if self.Bt_access:
            B_t = self.transform(B_t)
            return {
                'O_t': O_t,
                'O_s': O_s,
                'B_s': B_s,
                'B_t': B_t,
                'path': self.O_s_file
            }
        else:
            return {
                'O_t':O_t,
                'O_s':O_s,
                'B_s':B_s,
                'path':self.O_s_file
            }


    def __len__(self):
        """Return the total number of images."""
        length = min(len(self.O_s_name_list),len(self.O_t_name_list))
        return length
