"""Model class template

This module provides a template for users to implement custom models.
You can specify '--model template' to use this model.
The class name should be consistent with both the filename and its model option.
The filename should be <model>_dataset.py
The class name should be <Model>Dataset.py
It implements a simple image-to-image translation baseline based on regression loss.
Given input-output pairs (data_A, data_B), it learns a network netG that can minimize the following L1 loss:
    min_<netG> ||netG(data_A) - data_B||_1
You need to implement the following functions:
    <modify_commandline_options>:　Add model-specific options and rewrite default values for existing options.
    <__init__>: Initialize this model class.
    <set_input>: Unpack input data and perform data pre-processing.
    <forward>: Run forward pass. This will be called by both <optimize_parameters> and <test>.
    <optimize_parameters>: Update network weights; it will be called in every training iteration.
"""
import torch
from .base_model import BaseModel
from . import networks
import itertools
from util.image_pool import ImagePool
import os


# current train strategy: fix G1 and train G2, G3, G4 initial G3 with G1. data: 2020.9.18

class RainCycleModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(dataset_mode='rain')  # You can rewrite default values for this model. For example, this model usually uses aligned dataset as its dataset.
        if is_train:
            parser.add_argument('--lambda_MSE', type=float, default=40.0, help='weight for the mse loss')  # You can define new arguments for this model.
            parser.add_argument('--lambda_GAN', type=float, default=4.0, help='weight for the gan loss')
            parser.add_argument('--lambda_Cycle', type=float, default=40.0, help='weight for the cycle loss')
            parser.add_argument('--lambda_Idt', type=float, default=20.0, help='weight for the identity loss')
            parser.add_argument('--init_derain', type=str, default='1,3', help='weight for the identity loss')

        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        # specify the training losses you want to print out. The program will call base_model.get_current_losses to plot the losses to the console and save them to the disk.
        self.loss_names = ['MSE','GAN','Cycle','G_total','D_Ot', 'D_Os','D_B']
        # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.
        if self.isTrain:
            self.visual_names = ['Os','Ot','Bs',
                                 'pred_Bs','pred_Ot','pred_pred_Bs','pred_pred_Os',
                                 'pred_Bt','pred_Os','pred_pred_Bt','pred_pred_Ot',
                                 'pred_Rs', 'pred_Rst', 'pred_pred_Rt', 'pred_pred_Rts',
                                 'pred_Rt', 'pred_Rts', 'pred_pred_Rs', 'pred_pred_Rst']
        else:
            self.visual_names = ['Os', 'Ot', 'Bs', 'Bt', 'Rs', 'Rt',
                                 'pred_Bs', 'pred_Ot', 'pred_pred_Bs', 'pred_pred_Os',
                                 'pred_Bt', 'pred_Os', 'pred_pred_Bt', 'pred_pred_Ot',
                                 'pred_Rs', 'pred_Rst', 'pred_pred_Rt', 'pred_pred_Rts',
                                 'pred_Rt', 'pred_Rts', 'pred_pred_Rs', 'pred_pred_Rst']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks to save and load networks.
        self.model_names = ['G1', 'G2', 'G3', 'G4', 'D_B', 'D_Os', 'D_Ot']


        # define networks; you can use opt.isTrain to specify different behaviors for training and test.
        self.netG2 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, gpu_ids=self.gpu_ids)
        self.netG4 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, gpu_ids=self.gpu_ids)
        # self.netG1 = networks.SingleDNCNN(channels=3)
        # self.netG3 = networks.SingleDNCNN(channels=3)
        # self.netG1.to(self.device)
        # self.netG1 = torch.nn.DataParallel(self.netG1, self.gpu_ids)
        # self.netG3.to(self.device)
        # self.netG3 = torch.nn.DataParallel(self.netG3, self.gpu_ids)
        self.netG1 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'unet_128', gpu_ids=self.gpu_ids)
        self.netG3 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'unet_128', gpu_ids=self.gpu_ids)


        self.netD_Ot = networks.define_D(opt.output_nc, opt.ndf, opt.netD,opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netD_Os = networks.define_D(opt.output_nc, opt.ndf, opt.netD,opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netD_B = networks.define_D(opt.output_nc, opt.ndf, opt.netD,opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        self.feature_dim = networks.DnCNN(channels=3).out_feature_dim
        self.netD_feature = networks.define_D(self.feature_dim, opt.ndf, opt.netD,
                                              opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # only defined during training time
            # -----------load pre-trained model for G1 and G3--------------
            # self.netG1_model = self.netG1.module
            # self.netG3_model = self.netG3.module
            # load_filename = 'latest_net_DNCNN.pth'
            # if self.opt.singleDNCNN_load_path == None:
            #     print('FileNotFoundError: singleDNCNN_load_path is not found!')
            #     raise FileNotFoundError
            # load_path = os.path.join(self.opt.singleDNCNN_load_path, load_filename)
            # save_model = torch.load(load_path, map_location=self.device)
            # SingleDncnn_dict = {}
            # for k, v in save_model.items():
            #     if 'dncnn_front_1' in k:
            #         SingleDncnn_dict[k.replace('dncnn_front_1', 'dncnn_front')] = v
            # for k, v in save_model.items():
            #     if 'dncnn_back' in k:
            #         SingleDncnn_dict[k] = v
            # model_dict1 = self.netG1_model.state_dict()
            # model_dict1.update(SingleDncnn_dict)
            # self.netG1_model.load_state_dict(model_dict1)
            #
            # model_dict2 = self.netG3_model.state_dict()
            # model_dict2.update(SingleDncnn_dict)
            # self.netG3_model.load_state_dict(model_dict2)
            # # self.netDNCNN_model.fix_parameters()
            # print('initial netG1, netG3 with %s successfully!' % load_path)
            # self.fix_parameters()
            if self.opt.init_derain != '0':
                load_filename = 'latest_net_UnetDerain.pth'
                if self.opt.singleDNCNN_load_path == None:
                    print('FileNotFoundError: singleDNCNN_load_path is not found!')
                    raise FileNotFoundError
                load_path = os.path.join(self.opt.singleDNCNN_load_path, load_filename)
                state_dict = torch.load(load_path, map_location=str(self.device))
                if '1' in self.opt.init_derain:
                    model_G1 = self.netG1.module
                    model_G1.load_state_dict(state_dict)
                if '3' in self.opt.init_derain:
                    model_G3 = self.netG3.module
                    model_G3.load_state_dict(state_dict)
                    print('initial netG1, netG3 with %s successfully!' % load_path)
            # for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
            #     self.patch_instance_norm_state_dict(state_dict, self.model_G3, key.split('.'))
            # self.model_G3.load_state_dict(state_dict)
            # self.netG1.eval()

            # define your loss functions. You can use losses provided by torch.nn such as torch.nn.L1Loss.
            # We also provide a GANLoss class "networks.GANLoss". self.criterionGAN = networks.GANLoss().to(self.device)
            self.criterion_MSE = torch.nn.MSELoss()
            self.criterion_GAN = self.criterion_GAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()


            self.Pool_fake_Os = ImagePool(opt.pool_size)
            self.Pool_fake_Ot = ImagePool(opt.pool_size)
            self.Pool_pred_Bs = ImagePool(opt.pool_size)
            self.Pool_pred_pred_Bs = ImagePool(opt.pool_size)
            self.Pool_pred_B_t = ImagePool(opt.pool_size)
            self.Pool_pred_pred_B_t = ImagePool(opt.pool_size)
            self.Pool_real_B = ImagePool(opt.pool_size)

            # define and initialize optimizers. You can define one optimizer for each network.
            # If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
            if '1' in self.opt.init_derain:
                parameters_list=[dict(params=self.netG1.parameters(), lr=opt.lr / 100)]
            else:
                parameters_list = [dict(params=self.netG1.parameters(), lr=opt.lr)]
            if '3' in self.opt.init_derain:
                parameters_list.append(dict(params=self.netG3.parameters(), lr=opt.lr / 10))
            else:
                parameters_list.append(dict(params=self.netG3.parameters(), lr=opt.lr))
            parameters_list.append(dict(params=self.netG2.parameters(), lr=opt.lr))
            parameters_list.append(dict(params=self.netG4.parameters(), lr=opt.lr))
            self.optimizer_G = torch.optim.Adam(parameters_list, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_Os = torch.optim.Adam(self.netD_Os.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_Ot = torch.optim.Adam(self.netD_Ot.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers = [self.optimizer_G, self.optimizer_D_Os, self.optimizer_D_Ot, self.optimizer_B]
        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks
        else:
            self.visual_names.append('Bt')

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.Os = input['O_s'].to(self.device)
        self.Bs = input['B_s'].to(self.device)
        self.Ot = input['O_t'].to(self.device)
        if not self.isTrain:
            self.Bt = input['B_t'].to(self.device)
        # self.Bt = input['B_t'].to(self.device)
        self.image_paths = input['path']  # for test

    def fix_parameters(self):
        self.netG1.eval()
        print('fix netG1')

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        if not self.isTrain:
            self.Rs = self.Os - self.Bs
            self.Rt = self.Ot - self.Bt
        # syn - real
        self.pred_Bs = self.netG1(self.Os)
        self.pred_Rs = self.Os - self.pred_Bs
        self.pred_Rst = self.netG2(self.pred_Rs)
        self.pred_Ot = self.pred_Bs + self.pred_Rst
        # syn - real - syn
        self.pred_pred_Bs = self.netG3(self.pred_Ot)
        self.pred_pred_Rt = self.pred_Ot - self.pred_pred_Bs
        self.pred_pred_Rts =  self.netG4(self.pred_pred_Rt)
        self.pred_pred_Os = self.pred_pred_Bs + self.pred_pred_Rts
        # real - syn
        self.pred_Bt = self.netG3(self.Ot)
        self.pred_Rt = self.Ot - self.pred_Bt
        self.pred_Rts = self.netG4(self.pred_Rt)
        self.pred_Os = self.pred_Bt + self.pred_Rts
        # real - syn - real
        self.pred_pred_Bt = self.netG1(self.pred_Os)
        self.pred_pred_Rs = self.pred_Os - self.pred_pred_Bt
        self.pred_pred_Rst = self.netG2(self.pred_pred_Rs)
        self.pred_pred_Ot = self.pred_pred_Bt + self.pred_pred_Rst

    def backward_G(self):
        lambda_MSE = self.opt.lambda_MSE
        lambda_GAN = self.opt.lambda_GAN
        lambda_Cycle = self.opt.lambda_Cycle
        lambda_Idt = self.opt.lambda_Idt
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results

        # # Identity Loss
        # if lambda_Idt >0:
        #     self.Idt_Bs = self.netG1(self.Bs)


        # Cycle Loss
        self.Cycle_Os = self.criterionCycle(self.Os, self.pred_pred_Os)
        self.Cycle_Ot = self.criterionCycle(self.Ot, self.pred_pred_Ot)
        self.Cycle_Bs = self.criterionCycle(self.pred_Bs, self.pred_pred_Bs)
        self.Cycle_Bt = self.criterionCycle(self.pred_Bt, self.pred_pred_Bt)
        self.loss_Cycle = self.Cycle_Os + self.Cycle_Ot + self.Cycle_Bs + self.Cycle_Bt

        # GAN Loss
        self.GAN_Ot = self.criterion_GAN(self.netD_Ot(self.pred_Ot), True)
        self.GAN_Os = self.criterion_GAN(self.netD_Os(self.pred_Os), True)
        self.GAN_pred_Bs = self.criterion_GAN(self.netD_B(self.pred_Bs), True)
        self.GAN_pred_pred_Bs = self.criterion_GAN(self.netD_B(self.pred_pred_Bs), True)
        self.GAN_pred_Bt = self.criterion_GAN(self.netD_B(self.pred_Bt), True)
        self.GAN_pred_pred_Bt = self.criterion_GAN(self.netD_B(self.pred_pred_Bt), True)
        self.loss_GAN = self.GAN_Ot + self.GAN_Os + self.GAN_pred_Bs + self.GAN_pred_pred_Bs + self.GAN_pred_Bt + self.GAN_pred_pred_Bt

        # MSE Loss
        self.MSE_pred_Bs = self.criterion_MSE(self.pred_Bs, self.Bs)
        self.MSE_pred_pred_Bs = self.criterion_MSE(self.pred_pred_Bs, self.Bs)
        self.loss_MSE = self.MSE_pred_Bs + self.MSE_pred_pred_Bs

        self.loss_G_total = lambda_MSE * self.loss_MSE +  lambda_GAN * self.loss_GAN + lambda_Cycle * self.loss_Cycle
        self.loss_G_total.backward()

    def cal_GAN_loss_D_basic(self, netD, real, fake):
        pred_real = netD(real)
        loss_D_real = self.criterion_GAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterion_GAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        return loss_D

    def backward_D_Ot(self):
        fake_Ot = self.Pool_fake_Ot.query(self.pred_Ot)
        self.loss_D_Ot = self.cal_GAN_loss_D_basic(self.netD_Ot, self.Ot, fake_Ot)
        self.loss_D_Ot.backward()


    def backward_D_Os(self):
        fake_Os = self.Pool_fake_Os.query(self.pred_Os)
        self.loss_D_Os = self.cal_GAN_loss_D_basic(self.netD_Os, self.Os, fake_Os)
        self.loss_D_Os.backward()

    def backward_D_B(self):
        pred_Bs = self.Pool_pred_Bs.query(self.pred_Bs)
        pred_pred_Bs = self.Pool_pred_pred_Bs.query(self.pred_pred_Bs)
        pred_Bt = self.Pool_pred_B_t.query(self.pred_Bt)
        pred_pred_Bt =self.Pool_pred_pred_B_t.query(self.pred_pred_Bt)
        real_images = self.Pool_real_B.query(self.Bs)
        real_images = self.Pool_real_B.query(return_num=4)
        self.loss_D_B = self.cal_GAN_loss_D_basic(self.netD_B, real_images[0], pred_Bs) + self.cal_GAN_loss_D_basic(self.netD_B, real_images[1], pred_pred_Bs) + self.cal_GAN_loss_D_basic(self.netD_B, real_images[2],pred_Bt) + self.cal_GAN_loss_D_basic(self.netD_B, real_images[3], pred_pred_Bt)
        self.loss_D_B.backward()


    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()               # first call forward to calculate intermediate results

        self.set_requires_grad([self.netD_B, self.netD_Ot, self.netD_Os], False)
        self.optimizer_G.zero_grad()   # clear network G's existing gradients
        self.backward_G()              # calculate gradients for network G
        self.optimizer_G.step()        # update gradients for network G

        self.set_requires_grad([self.netD_B, self.netD_Ot, self.netD_Os], True)
        self.optimizer_B.zero_grad()
        self.backward_D_B()
        self.optimizer_B.step()

        self.optimizer_D_Ot.zero_grad()
        self.backward_D_Ot()
        self.optimizer_D_Ot.step()

        self.optimizer_D_Os.zero_grad()
        self.backward_D_Os()
        self.optimizer_D_Os.step()



