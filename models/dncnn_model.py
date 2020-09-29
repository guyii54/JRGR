#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time : 20-9-16 下午4:17
# @Author : liziqin
import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import itertools
import os


# Domain adaptation in DNCNN
class DNCNNModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_MSE', type=float, default=10.0, help='weight for MSE loss (B_s)')
            parser.add_argument('--lambda_feature_GAN', type=float, default=0.0, help='weight for MSE loss (B_s)')
            parser.add_argument('--lambda_B_GAN', type=float, default=1.0, help='weight for MSE loss (B_s)')
            # parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            # parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            # parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.loss_names = ['DNCNN' ,'MSE', 'GAN', 'D_feature','D_B']
        self.visual_names = ['pred_B_s','O_s','B_s','O_t','B_t','pred_B_t']
        self.model_names = ['DNCNN','D_feature','D_B']
        # if self.isTrain:
        self.feature_dim = networks.DnCNN(channels=3).out_feature_dim

        self.netDNCNN = networks.DnCNN(channels=3)
        self.netDNCNN = networks.init_net(self.netDNCNN, opt.init_type, opt.init_gain, opt.gpu_ids)

        self.netD_feature = networks.define_D(self.feature_dim, opt.ndf, opt.netD,
                                              opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netD_B = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                        opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if isinstance(self.netDNCNN, torch.nn.DataParallel):
                self.netDNCNN_model = self.netDNCNN.module
            load_filename = 'latest_net_DNCNN.pth'
            load_path = os.path.join(self.opt.singleDNCNN_load_path, load_filename)
            save_model = torch.load(load_path, map_location=self.device)
            front1_dict = {}
            front2_dict = {}
            for k, v in save_model.items():
                if 'dncnn_front_1' in k:
                    front1_dict[k] = v
                    front2_dict[k.replace('dncnn_front_1', 'dncnn_front_2')] = v
            model_dict1 = self.netDNCNN_model.state_dict()
            model_dict1.update(front1_dict)
            self.netDNCNN_model.load_state_dict(model_dict1)
            model_dict2 = self.netDNCNN_model.state_dict()
            model_dict2.update(front2_dict)
            self.netDNCNN_model.load_state_dict(model_dict2)
            # self.netDNCNN_model.fix_parameters()
            print('initial DNCNN with %s successfully!' % load_path)


            # optimize all parameters together
            self.O_s_feature_pool = ImagePool(opt.pool_size)
            self.O_t_feature_pool = ImagePool(opt.pool_size)
            self.pred_B_s_pool = ImagePool(opt.pool_size)
            self.pred_B_t_pool = ImagePool(opt.pool_size)

            self.optimizer_D_feature = torch.optim.Adam(itertools.chain(self.netD_feature.parameters()), lr=opt.lr,betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(itertools.chain(self.netD_B.parameters()), lr=opt.lr,betas=(opt.beta1, 0.999))
            self.optimizer_DnCNN = torch.optim.Adam(itertools.chain(self.netDNCNN.parameters()), lr=opt.lr,betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_D_feature)
            self.optimizers.append(self.optimizer_D_B)
            self.optimizers.append(self.optimizer_DnCNN)
            self.criterion_GAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterion_B_s = torch.nn.MSELoss()


    def set_input(self, input):
        self.O_s = input['O_s'].to(self.device)
        self.B_s = input['B_s'].to(self.device)
        self.O_t = input['O_t'].to(self.device)
        self.B_t = input['B_t'].to(self.device)
        self.image_paths = input['path']  # for test

    def test(self):
        self.pred_O_s_feature, self.pred_O_t_feature, self.pred_B_s, self.pred_B_t = self.netDNCNN(self.O_s, self.O_t)

    def forward(self):
        pass
        # self.loss_FeatureDomain = self.criterion_FeatureDomain(self.pred_O_s_feature,self.pred_O_t_feature)

    def backward_DnCNN(self):
        lambda_MSE = self.opt.lambda_MSE
        lambda_feature = self.opt.lambda_feature_GAN
        lambda_B = self.opt.lambda_B_GAN
        self.pred_O_s_feature, self.pred_O_t_feature, self.pred_B_s, self.pred_B_t = self.netDNCNN(self.O_s, self.O_t)
        self.loss_MSE = self.criterion_B_s(self.pred_B_s,self.B_s)
        self.loss_GAN_O_s = self.criterion_GAN(self.netD_feature(self.pred_O_s_feature), False)
        self.loss_GAN_O_t = self.criterion_GAN(self.netD_feature(self.pred_O_t_feature), True)
        self.loss_GAN_B_s = self.criterion_GAN(self.netD_B(self.pred_B_s), False)
        self.loss_GAN_B_t = self.criterion_GAN(self.netD_B(self.pred_B_t), True)
        self.loss_GAN = lambda_feature * (self.loss_GAN_O_t + self.loss_GAN_O_s) + lambda_B * (self.loss_GAN_B_s + self.loss_GAN_B_t)
        self.loss_DNCNN = lambda_MSE * self.loss_MSE + self.loss_GAN
        self.loss_DNCNN.backward()


    def backward_D_feature(self):
        O_s_feature = self.O_s_feature_pool.query(self.pred_O_s_feature)
        O_t_feature = self.O_t_feature_pool.query(self.pred_O_t_feature)
        self.O_s_domain = self.netD_feature(O_s_feature)
        self.O_t_domain = self.netD_feature(O_t_feature)
        self.loss_D_O_s = self.criterion_GAN(self.O_s_domain, True)
        self.loss_D_O_t = self.criterion_GAN(self.O_s_domain, False)
        self.loss_D_feature = self.loss_D_O_s + self.loss_D_O_t
        self.loss_D_feature.backward()

    def backward_D_B(self):
        pred_B_s = self.pred_B_s_pool.query(self.pred_B_s)
        pred_B_t = self.pred_B_t_pool.query(self.pred_B_t)
        self.pred_B_s_domain = self.netD_B(pred_B_s)
        self.real_B_s = self.netD_B(self.B_s)
        self.real_B_t = self.netD_B(self.B_t)
        self.pred_B_t_domain = self.netD_B(pred_B_t)
        self.loss_D_B = self.criterion_GAN(self.pred_B_s_domain, False) \
                        + self.criterion_GAN(self.pred_B_t_domain, False) \
                        + self.criterion_GAN(self.real_B_s, True) \
                        + self.criterion_GAN(self.real_B_t, True)

        self.loss_D_B.backward()


    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad([self.netD_B,self.netD_feature], False)
        self.optimizer_DnCNN.zero_grad()
        self.backward_DnCNN()
        self.optimizer_DnCNN.step()

        self.set_requires_grad([self.netD_B, self.netD_feature], True)
        self.optimizer_D_feature.zero_grad()
        self.backward_D_feature()
        self.optimizer_D_feature.step()

        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()




