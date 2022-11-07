################################################################################
# MC-GAN
# Glyph Network Model
# By Samaneh Azadi
################################################################################

import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from scipy import misc
import random

from PIL import Image
import torchvision.transforms as transforms

class cGANModel(BaseModel):
    def name(self):
        return 'cGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # define tensors
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.fineSize, opt.fineSize)

        # load/define networks
        self.netG_3d = networks.define_G_3d(opt.input_nc, opt.input_nc, norm=opt.norm, groups=opt.grps, gpu_ids=self.gpu_ids)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                    opt.which_model_netG, opt.norm, opt.use_dropout, gpu_ids=self.gpu_ids)
        
        disc_ch = opt.input_nc
            
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.preNet_A = networks.define_preNet(disc_ch+disc_ch, disc_ch+disc_ch, which_model_preNet=opt.which_model_preNet,norm=opt.norm, gpu_ids=self.gpu_ids)
            nif = disc_ch+disc_ch
            netD_norm = opt.norm
            self.netD = networks.define_D(nif, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, netD_norm, use_sigmoid, gpu_ids=self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG_3d, 'G_3d', opt.which_epoch)
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.preNet_A, 'PRE_A', opt.which_epoch)
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()


            # initialize optimizers
            self.optimizer_G_3d = torch.optim.Adam(self.netG_3d.parameters(),
                                                     lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_preA = torch.optim.Adam(self.preNet_A.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            print('---------- Networks initialized -------------')
            if self.opt.conv3d:
                networks.print_network(self.netG_3d)
            networks.print_network(self.netG)
            if opt.which_model_preNet != 'none':
                networks.print_network(self.preNet_A)
            networks.print_network(self.netD)
            print('-----------------------------------------------')

        image=Image.open("./datasets/Capitals64/BASE/Code New Roman.0.0.png")
        image = image.convert("L")
        image=np.asarray(image)
        image=image.reshape((-1,64,1664))
        image=image/255

        image = torch.from_numpy(image)

        transform = transforms.Compose([
            transforms.Resize((64)),
            transforms.Normalize(( 0.5),
                                ( 0.5)),
            
            ])
        img_tensor = transform(image)
        img_tensor=torch.reshape(img_tensor,(1,26,64,64))        
        self.base_font=img_tensor.to("cuda:0")

    def set_input(self, input):
        input_A = input['A']
        input_B = input['B']        
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
    
        self.real_A_indep = self.netG_3d.forward(self.real_A.unsqueeze(2))
        self.fake_B = self.netG.forward(self.real_A_indep.squeeze(2))
    
        self.real_B = Variable(self.input_B)
        real_B = util.tensor2im(self.real_B.data)
        real_A = util.tensor2im(self.real_A.data)
    
    def add_noise_disc(self,real):
        #add noise to the discriminator target labels
        #real: True/False? 
        if self.opt.noisy_disc:
            rand_lbl = random.random()
            if rand_lbl<0.6:
                label = (not real)
            else:
                label = (real)
        else:  
            label = (real)
        return label
            
                

    
    # no backprop gradients
    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.real_A_indep = self.netG_3d.forward(self.real_A.unsqueeze(2))
        self.fake_B = self.netG.forward(self.real_A_indep.squeeze(2))
    
        self.real_B = Variable(self.input_B, volatile=True)

    #get image paths
    def get_image_paths(self):
        return self.image_paths

    
    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        label_fake = self.add_noise_disc(False)

        b,c,m,n = self.fake_B.size()
        rgb = 3 if self.opt.rgb else 1

        self.fake_B_reshaped = self.fake_B
        self.real_A_reshaped = self.real_A
        self.real_B_reshaped = self.real_B

        
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A_reshaped, self.fake_B_reshaped), 1))
        self.pred_fake_patch = self.netD.forward(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(self.pred_fake_patch, label_fake)
        #transform the input
        transformed_AB = self.preNet_A.forward(fake_AB.detach())
        self.pred_fake = self.netD.forward(transformed_AB)
        self.loss_D_fake += self.criterionGAN(self.pred_fake, label_fake)
                            
       
        # Real
        label_real = self.add_noise_disc(True)
        
        real_AB = torch.cat((self.real_A_reshaped, self.real_B_reshaped), 1)#.detach()
        self.pred_real_patch = self.netD.forward(real_AB)
        self.loss_D_real = self.criterionGAN(self.pred_real_patch, label_real)
        #transform the input
        transformed_A_real = self.preNet_A.forward(real_AB)
        self.pred_real = self.netD.forward(transformed_A_real)
        self.loss_D_real += self.criterionGAN(self.pred_real, label_real)
                            
        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()


    def backward_G(self):
        # First, G(A) should fake the discriminator
        #PATCH GAN
        fake_AB = (torch.cat((self.real_A_reshaped, self.fake_B_reshaped), 1))
        pred_fake_patch = self.netD.forward(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake_patch, True)
        #global disc
        transformed_A = self.preNet_A.forward(fake_AB)
        pred_fake = self.netD.forward(transformed_A)
        self.loss_G_GAN += self.criterionGAN(pred_fake, True)
 

        #self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * (self.opt.lambda_A / 100)
        img_tensor=torch.tile(self.base_font,(self.fake_B.size(dim=0),1,1,1))
        aux=torch.abs(torch.sub(self.real_B,img_tensor))
        aux1=torch.abs(torch.sub(self.fake_B,self.real_B))
        aux2=torch.mul(aux,aux1)
        self.loss_G_L1 = torch.mean(aux2)* (self.opt.lambda_A / 100)

        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.optimizer_preA.zero_grad()

        self.backward_D()

        self.optimizer_D.step()
        self.optimizer_preA.step()


        self.optimizer_G.zero_grad()
        self.optimizer_G_3d.zero_grad()

        self.backward_G()

        self.optimizer_G.step()
        self.optimizer_G_3d.step()
        
    

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.data),
                ('G_L1', self.loss_G_L1.data),
                ('D_real', self.loss_D_real.data),
                ('D_fake', self.loss_D_fake.data)
        ])


    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

    def save(self, label):
        self.save_network(self.netG_3d, 'G_3d', label, gpu_ids=self.gpu_ids)
        self.save_network(self.netG, 'G', label, gpu_ids=self.gpu_ids)
        self.save_network(self.netD, 'D', label, gpu_ids=self.gpu_ids)
        self.save_network(self.preNet_A, 'PRE_A', label, gpu_ids=self.gpu_ids)
            

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        
        for param_group in self.optimizer_preA.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        
        for param_group in self.optimizer_G_3d.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
