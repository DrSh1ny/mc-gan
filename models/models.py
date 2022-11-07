################################################################################
# MC-GAN
# Modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
# By Samaneh Azadi
################################################################################
from .cGAN_model import cGANModel

def create_model(opt):
    model = None
    print(opt.model)
    model = cGANModel()
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
