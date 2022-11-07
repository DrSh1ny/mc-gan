#=============================
# MC-GAN
# Modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
# By Samaneh Azadi
#=============================



from data.image_folder import ImageFolder
from data.base_data_loader import BaseDataLoader
import random
import torch.utils.data
import torchvision.transforms as transforms
from builtins import object
import os
import numpy as np
from torch import LongTensor
import warnings
import pickle


def normalize_stack(input,val=0.5):
    #normalize an tensor with arbitrary number of channels:
    # each channel with mean=std=val
    val=0.5
    len_ = input.size(0)
    mean = [val,]*len_
    std = [val,]*len_
    t_normal_stack = transforms.Compose([
        transforms.Normalize(mean,std)])
    return t_normal_stack(input)

def CreateDataLoader(opt):
    data_loader = DataLoader()
    data_loader.initialize(opt)
    return data_loader

class Data(object):
    def __init__(self, data_loader, fineSize, max_dataset_size, rgb, dict_test={}, blanks=0.7):
        self.data_loader = data_loader
        self.fineSize = fineSize
        self.max_dataset_size = max_dataset_size
        self.rgb = rgb
        self.blanks = blanks
        self.random_dict=dict_test
        self.dict = False
        if len(self.random_dict.keys()):
            self.dict = True

        
    def __iter__(self):
        self.data_loader_iter = iter(self.data_loader)
        self.iter = 0
        return self

    def __next__(self):
        self.iter += 1
        if self.iter > self.max_dataset_size:
            raise StopIteration
        AB, AB_paths = next(self.data_loader_iter)
        w_total = AB.size(3)
        w = int(w_total / 2)
        h = AB.size(2)
        w_offset = random.randint(0, max(0, w - self.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.fineSize - 1))
        A = AB[:, :, h_offset:h_offset + self.fineSize,
               w_offset:w_offset + self.fineSize]
        B = AB[:, :, h_offset:h_offset + self.fineSize,
               w_offset: w_offset + self.fineSize]
        n_rgb = 3 if self.rgb else 1
        
        
        if self.blanks == 0:
            AA = A.clone()
        else: 
            #randomly remove some of the glyphs in input
            if not self.dict:
                blank_ind = np.repeat(np.random.permutation(A.size(1)//n_rgb)[0:int(self.blanks*A.size(1)//n_rgb)],n_rgb)
            else:
                file_name = map(lambda x:x.split("/")[-1],AB_paths)
                if len(file_name)>1:
                    raise Exception('batch size should be 1')
                file_name=file_name[0]
                blank_ind = self.random_dict[file_name][0:int(self.blanks*A.size(1)//n_rgb)]

            rgb_inds = np.tile(range(n_rgb),int(self.blanks*A.size(1)//n_rgb))
            blank_ind = blank_ind*n_rgb + rgb_inds
            AA = A.clone()
            AA.index_fill_(1,LongTensor(list(blank_ind)),1)
        

        return {'A': AA, 'A_paths': AB_paths, 'B':B, 'B_paths':AB_paths}

class DataLoader(BaseDataLoader):
    def initialize(self, opt):
        
        BaseDataLoader.initialize(self, opt)
        self.fineSize = opt.fineSize
        transform = transforms.Compose([
            # TODO: Scale
            transforms.Resize(opt.loadSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))])
        # Dataset A
        dataset = ImageFolder(root=opt.dataroot + opt.phase + '/',
                              transform=transform, return_paths=True, font_trans=(not opt.flat), rgb=opt.rgb,
                               fineSize=opt.fineSize, loadSize=opt.loadSize) 
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.opt.batchSize,
            shuffle=not self.opt.serial_batches,
            num_workers=int(self.opt.nThreads))
            
       
        self.dataset = dataset
        dict_inds = {}
        test_dict = opt.dataroot+'/test_dict/dict.pkl'
        if opt.phase=='test':
            if os.path.isfile(test_dict):
                dict_inds = pickle.load(open(test_dict))
            else:
                warnings.warn('Blanks in test data are random. create a pkl file in ~/data_path/test_dict/dict.pkl including predifined random indices')

        self._data = Data(data_loader, opt.fineSize, opt.max_dataset_size, opt.rgb, dict_inds, opt.blanks)

    def name(self):
        return 'DataLoader'

    def load_data(self):
        return self._data

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
