#-*-coding:utf-8-*-
import os.path
import random
import torchvision.transforms as transforms
import torch
import random
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import numpy as np

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.dir_A = opt.dataroot
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.dir_B = opt.depth_gt_root
        self.B_paths = sorted(make_dataset(self.dir_B))
        if self.opt.offline_loading_mask:
            self.mask_folder = self.opt.training_mask_folder if self.opt.isTrain else self.opt.testing_mask_folder
            self.mask_paths = sorted(make_dataset(self.mask_folder))

        assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]

        self.transform_A = transforms.Compose(transform_list)

        transform_list = [transforms.ToTensor()]

        self.transform_B = transforms.Compose(transform_list)

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        A = Image.open(A_path).resize((448,448)).convert('RGB')
        w, h = A.size
        A = np.array(A)

        ### A is rgb_gt, B is depth_gt
        B_path = self.B_paths[index]
        # print("a_path:",A_path,"b_path",B_path)
        B = np.load(B_path)
        A[B==0] = 0
        A = Image.fromarray(A)
        B = Image.fromarray(B)
        # B = Image.open(B_path).resize((448,448)).convert('RGB')

        # if w < h:
        #     ht_1 = self.opt.loadSize * h // w
        #     wd_1 = self.opt.loadSize
        #     A = A.resize((wd_1, ht_1), Image.BICUBIC)
        # else:
        #     wd_1 = self.opt.loadSize * w // h
        #     ht_1 = self.opt.loadSize
        #     A = A.resize((wd_1, ht_1), Image.BICUBIC)

        A = self.transform_A(A)
        B = self.transform_B(B)
        h = A.size(1)
        w = A.size(2)
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        A = A[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]
        B = B[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]

        # print("aligneddataset",B.shape)
        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)] # size(2)-1, size(2)-2, ... , 0
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)

        # let B directly equals A
        # B = A.clone()

        # Just zero the mask is fine if not offline_loading_mask.
        mask = A.clone().zero_()
        if self.opt.offline_loading_mask:
            mask = Image.open(self.mask_paths[random.randint(0, len(self.mask_paths)-1)])
            # mask = mask.resize((self.opt.fineSize, self.opt.fineSize), Image.NEAREST)
            mask = transforms.ToTensor()(mask)
        
        return {'A': A, 'B': B, 'M': mask,
                'A_paths': A_path, 'B_path': B_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDataset'
