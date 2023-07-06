import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from os.path import join as pjoin, splitext as spt


def check_validness(f):
    return any([i in spt(f)[1] for i in ['jpg','png']])

class get_data(Dataset):

    def __init__(self, root):
        super(get_data, self).__init__()
        self.img_t0_root = pjoin(root, 'A')
        self.img_t1_root = pjoin(root, 'B')
        self.img_mask_root = pjoin(root, 'label')
        self.filename = list(spt(f)[0] for f in os.listdir(self.img_mask_root) if check_validness(f))
        self.filename.sort()

    def __getitem__(self, index):

        fn = self.filename[index]
        fn_t0 = pjoin(self.img_t0_root, fn + '.png')###LEVIR->png,CDD->jpg
        fn_t1 = pjoin(self.img_t1_root, fn + '.png')
        fn_mask = pjoin(self.img_mask_root, fn + '.png')

        if os.path.isfile(fn_t0) == False:
            print('Read error' + fn_t0)
            exit(-1)
        if os.path.isfile(fn_t1) == False:
            print('Read error' + fn_t1)
            exit(-1)
        if os.path.isfile(fn_mask) == False:
            print('Read error' + fn_mask)
            exit(-1)

        img_t0 = cv2.imread(fn_t0, 1)
        img_t1 = cv2.imread(fn_t1, 1)
        mask = cv2.imread(fn_mask, 0)

        mask_r = mask[:, :, np.newaxis]

        img_t0_r = np.asarray(img_t0).astype('f').transpose(2, 0, 1) / 128.0 - 1.0
        img_t1_r = np.asarray(img_t1).astype('f').transpose(2, 0, 1) / 128.0 - 1.0
        mask_r_ = np.asarray(mask_r > 128).astype('f').transpose(2, 0, 1)
        input_ = torch.from_numpy(np.concatenate((img_t0_r, img_t1_r), axis=0))
        mask_ = torch.from_numpy(mask_r_).long()

        return input_, mask_

    def __len__(self):
        return len(self.filename)

    def get_random_image(self):
        idx = np.random.randint(0,len(self))
        return self.__getitem__(idx)

class get_testdata(Dataset):

    def __init__(self, root):
        super(get_testdata, self).__init__()
        self.img_t0_root = pjoin(root, 'A')
        self.img_t1_root = pjoin(root, 'B')
        self.img_mask_root = pjoin(root, 'label')
        self.filename = list(spt(f)[0] for f in os.listdir(self.img_mask_root) if check_validness(f))
        self.filename.sort()

    def __getitem__(self, index):

        fn = self.filename[index]
        fn_t0 = pjoin(self.img_t0_root, fn + '.png')
        fn_t1 = pjoin(self.img_t1_root, fn + '.png')
        fn_mask = pjoin(self.img_mask_root, fn + '.png')

        if os.path.isfile(fn_t0) == False:
            print('Read error' + fn_t0)
            exit(-1)
        if os.path.isfile(fn_t1) == False:
            print('Read error' + fn_t1)
            exit(-1)
        if os.path.isfile(fn_mask) == False:
            print('Read error' + fn_mask)
            exit(-1)

        img_t0 = cv2.imread(fn_t0, 1)
        img_t1 = cv2.imread(fn_t1, 1)
        mask = cv2.imread(fn_mask, 0)

        w, h, c = img_t0.shape
        w_r = int(256 * max(w / 256, 1))
        h_r = int(256 * max(h / 256, 1))

        img_t0_r = cv2.resize(img_t0, (h_r, w_r))
        img_t1_r = cv2.resize(img_t1, (h_r, w_r))
        mask_r = cv2.resize(mask, (h_r, w_r))[:, :, np.newaxis]

        img_t0_r_ = np.asarray(img_t0_r).astype('f').transpose(2, 0, 1) / 128.0 - 1.0
        img_t1_r_ = np.asarray(img_t1_r).astype('f').transpose(2, 0, 1) / 128.0 - 1.0
        mask_r_ = np.asarray(mask_r > 128).astype('f').transpose(2, 0, 1)

        return img_t0_r_, img_t1_r_, mask_r_, w, h, w_r, h_r

    def __len__(self):
        return len(self.filename)

    def get_random_image(self):
        idx = np.random.randint(0,len(self))
        return self.__getitem__(idx)

    def name(self):
        return self.filename






