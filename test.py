import datasets
from GeSANet import GeSANet
import os
# import csv
import cv2
import torch
import torch.nn as nn
import numpy as np
from os.path import join as pjoin
from tqdm import tqdm
import torch.nn.functional as F
import argparse
import time
class Evaluate:

    def __init__(self):
        self.args = None
        self.set = None

    def eval(self):

        input = torch.from_numpy(np.concatenate((self.t0,self.t1),axis=0)).contiguous()
        input = input.view(1,-1,self.w_r,self.h_r)
        input = input.cuda()

        output= self.model(input)


        input = input[0].cpu().data
        img_t0 = input[0:3,:,:]
        img_t1 = input[3:6,:,:]
        img_t0 = (img_t0+1)*128
        img_t1 = (img_t1+1)*128
        output = output[0].cpu().data

        mask_pred = np.where(F.softmax(output[0:2,:,:],dim=0)[0]>0.5, 255, 0)
        mask_gt = np.squeeze(np.where(self.mask==True,255,0),axis=0)
        precision, recall, accuracy, f1_score = self.store_imgs_and_cal_matrics(img_t0,img_t1,mask_gt,mask_pred)

        return (precision, recall, accuracy, f1_score)


    def store_imgs_and_cal_matrics(self, t0, t1, mask_gt, mask_pred):

        w, h = self.w_r, self.h_r
        img_save = np.zeros((w , h), dtype=np.uint8)
        # img_save[0:w, 0:h, :] = np.transpose(t0.numpy(), (1, 2, 0)).astype(np.uint8)
        # img_save[0:w, h:h * 2, :] = np.transpose(t1.numpy(), (1, 2, 0)).astype(np.uint8)
        # img_save[w:w * 2, 0:h, :] = cv2.cvtColor(mask_gt.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        img_save = cv2.cvtColor(mask_pred.astype(np.uint8), cv2.COLOR_GRAY2RGB)

        if w != self.w_ori or h != self.h_ori:
            img_save = cv2.resize(img_save, (self.h_ori, self.w_ori))

        fn_save = self.fn_img
        if not os.path.exists(self.dir_img):
            os.makedirs(self.dir_img)

        print('Save' + fn_save)
        cv2.imwrite(fn_save, img_save)
        precision, recall, accuracy, f1_score = self.cal_metrcis(mask_pred,mask_gt)
        return (precision, recall, accuracy, f1_score)

    def cal_metrcis(self,pred,target):

        temp = np.dstack((pred == 0, target == 0))
        TP = sum(sum(np.all(temp,axis=2)))

        temp = np.dstack((pred == 0, target == 255))
        FP = sum(sum(np.all(temp,axis=2)))

        temp = np.dstack((pred == 255, target == 0))
        FN = sum(sum(np.all(temp, axis=2)))

        temp = np.dstack((pred == 255, target == 255))
        TN = sum(sum(np.all(temp, axis=2)))

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        accuracy = (TP + TN) / (TP + FP + FN + TN)
        f1_score = 2 * recall * precision / (precision + recall)

        return (precision, recall, accuracy, f1_score)

    def Init(self):

        model_name = 'GeSANet'
        self.resultdir = pjoin(self.args.resultdir, model_name, self.args.dataset)
        if not os.path.exists(self.resultdir):
            os.makedirs(self.resultdir)

    def run(self):

        if os.path.isfile(self.fn_model) is False:
            print("Read error" + self.fn_model)
            exit(-1)
        else:
            print("Read" + self.fn_model)

        self.model = GeSANet(self.args.encoder_arch)
        self.model.load_state_dict(torch.load(self.fn_model))
        self.model = self.model.cuda()
        self.model.eval()

class test_dataset(Evaluate):

    def __init__(self, arguments):
        super(test_dataset, self).__init__()
        self.args = arguments

    def Init(self):
        super(test_dataset,self).Init()
        self.ds = 1
        self.index = 0
        self.dir_img = pjoin(self.resultdir)
        self.fn_model = pjoin(self.args.checkpointdir,'GeSANet', self.args.dataset,'00000004.pth')

    def test(self):

        input = torch.from_numpy(np.concatenate((self.t0,self.t1),axis=0)).contiguous()
        input = input.view(1,-1,self.w_r,self.h_r)
        input = input.cuda()

        output= self.model(input)

        input = input[0].cpu().data
        img_t0 = input[0:3,:,:]
        img_t1 = input[3:6,:,:]
        img_t0 = (img_t0+1)*128
        img_t1 = (img_t1+1)*128
        output = output[0].cpu().data
        mask_pred = np.where(F.softmax(output[0:2,:,:],dim=0)[0]>0.5, 0, 255)
        mask_gt = np.squeeze(np.where(self.mask==True,255,0),axis=0)
        precision, recall, accuracy, f1_score = self.store_imgs_and_cal_matrics(img_t0,img_t1,mask_gt,mask_pred)

        return (precision, recall, accuracy, f1_score)

    def run(self):
        super(test_dataset, self).run()

        img_cnt = 0
        metrics = np.array([0, 0, 0, 0], dtype='float64')
        for idx in range(0, 1):
            test_loader = datasets.get_testdata(pjoin(self.args.datadir))
            img_cnt += len(test_loader)
            self.ds = idx
            for i in range(0, len(test_loader)):
                self.index = i
                self.fn_img = pjoin(self.dir_img, '{0:04d}.png'.format(self.index))
                self.t0, self.t1, self.mask, self.w_ori, self.h_ori, self.w_r, self.h_r = test_loader[i]
                metrics += np.array(self.test())


if __name__ =='__main__':

    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--datadir', required=True)
    parser.add_argument('--resultdir',type=str, default='./result')
    parser.add_argument('--checkpointdir',type=str, default='./checkpoint')
    parser.add_argument('--encoder-arch',  type=str, default='resnet18')

    if parser.parse_args().dataset == 'levir' or parser.parse_args().dataset == 'cdd':
        test = test_dataset(parser.parse_args())
        test.Init()
        test.run()
    else:
        print('Dataset error.')
        exit(-1)