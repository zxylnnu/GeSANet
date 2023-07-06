import os
from os.path import join as pjoin
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import datasets
from GeSANet import *

class criterion_CEloss(nn.Module):

    def __init__(self,weight=None):
        super(criterion_CEloss, self).__init__()
        self.loss = nn.NLLLoss(weight)
    def forward(self,output,target):
        return self.loss(F.log_softmax(output, dim=1), target)

class Train:

    def __init__(self):
        self.epoch = 0
        self.step = 0

    def train(self):

        weight = torch.ones(2)
        criterion = criterion_CEloss(weight.cuda())
        optimizer = torch.optim.Adam(self.model.parameters(),lr=0.001,betas=(0.9,0.999))
        lambda_lr = lambda epoch:(float)(self.args.max_epochs*len(self.dataset_train_loader)-self.step)/(float)(self.args.max_epochs*len(self.dataset_train_loader))
        model_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda_lr)

        loss_item = []

        while self.epoch < self.args.max_epochs:

            for step,(inputs_train,mask_train) in enumerate(tqdm(self.dataset_train_loader)):
                self.model.train()
                inputs_train = inputs_train.cuda()
                mask_train = mask_train.cuda()
                output_train = self.model(inputs_train)
                optimizer.zero_grad()
                self.loss = criterion(output_train, mask_train[:,0])
                loss_item.append(self.loss)
                self.loss.backward()
                optimizer.step()
                self.step += 1

                # If you need, add here.
                # if self.args.step_test>0 and self.step % self.args.step_test == 0:
                #     print('val')
                #     self.model.eval()

            print('Loss for Epoch {}:{:.07f}'.format(self.epoch, sum(loss_item)/len(self.dataset_train_loader)))
            loss_item.clear()
            model_lr_scheduler.step()
            self.epoch += 1
            if self.args.epoch_save>0 and self.epoch % self.args.epoch_save == 0:
                self.checkpoint()

    def checkpoint(self):

        filename = '{:08d}.pth'.format(self.step)
        cp_path = pjoin(self.checkpoint_save)
        if not os.path.exists(cp_path):
            os.makedirs(cp_path)
        torch.save(self.model.state_dict(),pjoin(cp_path,filename))
        print("Save checkpoint--{:08d}".format(self.step))

    def run(self):

        self.model = GeSANet(self.args.encoder_arch).cuda()
        self.train()

class train_dataset(Train):

    def __init__(self, arguments):
        super(train_dataset, self).__init__()
        self.args = arguments

    def Init(self):

        folder_name = 'GeSANet'
        self.dataset_train_loader = DataLoader(datasets.get_data(pjoin(self.args.datadir, "train")),
                                          num_workers=0, batch_size=self.args.batch_size,
                                          shuffle=True)
        # self.dataset_val = datasets.get_data(pjoin(self.args.datadir, 'val'))
        self.checkpoint_save = pjoin(self.args.checkpointdir, folder_name, self.args.dataset)
        if not os.path.exists(self.checkpoint_save):
            os.makedirs(self.checkpoint_save)


if __name__ =="__main__":
    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--datadir', required=True)

    parser.add_argument('--checkpointdir', type=str, default='./checkpoint')
    parser.add_argument('--encoder-arch', type=str, default='resnet18')

    parser.add_argument('--max-epochs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epoch-save', type=int, default=2)
    # parser.add_argument('--step-test', type=int, default=2)


    if parser.parse_args().dataset == 'levir' or parser.parse_args().dataset == 'cdd' :
        train = train_dataset(parser.parse_args())
        train.Init()
        train.run()
    else:
        print('Dataset error.')
        exit(-1)