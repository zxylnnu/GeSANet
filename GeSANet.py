import torch
import torch.nn as nn
from util import upsample
from Gparts import *
from CRM import *

class GeSANet(nn.Module):

    def __init__(self, encoder_arch):
        super(GeSANet, self).__init__()

        self.encoder1, channels = get_encoder(encoder_arch,pretrained=True)
        self.encoder2, _ = get_encoder(encoder_arch,pretrained=True)
        self.pmm = achieve_pmm(channels)
        self.decoder = get_decoder(channels=channels)
        self.classifier = nn.Conv2d(channels[0], 2, 1, padding=0, stride=1)
        self.bn = nn.BatchNorm2d(channels[0])
        self.relu = nn.ReLU(inplace=True)

        self.crm = achieve_crm(64)

    def forward(self, img):

        img_t0,img_t1 = torch.split(img,3,1)
        features_t0 = self.encoder1(img_t0)
        features_t1 = self.encoder2(img_t1)
        features = features_t0 + features_t1
        features_map = self.pmm(features)
        pred1 = self.decoder(features_map)

        pred1_ = upsample(pred1,[pred1.size()[2]*2, pred1.size()[3]*2])
        pred1_ = self.bn(pred1_)
        pred1_ = upsample(pred1_,[pred1_.size()[2]*2, pred1_.size()[3]*2])
        pred1_ = self.relu(pred1_)
        pre_1 = self.classifier(pred1_)

        reco = self.crm(pred1)
        pred2 = pred1 + reco

        pred2_ = upsample(pred2,[pred2.size()[2]*2, pred2.size()[3]*2])
        pred2_ = self.bn(pred2_)
        pred2_ = upsample(pred2_,[pred2_.size()[2]*2, pred2_.size()[3]*2])
        pred2_ = self.relu(pred2_)
        pre_2 = self.classifier(pred2_) + pre_1

        return pre_2


