import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import torch.utils.model_zoo as model_zoo

from util import *
from deform import *

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
}

def block_function_factory(conv,norm,relu=None):
    def block_function(x):
        x = conv(x)
        if norm is not None:
            x = norm(x)
        if relu is not None:
            x = relu(x)
        return x
    return block_function

def do_efficient_fwd(block_f,x,efficient):
    if efficient and x.requires_grad:
        return cp.checkpoint(block_f,x)
    else:
        return block_f(x)

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self,in_c,out_c,stride=1,downsample = None,efficient=True,use_bn=True):
        super(BasicBlock, self).__init__()
        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(in_channels=in_c,out_channels=out_c,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_c) if self.use_bn else None
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_c,out_channels=out_c,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_c) if self.use_bn else None
        self.downsample = downsample
        self.stride = stride
        self.efficient = efficient

    def forward(self, x):
        residual = x

        if self.downsample is not None:
            residual = self.downsample(x)

        block_f1 = block_function_factory(self.conv1,self.bn1,self.relu)
        block_f2 = block_function_factory(self.conv2,self.bn2)

        out = do_efficient_fwd(block_f1,x,self.efficient)
        out = do_efficient_fwd(block_f2,out,self.efficient)

        out = out + residual
        relu_out = self.relu(out)

        return relu_out,out

class ResNet(nn.Module):

    def __init__(self, block, layers, efficient=False, use_bn=True, **kwargs):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.use_bn = use_bn
        self.efficient = efficient

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64) if self.use_bn else lambda x:x
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layers = [nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)]
            if self.use_bn:
                layers += [nn.BatchNorm2d(planes * block.expansion)]
            downsample = nn.Sequential(*layers)
        layers = [block(self.inplanes, planes, stride, downsample, efficient=self.efficient, use_bn=self.use_bn)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers += [block(self.inplanes, planes, efficient=self.efficient, use_bn=self.use_bn)]

        return nn.Sequential(*layers)

    def forward_resblock(self, x, layers):
        skip = None
        for l in layers:
            x = l(x)
            if isinstance(x, tuple):
                x, skip = x
        return x, skip

    def forward(self, image):

        x = self.conv1(image)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        features = []
        x, skip = self.forward_resblock(x, self.layer1)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer2)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer3)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer4)
        features += [skip]
        return features

class PMM(nn.Module):

    def __init__(self, channels = [64,128,256,512]):
        super(PMM, self).__init__()


        self.downsample1 = nn.Conv2d(in_channels=channels[0], out_channels = channels[1],kernel_size=3, stride=2,padding=1,bias=False)
        self.downsample2 = nn.Conv2d(in_channels=channels[1]*2, out_channels = channels[2],kernel_size=3, stride=2,padding=1,bias=False)
        self.downsample3 = nn.Conv2d(in_channels=channels[2]*2, out_channels = channels[3],kernel_size=3, stride=2,padding=1,bias=False)

        self.def1 = DDeform(channels[0],channels[0])
        self.def2 = DDeform(channels[1], channels[1])
        self.def3 = DDeform(channels[2], channels[2])
        self.def4 = DDeform(channels[3], channels[3])

    def forward(self, features):

        features_t0, features_t1 = features[:4], features[4:]

        fm1 = self.def1(features_t0[0])
        attention1 = features_t1[0] - fm1
        fm2 = self.def2(features_t0[1])
        attention2 = features_t1[1] - fm2
        fm3 = self.def3(features_t0[2])
        attention3 = features_t1[2] - fm3
        fm4 = self.def4(features_t0[3])
        attention4 = features_t1[3] - fm4

        downsampled_attention1 = self.downsample1(attention1)
        cat_attention2 = torch.cat([downsampled_attention1,attention2], 1)
        downsampled_attention2 = self.downsample2(cat_attention2)
        cat_attention3 = torch.cat([downsampled_attention2,attention3], 1)
        downsampled_attention3 = self.downsample3(cat_attention3)
        final_attention_map = torch.cat([downsampled_attention3,attention4], 1)

        features_map = [final_attention_map,attention4,attention3,attention2,attention1]
        return features_map


class Decoder(nn.Module):
    
    def __init__(self,channels=[64,128,256,512]):
        super(Decoder, self).__init__()
        self.upsample1 = Upsample(num_maps_in=channels[3]*2, skip_maps_in=channels[3], num_maps_out=channels[3])
        self.upsample2 = Upsample(num_maps_in=channels[2]*2, skip_maps_in=channels[2], num_maps_out=channels[2])
        self.upsample3 = Upsample(num_maps_in=channels[1]*2, skip_maps_in=channels[1], num_maps_out=channels[1])
        self.upsample4 = Upsample(num_maps_in=channels[0]*2, skip_maps_in=channels[0], num_maps_out=channels[0])

    def forward(self, feutures_map):
        
        x = feutures_map[0]
        x = self.upsample1(x, feutures_map[1])
        x = self.upsample2(x, feutures_map[2])
        x = self.upsample3(x, feutures_map[3])
        x = self.upsample4(x, feutures_map[4])
        return x


def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    channels = [64,128,256,512]
    return model,channels

def get_encoder(arch,pretrained=True):
    if arch == 'resnet18':
        return resnet18(pretrained)
    else:
        print('Arch error.')
        exit(-1)

def achieve_pmm(channels=[64,128,256,512]):
    return PMM(channels=channels)
def get_decoder(channels=[64,128,256,512]):
    return Decoder(channels=channels)

