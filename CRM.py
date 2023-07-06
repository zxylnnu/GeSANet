import torch
import torch.nn as nn

class get_lr(nn.Module):
    def __init__(self, dim=64):
        super(get_lr, self).__init__()
        self.dim = dim
        convc, convh, convw = self.gen_lr(64)

        self.convc = convc
        self.convh = convh
        self.convw = convw

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        b, c, height, width = x.size()
        C = self.pool(x)
        H = self.pool(x.permute(0, 3, 1, 2).contiguous())
        W = self.pool(x.permute(0, 2, 3, 1).contiguous())

        list = []
        for i in range(0, self.dim):
            list.append(self.recon(b, 64, self.convc[i](C), self.convh[i](H), self.convw[i](W)))

        LR = sum(list)
        return LR

    def gen_lr(self, dim=64):
        conv1 = []
        for _ in range(0, dim):
            conv1.append(nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=1, bias=False, groups=1),
                nn.Sigmoid(),
            ))
        conv1 = nn.ModuleList(conv1)

        conv2 = []
        for _ in range(0, dim):
            conv2.append(nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=1, bias=False, groups=1),
                nn.Sigmoid(),
            ))
        conv2 = nn.ModuleList(conv2)

        conv3 = []
        for _ in range(0, dim):
            conv3.append(nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=1, bias=False, groups=1),
                nn.Sigmoid(),
            ))
        conv3 = nn.ModuleList(conv3)

        return conv1, conv2, conv3

    def recon(self, batch_size, dim, feat, feat2, feat3):
        b = batch_size
        C = feat.view(b, -1, 1)
        H = feat2.view(b, 1, -1)
        W = feat3.view(b, 1 * 1, -1)
        reco = torch.bmm(torch.bmm(C, H).view(b, -1, 1 * 1), W).view(b, -1, dim, dim)
        return reco

class achieve_crm(nn.Module):
    def __init__(self, dim=64):
        super(achieve_crm, self).__init__()

        self.dim = dim
        self.lr = get_lr(dim)
        self.update = nn.Sequential(
            nn.Conv2d(64, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout2d(0.05, False),
        )

    def forward(self, x):
        lrs = self.lr(x)
        hrs = self.update(lrs)
        return hrs
