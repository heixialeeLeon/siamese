import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

class DistanceLoss(torch.nn.Module):
    def __init__(self, margin=1.0, p=2.0):
        super(DistanceLoss, self).__init__()
        self.margin = margin

    def forward(self, input, pos, neg):
        pos_distance = F.pairwise_distance(input,pos)
        neg_distance = F.pairwise_distance(input,neg)
        loss = torch.clamp(pos_distance - neg_distance+self.margin, min=0.0)
        return loss

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                                      (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

    def set_margin(self, margin_value):
        self.margin = margin_value


class ContrastiveLoss_2(torch.nn.Module):
    def __init__(self):
        super(ContrastiveLoss_2, self).__init__()
        self.linear1 = nn.Linear(256,128)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(128,1)
        self.wd = 0.1

        self.features = nn.Sequential(
            self.linear1,
            self.relu,
            self.linear2
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias,0)

    def forward(self, input1, input2, label):
        data1 = torch.cat((input1,input2),dim=1)
        data2 = torch.cat((input2,input1),dim=1)
        out1 = self.features(data1)
        out2 = self.features(data2)
        out = out1 + out2
        loss = torch.clamp((1-(2*label-1)*out), min=0.0)
        loss = loss.mean()

        l2_reg = 0
        for m in self.modules():
            if isinstance(m, nn.Linear):
                l2_reg += torch.mean(m.weight**2)
        loss += self.wd * l2_reg
        return loss

class ContrastiveLoss_3(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=0.8):
        super(ContrastiveLoss_3, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        euclidean_distance = F.normalize(euclidean_distance, dim = 0)
        loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                                      (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

    def set_margin(self, margin_value):
        self.margin = margin_value

if __name__ == "__main__":
    import numpy as np
    f1 = Variable(torch.randn(1,128))
    f2 = Variable(torch.randn(1,128))
    f3 = float(np.arange(0,0.9,0.1).mean())

    f1 = f1.cuda()
    f2 = f2.cuda()
    loss = ContrastiveLoss()
    out = loss(f1, f2, 0)
    print(out)
    out = loss(f1,f2, 1)
    print(out)
    loss.set_margin(f3)
    out = loss(f1,f2,1)
    print(out)