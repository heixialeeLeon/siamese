import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

class TripleDistanceLoss(torch.nn.Module):
    def __init__(self, margin=1.0, p=2.0):
        super(TripleDistanceLoss, self).__init__()
        self.margin = margin

    def forward(self, input, pos, neg):
        pos_distance = F.pairwise_distance(input,pos)
        neg_distance = F.pairwise_distance(input,neg)
        loss = torch.clamp(pos_distance - neg_distance+self.margin, min=0.0)
        return loss.mean()

class TripleContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0, p=2.0):
        super(TripleContrastiveLoss, self).__init__()
        self.margin = margin
        self.p = p

    def forward(self, input, pos, neg):
        loss_1 = self.contrastiveLoss(input, pos, 1)
        loss_2 = self.contrastiveLoss(input,neg, 0)
        loss_3 = self.contrastiveLoss(pos, neg, 0)
        return loss_1 + loss_2 + loss_3

    def contrastiveLoss(self, input1, input2, label):
        euclidean_distance = F.pairwise_distance(input1, input2, p = self.p)
        loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                                    (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

if __name__ == "__main__":
    data = Variable(torch.randn(3,128))
    data_pos = Variable(torch.randn(3, 128))
    data_neg = Variable(torch.randn(3, 128))

    loss_func = TripleDistanceLoss()
    print(loss_func(data,data_pos,data_neg))

    loss_func = nn.TripletMarginLoss(margin=1.0, p=2)
    print(loss_func(data,data_pos,data_neg))

    loss_func=TripleContrastiveLoss()
    print(loss_func(data,data_pos,data_neg))