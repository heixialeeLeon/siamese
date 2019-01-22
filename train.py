import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.nn import functional as F
import numpy as np

from model.siamesResnet import *
from model.siamesSimple import *
from dataset import *
from ContrastiveLoss import ContrastiveLoss
from tools import EuclideanDistance

parser = argparse.ArgumentParser(description='PyTorch Siamese')
# Optimization options
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--lr','--learning-rate', default=0.01, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--use_cuda', default=True, type=bool,
                    help='Cuda Option')
parser.add_argument('--steps-loss', type=int, default=100)
parser.add_argument('--save_folder', default='weights',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()

train_dataset = PairData("/data_1/data/signatureCompare/1206new", "/data_1/data/signatureCompare/shuffle_train_label.txt", "/data_1/data/signatureCompare/shuffle_train_p_label.txt")
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

test_dataset = PairData_Test("/data_1/data/signatureCompare/1206new", "/data_1/data/signatureCompare/test2584.txt", "/data_1/data/signatureCompare/test2584_p.txt")
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

criterion = ContrastiveLoss(margin=1.0)
#net = siamesResnet(3)
net = SiameseSimple()
if (args.use_cuda):
    net = net.cuda()
# optimizer = torch.optim.SGD(net.parameters(), args.lr)
optimizer = torch.optim.Adam(net.parameters(), args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)

def save_model(net,epoch):
    save_fileName = "{}/model_{}.pth".format(args.save_folder,epoch)
    torch.save(net, save_fileName)

def load_model(fileName):
    net = torch.load(fileName)
    return net

def train_epoch(net, epoch):
    epoch_loss= []
    distance_total = np.empty(shape=(0,1))
    label_total = np.empty(shape=(0,1))
    for index, (data1,data2,label) in enumerate(train_dataloader):
        # data prepare
        if(args.use_cuda):
            data1 = data1.cuda()
            data2 = data2.cuda()
            label = label.cuda()

        data1 = Variable(data1)
        data2 = Variable(data2)
        label = Variable(label)

        # forward
        out1, out2 = net(data1, data2)
        loss = criterion(out1, out2, label.float())
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # distance record
        distance = F.pairwise_distance(out1, out2)
        distance_total = np.append(distance_total, distance.data.cpu().numpy())
        label_total = np.append(label_total,label.data.cpu().numpy())

        epoch_loss.append(loss.item())
        if args.steps_loss > 0 and index % args.steps_loss == 0:
            print(f'epoch : {epoch}  {len(train_dataloader)}:{index}  loss:{loss.item():.6f}')
    print(f'epoch {epoch}  average loss: {sum(epoch_loss)/len(epoch_loss)}')
    scheduler.step(epoch)

    # distance compute
    distance_total.reshape(-1,1)
    label_total.reshape(-1,1)
    pos_index = label_total == 1
    neg_index = label_total == 0
    pos_mean_distance = distance_total[pos_index].mean()
    pos_min_distance = np.min(distance_total[pos_index])
    pos_max_distance = np.max(distance_total[pos_index])
    neg_mean_distance = distance_total[neg_index].mean()
    neg_min_distance = np.min(distance_total[neg_index])
    neg_max_distance = np.max(distance_total[neg_index])
    print("postive mean distance: {}, min: {}, max: {}".format(pos_mean_distance, pos_min_distance, pos_max_distance))
    print("negitve mean distance: {}, min: {}, max: {}".format(neg_mean_distance, neg_min_distance, neg_max_distance))

    margin = float(0.5*(pos_mean_distance + neg_mean_distance))
    # if epoch % 2 == 0:
    #     margin= 1.0
    # else:
    #     margin = max(0.1, margin)
    margin = max(0.1, margin)
    criterion.set_margin(margin)
    print("set margin to : {}".format(margin))

def test(net):
    import numpy as np
    v1 = np.empty(shape=(0,128))
    v2 = np.empty(shape=(0,128))
    gt = np.empty(shape=(0,))
    epoch_loss = []
    for index, (data1, data2, label) in enumerate(test_dataloader):
        # data prepare
        if (args.use_cuda):
            data1 = data1.cuda()
            data2 = data2.cuda()
            label = label.cuda()

        data1 = Variable(data1)
        data2 = Variable(data2)
        label = Variable(label)

        # forward
        out1, out2 = net(data1, data2)
        loss = criterion(out1, out2, label.float())
        epoch_loss.append(loss.item())
        v1 = np.append(v1,out1.data.cpu().numpy())
        v2 = np.append(v2,out2.data.cpu().numpy())
        gt = np.append(gt,label.data.cpu().numpy())
    print(f'validate average loss: {sum(epoch_loss)/len(epoch_loss)}')

    v1 = v1.reshape(-1,128)
    v2 = v2.reshape(-1,128)
    distance = EuclideanDistance(v1,v2)
    threshold_step = np.arange(0,np.max(distance),0.01)

    max_score = 0.0
    best_threshold = 0
    for threshold in threshold_step :
        same_index = (distance < threshold).astype(np.int)
        right = (same_index == gt).sum()
        score = float(right) / float(len(gt))
        if score > max_score:
            max_score = score
            best_threshold = threshold
    print("best score: {}, best threshold: {}".format(max_score, best_threshold))



def train():
    for epoch in range(0, args.epochs):
        train_epoch(net, epoch)
        save_model(net, epoch)
        test(net)

if __name__ == "__main__":
    train()
    # net = load_model("weights/model_0.pth")
    # net.eval()
    # test(net)