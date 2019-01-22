import os
import math
import csv
import random
import numpy as np
from PIL import Image
import json
import torch
from torch.utils import data
import torchvision
from torchvision import transforms

default_transform = transforms.Compose([
    transforms.ToTensor(),
])


class PairDataTriple(data.Dataset):
    def __init__(self, root, list_file, pos_file, neg_file, transform=default_transform, has_class=False):
        self.data = []
        self.root = os.path.expanduser(root)
        self.list = [line.strip().split() for line in open(list_file).readlines()]
        self.list_pos = [line.strip().split() for line in open(pos_file).readlines()]
        self.list_neg = [line.strip().split() for line in open(neg_file).readlines()]
        assert(len(self.list) == len(self.list_pos))
        assert(len(self.list_neg) == len(self.list_pos))
        self.transform = transform
        self.has_class = has_class

    def __getitem__(self, index):
        item = self.list[index]
        item_pos = self.list_pos[index]
        item_neg = self.list_neg[index]

        image = Image.open(os.path.join(self.root, item[0]))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)

        image_pos = Image.open(os.path.join(self.root, item_pos[0]))
        if image_pos.mode != 'RGB':
            image_pos = image_pos.convert('RGB')
        if self.transform:
            image_pos = self.transform(image_pos)

        image_neg = Image.open(os.path.join(self.root, item_neg[0]))
        if image_neg.mode != 'RGB':
            image_neg = image_neg.convert('RGB')
        if self.transform:
            image_neg = self.transform(image_neg)

        if self.has_class:
            label = int(item[1])
            label_pos= int(item_pos[1])
            label_neg = int(item_neg[1])
            return image, image_pos, image_neg, label, label_pos, label_neg
        else:
            return image, image_pos, image_neg

    def __len__(self):
        return len(self.list)
        #return int(len(self.list1)/10)

class PairData_Test(data.Dataset):
    def __init__(self, root, list_file1, list_file2, transform=default_transform, has_class=False):
        self.data = []
        self.root = os.path.expanduser(root)
        self.list1 = [line.strip().split() for line in open(list_file1).readlines()]
        self.list2 = [line.strip().split() for line in open(list_file2).readlines()]
        assert(len(self.list1) == len(self.list2))
        self.transform = transform
        self.has_class = has_class

    def __getitem__(self, index):
        item1 = self.list1[index]
        item2 = self.list2[index]

        image1 = Image.open(os.path.join(self.root, item1[0]))
        if image1.mode != 'RGB':
            image1 = image1.convert('RGB')
        if self.transform:
            image1 = self.transform(image1)

        image2 = Image.open(os.path.join(self.root, item2[0]))
        if image2.mode != 'RGB':
            image2 = image2.convert('RGB')
        if self.transform:
            image2 = self.transform(image2)

        pair = int(item1[1])

        if self.has_class:
            label1 = int(item1[2])
            label2 = int(item2[2])
            return image1, image2, pair, label1, label2
        else:
            return image1, image2, pair

    def __len__(self):
        return len(self.list1)
        #return int(len(self.list1)/10)

if __name__ == "__main__":
    data = PairDataTriple("/data_1/data/signatureCompare/gaochao/data_1224/train",
                          "/data_1/data/signatureCompare/gaochao/data_1224/train_triplet_a.list",
                          "/data_1/data/signatureCompare/gaochao/data_1224/train_triplet_p.list",
                          "/data_1/data/signatureCompare/gaochao/data_1224/train_triplet_n.list",
                    transform=default_transform, has_class=True)
    for item in data:
        print(item[0].shape)
        print(item[1].shape)
        print(item[2].shape)
        print(item[3])
        print(item[4])
        print(item[5])

