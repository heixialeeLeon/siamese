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


class PairData(data.Dataset):
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
    data = PairData("/data_1/data/signatureCompare/1206new", "/data_1/data/signatureCompare/shuffle_train_label.txt", "/data_1/data/signatureCompare/shuffle_train_p_label.txt", transform=default_transform)
    for item in data:
        print(item[0].shape)
        print(item[1].shape)

