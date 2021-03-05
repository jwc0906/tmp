from __future__ import print_function

import os
import sys
import argparse
import time
import math

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from torch.utils.data import Dataset, DataLoader


import numpy as np

from util import TwoCropTransform

def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        opt.size= 32
        
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        opt.size= 32
        
    elif opt.dataset == 'SVHN':
        mean= (0.5, 0.5, 0.5)
        std= (0.5, 0.5, 0.5)
        opt.size= 32
        
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.mean)
        
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=TwoCropTransform(train_transform),
                                         download=True)
        opt.class_num= 10	

    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=TwoCropTransform(train_transform),
                                          download=True)
        opt.class_num= 100

    elif opt.dataset == 'SVHN':
        train_dataset= datasets.SVHN(root=opt.data_folder, transform=TwoCropTransform(train_transform), download=True)
        opt.class_num= 10

    elif opt.dataset == 'path':
        train_dataset = datasets.ImageFolder(root=opt.data_folder,
                                            transform=TwoCropTransform(train_transform))
    else:
        raise ValueError(opt.dataset)
    
    ### imbalance ratio

    imbalance_ratio=opt.imbalance_ratio
    class_num=opt.class_num
    imbalance_r= math.pow(imbalance_ratio, 1/(class_num-1))
    total_data_num=opt.total_data_num

    if imbalance_ratio==1:
        labelNum= [total_data_num//class_num]*class_num
    else:
        initial=0
        totals=[]
        total=0
        while True:
            preLabelNum=[]
            initial+=0.0001
            total=0
            for i in range(class_num):
                item= initial*math.pow(imbalance_r, i)
                total+=item
                preLabelNum.append(item)
            labelNum=[round(i) for i in preLabelNum]
            if sum(labelNum)>=total_data_num:
                break

        
        if opt.imbalance_order=="descent":
            labelNum.reverse()
    
    print("[Imbalance ratio]", imbalance_ratio)
    print(labelNum)
    print(sum(labelNum))
    
    if opt.dataset=='SVHN':
        labels= np.array(train_dataset.labels)
    else:
        labels= np.array(train_dataset.targets)

    lst = []
    top=0
    for i in range(class_num):
        idx= np.arange(len(labels))[labels==i]
        selectedIdx = np.random.choice(idx, labelNum[i], replace=False).tolist()
        lst += selectedIdx
        top+=labelNum[i]

    lst= np.array(lst)
    np.save(os.path.join(opt.model_path, opt.model_name)+"/index.npy", lst)


    if opt.dataset=='SVHN':
        labels= np.array(train_dataset.labels)
    else:
        labels= np.array(train_dataset.targets)

    train_dataset.data= train_dataset.data[lst]

    if opt.dataset=='SVHN':
        train_dataset.labels= labels[lst].tolist()
    else:
        train_dataset.targets= labels[lst].tolist()


    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)


    return train_loader