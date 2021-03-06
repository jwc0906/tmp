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

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet
from losses import SupConLoss

import numpy as np


try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'SVHN', "MNIST","VECTOR", 'path'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    #imbalance setting
    parser.add_argument('--imbalance_ratio', type=float, help='imbalance_ratio')
    parser.add_argument('--class_num', type=int, help='imbalance_ratio')
    parser.add_argument('--total_data_num', type=int, help='imbalance_ratio')

    #aug
    parser.add_argument('--aug', type=str, default='simclr')
    parser.add_argument('--aug_std', type=float)


    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate
    
    opt.model_name= '{}_ir_{}_td_{}_aug_{}_AUGSTD_{}'.format(opt.model_name, opt.imbalance_ratio, opt.total_data_num, opt.aug, opt.aug_std)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt



class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
    def __call__(self, tensor):
        rnd= torch.randn(tensor.size())
        ### maskting ###
        #rnd[0,0,0]=0
        #rnd[0,0,1]=0
        #rnd[0,0,2]=0
        return tensor + rnd * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)




# make vector dataset
DIM=1*28*28
DATA_NUM=10000
NUM_EACH_CLASS= [9000, 910,  90]
CLASS_NUM=3

DATA_STD=0.2

MU_C= np.zeros((CLASS_NUM,DIM))
for i in range(CLASS_NUM):
        MU_C[i,i]=1


def normal_nd(num, dim, mu_list, std):
    result= np.zeros((num, dim))
    for i in range(dim):
        result[:,i]= np.random.normal(mu_list[i], std, num)
    return result


class CustomDataset(Dataset):
    def __init__(self, transform, opt):
        # dataset
        self.transform= transform
        data_x= np.zeros((DATA_NUM, DIM))
        data_y= np.zeros(DATA_NUM)

        num_c_top=0
        for i in range(CLASS_NUM):
            data_x[num_c_top:num_c_top+ NUM_EACH_CLASS[i]]= normal_nd(num= NUM_EACH_CLASS[i], dim= DIM, mu_list=MU_C[i], std=DATA_STD)
            data_y[num_c_top:num_c_top+ NUM_EACH_CLASS[i]]= i
            num_c_top+=NUM_EACH_CLASS[i]

        # self.data_x= torch.Tensor(data_x)
        # self.data_y= torch.Tensor(data_y)
        self.data_x=data_x.reshape((-1,28,28,1)).astype(np.float32)
        self.data_y=data_y
        _min= self.data_x.min()
        _max= self.data_x.max()
        interval= _max - _min
        self.data_x=(((self.data_x - _min)/interval)*254).astype(np.uint8)
        

        #save
        np.save(os.path.join(opt.model_path, opt.model_name)+"/data_x.npy", self.data_x)
        np.save(os.path.join(opt.model_path, opt.model_name)+"/data_y.npy", self.data_y)

    def __len__(self):
        return self.data_x.shape[0]
    def __getitem__(self,idx):
        x= TwoCropTransform(self.transform)(self.data_x[idx])
        y= self.data_y[idx]
        return x, y


def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'SVHN':
        mean= (0.5, 0.5, 0.5)
        std= (0.5, 0.5, 0.5)
    elif opt.dataset=='MNIST':
        mean= (0.5,)
        std= (0.5,)
        opt.size= 28
    elif opt.dataset == "VECTOR":
        mean= (0.5,)
        std=(0.5,)
        opt.size= 28
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.mean)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)
    
    if opt.dataset=="VECTOR":
        if opt.aug=='simclr':
            train_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                normalize,
            ])
        else: #salt&papper
            train_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                normalize,
                AddGaussianNoise(0., opt.aug_std),
            ])
    else:
        if opt.aug=='simclr':
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
        else: #salt&papper
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
                AddGaussianNoise(0., opt.aug_std),
            ])  


    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=TwoCropTransform(train_transform),
                                         download=True)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=TwoCropTransform(train_transform),
                                          download=True)
    elif opt.dataset == 'SVHN':
        train_dataset= datasets.SVHN(root=opt.data_folder, transform=TwoCropTransform(train_transform), download=True)
    elif opt.dataset=='MNIST':
        train_dataset= datasets.MNIST(root=opt.data_folder, transform=TwoCropTransform(train_transform), download=True)
    elif opt.dataset== "VECTOR":
        train_dataset= CustomDataset(transform= train_transform, opt=opt)

    elif opt.dataset == 'path':
        train_dataset = datasets.ImageFolder(root=opt.data_folder,
                                            transform=TwoCropTransform(train_transform))
    else:
        raise ValueError(opt.dataset)

    print(opt.dataset)
    
    ### label imbalance 
    import math
    
    if opt.dataset !='VECTOR':

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
            #labelProb[top:top+labelNum[i]] = 1 / (labelNum[i] * probMul[i])
            top+=labelNum[i]

        lst= np.array(lst)
        np.save(os.path.join(opt.model_path, opt.model_name)+"/index.npy", lst)
        os.path.join(opt.model_path, opt.model_name) 
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






def set_model(opt):
    model = SupConResNet(name=opt.model)
    criterion = SupConLoss(temperature=opt.temp)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        

        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        if opt.method == 'SupCon':
            loss = criterion(features, labels)
        elif opt.method == 'SimCLR':
            loss = criterion(features)
        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(opt.method))

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg


def main():
    opt = parse_option()

    # build data loader
    train_loader = set_loader(opt)
    print("loader complete")
    # build model and criterion
    model, criterion = set_model(opt)
    print("model complete")
    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    print("1")
    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)
        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('loss', loss, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()
