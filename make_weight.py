from __future__ import print_function

import sys
import argparse
import time
import math

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer
from networks.supcon_net import SupConResNet, LinearClassifier

from torch.utils.data import Dataset, DataLoader
import numpy as np


#DIR_PATH= "./save/SupCon/MNIST_models/SimCLR_MNIST_resnet18_mnist_lr_0.5_decay_0.0001_bsz_512_temp_0.5_trial_0_cosine_warm_ir_150.0_td_10000"
#DIR_PATH= './save/SupCon/MNIST_models/SimCLR_MNIST_resnet18_mnist_lr_0.5_decay_0.0001_bsz_512_temp_0.5_trial_0_cosine_warm_ir_150.0_td_10000_saltandpapper'
#DIR_PATH= "./save/SupCon/VECTOR_models/SimCLR_VECTOR_fnn3_mnist_lr_0.5_decay_0.0001_bsz_512_temp_0.5_trial_0_cosine_warm_ir_None_td_None_aug_saltandpapper_AUGSTD_0.6"
#DIR_PATH= "./save/SupCon/MNIST_models/SimCLR_MNIST_resnet18_mnist_lr_0.5_decay_0.0001_bsz_512_temp_0.5_trial_0_cosine_warm_ir_150.0_td_10000_0"


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
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='SVHN',
                        choices=['cifar10', 'cifar100', 'SVHN', "MNIST", "VECTOR", "tiny-imagenet-200"], help='dataset')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')
    
    ###
    parser.add_argument('--dir_path', type=str)

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './datasets/'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
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

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    elif opt.dataset == 'SVHN':
        opt.n_cls=10
    elif opt.dataset=="MNIST":
        opt.n_cls=10
    elif opt.dataset=="VECTOR":
        opt.n_cls=3
    elif opt.dataset=="tiny-imagenet-200":
        opt.n_cls=200
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt


def set_model(opt):
    model = SupConResNet(name=opt.model, is_norm=False)
    criterion = torch.nn.CrossEntropyLoss()

    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

    ckpt = torch.load(opt.dir_path+"/last.pth", map_location='cpu')
    state_dict = ckpt['model']
    
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)

    return model, classifier, criterion









def main():
    best_acc = 0
    opt = parse_option()

    # build model and criterion
    model, classifier, criterion = set_model(opt)

    #dataset
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
    elif opt.dataset == "tiny-imagenet-200":
        mean= (0.485, 0.456, 0.406)
        std= (0.229, 0.224, 0.225)
        opt.size= 64




    import numpy as np
    normalize = transforms.Normalize(mean=mean, std=std)
    
    standard_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    
    if opt.dataset == "SVHN":
        train_dataset = datasets.SVHN(root=opt.data_folder,transform=standard_transform,download=True, split="train")
    elif opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=standard_transform,
                                         download=True)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=standard_transform,
                                          download=True)
    elif opt.dataset == 'SVHN':
        train_dataset= datasets.SVHN(root=opt.data_folder, transform=standard_transform, download=True)
    
    elif opt.dataset == 'tiny-imagenet-200':
        train_dataset = datasets.ImageFolder(root=opt.data_folder+"/tiny-imagenet-200/train", 
                transform= standard_transform)


    train_idx=np.load(opt.dir_path+"/index.npy")
    
    if opt.dataset=="tiny-imagenet-200":
        tmp_data=[]
        tmp_targets=[]
        for idxx in train_idx:
            tmp_data.append(train_dataset.imgs[idxx])
            tmp_targets.append(train_dataset[idxx][1])
        train_dataset.imgs=tmp_data
        train_dataset.samples=tmp_data
        train_dataset.targets= tmp_targets

    else:
        train_dataset.data= train_dataset.data[train_idx]
    
    if opt.dataset=="tiny-imagenet-200":
        pass
    elif opt.dataset=="SVHN":
        train_dataset.labels= np.array(train_dataset.labels)[train_idx].tolist()
    else:
        train_dataset.targets= np.array(train_dataset.targets)[train_idx].tolist()

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=2, drop_last=False, pin_memory=True)
    dataset_len= len(train_dataset)


    total_outputs = torch.zeros(dataset_len, 128)
    if opt.model == "resnet50":
        total_features = torch.zeros(dataset_len, 2048)
    elif opt.model == "VGG19":
        total_features = torch.zeros(dataset_len, 512)
    top = 0
    
    if opt.dataset=="SVHN":
        total_labels= np.array(train_dataset.labels)
    else:
        total_labels= np.array(train_dataset.targets)

    for idx, (images, labels) in enumerate(train_loader):
        print(idx)
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # compute loss

        with torch.no_grad():
            features = model.encoder(images) #2048i or 512
            outputs = model(images)  # 128
            total_outputs[top:top + bsz] = outputs
            total_features[top:top+bsz] = features
        top += bsz
    """
    sum_v= np.zeros(10)
    for class_num in range(10):
        for i in range(128):
            sum_v[class_num]+=total_outputs[total_labels == class_num][:, i].var()
        print(sum_v[class_num])
    """
    print("each class vector size mean")
    for i in range(opt.n_cls):
        l2_norm= np.linalg.norm(total_outputs[total_labels==i],2,1)
        print(str(i), "norm_mean:", l2_norm.mean(), "norm_std:", l2_norm.std())
        print(np.sort(l2_norm)[0],'|', np.sort(l2_norm)[-1])
    print("each class feature vector size mean")
    for i in range(opt.n_cls):
        l2_norm= np.linalg.norm(total_features[total_labels==i],2,1)
        print(str(i), "norm_mean:", l2_norm.mean(),"norm_std:", l2_norm.std())
    
    print('min/max norm size')
    l2_norm= np.linalg.norm(total_outputs,2,1)
    print("max/min: ", np.sort(l2_norm)[-1]/np.sort(l2_norm)[0])
    print("max/min (100 mean): ", np.sort(l2_norm)[-100:].mean() / np.sort(l2_norm)[:100].mean())
    
    ###############
    # make weight #
    ###############


    np.save(opt.dir_path+"/weight.npy",  np.linalg.norm(total_outputs,2,1))
    print("weight save!")



if __name__ == '__main__':
    main()
