import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset

from datasets.inaturalist import INaturalist
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import itertools

from generalized_wasserstein_dice_loss.loss import GeneralizedWassersteinDiceLoss
from myloss import BCEWithSoftmaxLoss, BCEWithSoftmaxFocalLoss

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='iNaturalist Training, using Pytorch ImageNet training code')
parser.add_argument('data', metavar='DIR', nargs='?', default='iNaturalist',
                    help='path to dataset (default: iNaturalist)')
parser.add_argument('--target-type', nargs='+', type=str, default='full', choices=["full", "kingdom", "phylum", 
                    "class", "order", "family", "genus", "super"], help='Type of target to use')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")
parser.add_argument('--output-dir', default='./runs/train/exp', type=str,
                    help='tensorboard output directory')
parser.add_argument('--loss', default='CrossEntropyLoss', type=str, choices=['CrossEntropyLoss', 'BCELoss',
            'BCEWithLogitsLoss', 'GWDL', 'BCEWithSoftmaxLoss', 'BCEWithSoftmaxFocalLoss', 'tree_layer_loss'], help='loss function')
parser.add_argument('--loss-reduction', default='mean', type=str, choices=['none', 'sum',
            'mean',], help='loss reduction')
parser.add_argument('--scheduler', default='StepLR', type=str, choices=['StepLR', 'CyclicLR'],
            help='learning rate scheduler')
parser.add_argument('--use-weight', type=str, choices=['A', 'B'], 
            help="use weight when loss is BCELoss or BCEWithLogitsLoss")
parser.add_argument('--weight-type', type=str, choices=['A', 'B'], 
            help="use weight when loss is BCELoss or BCEWithLogitsLoss")
parser.add_argument('--lr-range-test', action='store_true', help="use CyclicLR scheduler to test the reasonable learning rate range")
parser.add_argument('--min-lr', '--min-learning-rate', default=0.0001, type=float, metavar='LR', help='minimum learning rate for CyclicLR', dest='min_lr')
parser.add_argument('--step-size', default=5, type=int, help='step size for reducing the learning rate')


best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

    if args.use_weight is not None:
        args.weight_type = args.use_weight
        args.use_weight = True


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # Data loading code ====>pre model creating
    if args.dummy:
        print("=> Dummy data is used!")
        # train_dataset = datasets.FakeData(1281167, (3, 224, 224), 1000, transforms.ToTensor())
        # val_dataset = datasets.FakeData(50000, (3, 224, 224), 1000, transforms.ToTensor())
        train_dataset = datasets.FakeData(32, (3, 224, 224), 1000, transforms.ToTensor())
        val_dataset = datasets.FakeData(16, (3, 224, 224), 1000, transforms.ToTensor())
    else:
        # traindir = os.path.join(args.data, 'train')
        # valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


        # train_dataset = datasets.ImageFolder(
        #     traindir,
        #     transforms.Compose([
        #         transforms.RandomResizedCrop(224),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         normalize,
        #     ]))

        # val_dataset = datasets.ImageFolder(
        #     valdir,
        #     transforms.Compose([
        #         transforms.Resize(256),
        #         transforms.CenterCrop(224),
        #         transforms.ToTensor(),
        #         normalize,
        #     ]))

        # root_dir = '../../Datasets/iNaturalist'
        root_dir = '/home/Datasets/iNaturalist'
        train_dataset = INaturalist(root=root_dir, version='2021_train_mini', target_type=args.target_type, 
            build_tree=True,
            load_weight=args.use_weight, 
            transform=transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                # add new: wangcong
                # transforms.RandomRotation(0,180),
                # transforms.AugMix(),
                # ----------------
                transforms.ToTensor(),
                normalize,
            ]))
        
        val_dataset = INaturalist(root=root_dir, version='2021_valid', target_type=args.target_type,
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
        

    # create model
    if args.dummy:
        num_class =  1000
    else:
        if 'full' in args.target_type:
            num_class = 10000
        else:
            num_class = len(train_dataset.categories_index[args.target_type[0]])

    args.num_class = num_class
    print('num_class', num_class)

    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True, num_classes=num_class)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](num_classes=num_class)

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    #load distance matrix for BCEWithiLogitsLoss weight
    if args.use_weight and args.target_type == 'order':
        distance_matrix = torch.Tensor(train_dataset.full_distance_matrix).to(device)
        
        if args.loss == 'GWDL':
            distance_matrix = distance_matrix / 8  #max=1,min=0
        else:
            if args.weight_type == 'A':
                distance_matrix = distance_matrix / 4 + 1
                print("weight A")
            else: #'B'
                print("weight B")
                # 改为在对角线上+2 
                distance_matrix = distance_matrix / 4 + torch.eye(distance_matrix.size()[0]).to(device) * 2
    else:
        distance_matrix = None

    # define loss function (criterion), optimizer, and learning rate scheduler
    if args.loss == 'BCELoss':
        criterion = nn.BCELoss().to(device)
    elif args.loss == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss(reduction=args.loss_reduction).to(device)
    elif args.loss == 'GWDL':
        criterion = GeneralizedWassersteinDiceLoss(dist_matrix=distance_matrix, adding_ce=True, gamma=3)
    elif args.loss == 'BCEWithSoftmaxLoss':
        # print('hi')
        criterion = BCEWithSoftmaxLoss(reduction=args.loss_reduction).to(device)
    elif args.loss == 'BCEWithSoftmaxFocalLoss':
        criterion = BCEWithSoftmaxFocalLoss(reduction=args.loss_reduction).to(device)      
    else:
        # print('wrong')
        criterion = nn.CrossEntropyLoss().to(device)


    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.scheduler == 'CyclicLR':
        # GPU并行时再除以GPU个数
        iter_per_epoch = (len(train_dataset) / ngpus_per_node) // args.batch_size
        print('iter_per_epoch: ', iter_per_epoch)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,base_lr=args.min_lr, 
            max_lr=args.lr, step_size_up=args.step_size*iter_per_epoch, mode='triangular')
    else:    
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            elif torch.cuda.is_available():
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            # if args.gpu is not None:
            #     # best_acc1 may be from a checkpoint from a different GPU
            #     best_acc1 = best_acc1.to(args.gpu) #'float' object has no attribute 'to'
            model.load_state_dict(checkpoint['state_dict'])
            # print(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))



    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    treelayer_connection = train_dataset.tree_layer_connection #用于分层计算loss，权重固定
    lamda_list = [[0,0,0,1],[0,0,0.5,0.5],[0.1,0.8,0.1,0],[0.8,0.15,0.05,0]]

    # lamda_list = [[0.05,0.05,0.5,0.4],[0.1,0.5,0.3,0.1],[0.2,0.7,0.1,0],[0.8,0.15,0.05,0]]

    # create summary writer for tensorboard
    summary_writer = SummaryWriter(log_dir=args.output_dir)

    if args.evaluate:
        # test saved checkpoint TODO
        # if args.rank in {-1, 0}:
        #     print('Vis model......')
        #     dummy_input = torch.rand(20,3,224,224)
        #     summary_writer.add_graph(model, (dummy_input,))

        validate(val_loader, model, criterion, args, distance_matrix=distance_matrix, treelayer_connection=treelayer_connection, 
            device=device, plot_cm=True)
        return



    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if args.loss == "tree_layer_loss":
            if epoch < 2: 
                lamda = lamda_list[0]
            elif epoch < 5 : 
                lamda = lamda_list[1]
            elif epoch < 10 :
                lamda = lamda_list[2]
            else:
                lamda = lamda_list[3]
        else:
            lamda = None

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, device, args, summary_writer, distance_matrix, scheduler, treelayer_connection, lamda)

        # evaluate on validation set
        plot_cm = False
        if epoch == args.epochs-1: 
            plot_cm = True
        acc1 = validate(val_loader, model, criterion, args, summary_writer, epoch, distance_matrix, treelayer_connection, device, plot_cm=plot_cm)
        if args.rank in {-1, 0}:
            summary_writer.add_scalar('validation_acc@1', acc1, epoch)
        
        if scheduler and not args.scheduler.startswith('Cyclic'):
            scheduler.step()
        
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, is_best, args.output_dir)

        


def train(train_loader, model, criterion, optimizer, epoch, device, args, summary_writer, 
    distance_matrix, scheduler, treelayer_connection=[], lamda=None):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    lr = AverageMeter('LR', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, lr, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    if args.loss == "tree_layer_loss":
        tree_losses = AverageMeter('Treeloss', ':.4e') #wangcong
        # family_top1 = AverageMeter('FamilyAcc@1', ':6.2f')
        # order_top1 = AverageMeter('OrderAcc@1', ':6.2f')
        class_top1 = AverageMeter('ClassAcc@1', ':6.2f')        
        phylum_top1 = AverageMeter('PhylumAcc@1', ':6.2f')
        kingdom_top1 = AverageMeter('KingdomAcc@1', ':6.2f')
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, lr, losses, tree_losses, top1, top5, class_top1, phylum_top1, kingdom_top1],
            prefix="Epoch: [{}]".format(epoch))
        # progress = ProgressMeter(
        #     len(train_loader),
        #     [batch_time, data_time, lr, losses, tree_losses, top1, top5, family_top1, order_top1, class_top1, phylum_top1,kingdom_top1],
        #     prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    if args.lr_range_test:
        lr_top1_loss = []
    for i, (images, target_all) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        
        if isinstance(target_all, list):
            target = target_all[0]
        else:
            target = target_all

        target = target.to(device, non_blocking=True)


        if args.loss.startswith('BCE'):
            target_one_hot = nn.functional.one_hot(target, num_classes=args.num_class).float()
            if args.use_weight and args.target_type == 'order':
                # print(target_one_hot.size(), distance_matrix.size())
                criterion.weight = torch.mm(target_one_hot, distance_matrix)
        else:
            target_one_hot = target

        

        # compute output
        output = model(images)
        # print(criterion)
        loss = criterion(output, target_one_hot)


        layer_top1 = []
        if args.loss == "tree_layer_loss":
            loss = loss * lamda[0]
            other_targets= target_all[1:]
            tree_loss = 0
            layer_output = output
            layer_depth = 0
            for weight in treelayer_connection:
                layer_output = layer_output @ weight.to(device)
                tree_loss += criterion(layer_output, other_targets[layer_depth].to(device)) * 0.1 * (len(treelayer_connection) - layer_depth)
                layer_top1.append(accuracy(layer_output, other_targets[layer_depth].to(device))[0])
                # print('layer_top1: ', layer_top1)
                layer_depth += 1
                tree_loss = tree_loss * lamda[layer_depth]

            # loss += (tree_loss / len(treelayer_connection))
            loss += tree_loss


        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        if args.loss == "tree_layer_loss":
            tree_losses.update(tree_loss.item(), images.size(0))
            class_top1.update(layer_top1[0][0], images.size(0))
            phylum_top1.update(layer_top1[1][0], images.size(0))
            kingdom_top1.update(layer_top1[2][0], images.size(0))

            # family_top1.update(layer_top1[1][0], images.size(0))
            # order_top1.update(layer_top1[2][0], images.size(0))
            # class_top1.update(layer_top1[3][0], images.size(0))
            # phylum_top1.update(layer_top1[4][0], images.size(0))
            # kingdom_top1.update(layer_top1[5][0], images.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # CyclicLR
        if scheduler and args.scheduler.startswith('Cyclic'):
            scheduler.step()
            if args.lr_range_test and args.rank in {-1, 0}:
                #draw LR range test curve
                # print(scheduler.get_last_lr())
                # summary_writer.add_scalar('LR_range_test', top1.avg, scheduler.get_last_lr()[0]*1000)
                lr_top1_loss.append([scheduler.get_last_lr()[0], top1.val.cpu(), losses.val])
                summary_writer.add_scalar('lr', lr.val, i)
                summary_writer.add_scalar('Acc@1_iter', top1.val, i)
                summary_writer.add_scalar('loss_iter', losses.val, i)
                summary_writer.add_scalars('Layer_Acc@1', {'order_top1':top1.val, 
                'class_top1':class_top1.val, 'phylum_top1':phylum_top1.val, 'kingdom_top1':kingdom_top1.val}, i)
                # summary_writer.add_scalars('Layer_Acc@1', {'family_top1':family_top1.val, 'order_top1':order_top1.val, 
                # 'class_top1':class_top1.val, 'phylum_top1':phylum_top1.val, 'kingdom_top1':kingdom_top1.val}, i)
                

            

        lr.update(scheduler.get_last_lr()[0])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)


    if args.rank in {-1, 0}:
        summary_writer.add_scalar('train_loss', losses.avg, epoch)
        summary_writer.add_scalar('train_Acc@1', top1.avg, epoch)
        summary_writer.add_scalar('train_Acc@5', top5.avg, epoch)
        if args.loss == "tree_layer_loss":
            summary_writer.add_scalar('train_tree_loss', tree_losses.avg, epoch)
            summary_writer.add_scalars('train_Layer_Acc@1', {'order_top1':top1.avg, 
                'class_top1':class_top1.avg, 'phylum_top1':phylum_top1.avg, 'kingdom_top1':kingdom_top1.avg}, epoch)

            # summary_writer.add_scalars('train_Layer_Acc@1', {'family_top1':family_top1.avg, 'order_top1':order_top1.avg, 
            #     'class_top1':class_top1.avg, 'phylum_top1':phylum_top1.avg, 'kingdom_top1':kingdom_top1.avg}, epoch)


        if args.lr_range_test:
            # drow LR range test curve
            plt.switch_backend('agg')
            # plt.ion() #交互模式
            lr_top1_loss = np.array(lr_top1_loss)
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.plot(lr_top1_loss[:,0], lr_top1_loss[:,1], label="train_Acc@1", color='b')
            plt.legend(loc='lower right')
            plt.xlabel('learnng rate')
            plt.ylabel('train_Acc@1')

            ax2 = ax1.twinx()
            ax2.plot(lr_top1_loss[:,0], lr_top1_loss[:,2], label="train_loss",color="b",linestyle='dotted')
            plt.legend(loc='upper right')
            plt.ylabel("train_loss")

            plt.savefig(os.path.join(args.output_dir,'lr_range_test.png'), dpi=200)
            



def validate(val_loader, model, criterion, args, summary_writer=None, epoch=None, distance_matrix=None, treelayer_connection=[], device=None, plot_cm=False):

    def run_validate(loader, base_progress=0):
        confusion_matrix = None
        layers_cm = [None, None, None]
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                if args.gpu is not None and torch.cuda.is_available():
                    images = images.cuda(args.gpu, non_blocking=True)
                # if torch.backends.mps.is_available(): # torch version=1.9 has no this para
                #     images = images.to('mps')
                #     target = target.to('mps')

                if isinstance(target, list):
                    target_all = target
                    target = target_all[0]
                    other_targets= target_all[1:]
                    

                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)

                if args.loss.startswith('BCE'):
                    target_one_hot = nn.functional.one_hot(target, num_classes=args.num_class).float()
                    if args.use_weight and args.target_type == 'order':
                        # print(target_one_hot.size(), distance_matrix.size())
                        criterion.weight = torch.mm(target_one_hot, distance_matrix)
                else:
                    target_one_hot = target

                # compute output
                output = model(images)
                # print(output.size(), target_one_hot.size())
                loss = criterion(output, target_one_hot)
                _, _preds = torch.max(output, 1)
                if plot_cm:
                    if confusion_matrix is None:
                        confusion_matrix = cal_confusion_matrix(target, _preds, labels=range(args.num_class))
                    else:              
                        confusion_matrix += cal_confusion_matrix(target, _preds, labels=range(args.num_class))

                layer_top1 = []
                layers_outputs = []
                if args.loss == 'tree_layer_loss':
                    if args.loss is None:
                        tree_loss = 0
                        layer_output = output
                        layer_depth = 0
                        for weight in treelayer_connection:
                            layer_output = layer_output @ weight.to(device)
                            layers_outputs.append(layer_output)
                            tree_loss += criterion(layer_output, other_targets[layer_depth].to(device)) * 0.1 * (len(treelayer_connection) - layer_depth)
                            layer_top1.append(accuracy(layer_output, other_targets[layer_depth].to(device))[0])

                            _, _layer_preds = torch.max(layer_output, 1)
                            if plot_cm:
                                layer_cm = cal_confusion_matrix(other_targets[layer_depth], _layer_preds, labels=range(layer_output.size()[1]))
                                if layers_cm[layer_depth] is None:
                                    layers_cm[layer_depth] = layer_cm
                                else:
                                    layers_cm[layer_depth] += layer_cm

                            layer_depth += 1

                        #自上而下再计算一遍
                        # for j in range(layer_depth):
                        #     layer_j = layer_depth - j - 1
                        #     weight_T = torch.transpose(treelayer_connection[layer_j], 0, 1)
                        #     _, j_layer_output = torch.max(layers_outputs[layer_j], 1)
                        #     j_layer_output = nn.functional.one_hot(j_layer_output, num_classes=weight_T.size()[0]).float()
                        #     next_layer_output = j_layer_output @ weight_T.to(device)
                            
                        #     if layer_j < 1: #最下面一层
                        #         output = next_layer_output * output
                        #     else:
                        #         filtered_output = next_layer_output * layers_outputs[layer_j - 1]
                        #         # 计算j+1层新的acc
                        #         j_top1 = accuracy(filtered_output, other_targets[layer_j - 1].to(device))[0]
                        #         layer_top1[layer_j - 1] = j_top1


                        loss += tree_loss
                    else:
                        # 只计算accuracy
                        _, layer_output = torch.max(output, 1)
                        layer_output = nn.functional.one_hot(layer_output, num_classes=args.num_class).float()
                        layer_depth = 0
                        for weight in treelayer_connection:
                            layer_output = layer_output @ weight.to(device)
                            # tree_loss += criterion(layer_output, other_targets[layer_depth].to(device)) * 0.1 * (len(treelayer_connection) - layer_depth)
                            layer_top1.append(accuracy(layer_output, other_targets[layer_depth].to(device))[0])

                            _, _layer_preds = torch.max(layer_output, 1)
                            if plot_cm:
                                layer_cm = cal_confusion_matrix(other_targets[layer_depth], _layer_preds, labels=range(layer_output.size()[1]))
                                if layers_cm[layer_depth] is None:
                                    layers_cm[layer_depth] = layer_cm
                                else:
                                    layers_cm[layer_depth] += layer_cm

                                # print(layer_depth, layers_cm[layer_depth])

                            # print('layer_top1: ', layer_top1)
                            layer_depth += 1


                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
                if args.loss == "tree_layer_loss":
                    class_top1.update(layer_top1[0][0], images.size(0))
                    phylum_top1.update(layer_top1[1][0], images.size(0))
                    kingdom_top1.update(layer_top1[2][0], images.size(0))


                    # tree_losses.update(tree_loss.item(), images.size(0))
                    # family_top1.update(layer_top1[1][0], images.size(0))
                    # order_top1.update(layer_top1[2][0], images.size(0))
                    # class_top1.update(layer_top1[3][0], images.size(0))
                    # phylum_top1.update(layer_top1[4][0], images.size(0))
                    # kingdom_top1.update(layer_top1[5][0], images.size(0))



                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

        return confusion_matrix, layers_cm

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    
    # progress = ProgressMeter(
    #     len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
    #     [batch_time, losses, top1, top5],
    #     prefix='Test: ')

    # if args.loss == "tree_layer_loss":
        #wangcong
    # family_top1 = AverageMeter('FamilyAcc@1', ':6.2f')
    # order_top1 = AverageMeter('OrderAcc@1', ':6.2f')
    class_top1 = AverageMeter('ClassAcc@1', ':6.2f')        
    phylum_top1 = AverageMeter('PhylumAcc@1', ':6.2f')
    kingdom_top1 = AverageMeter('KingdomAcc@1', ':6.2f')
    # class_cm = None
    # phylum_cm = None
    # kingdom_cm = None
        
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, top5, class_top1, phylum_top1, kingdom_top1],
        prefix='Test: ')
        # progress = ProgressMeter(
        #     len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        #     [batch_time, losses, top1, top5, family_top1, order_top1, class_top1, phylum_top1, kingdom_top1],
        #     prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    confusion_matrix, layers_cm = run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()
        if args.loss == "tree_layer_loss":
            class_top1.all_reduce()      
            phylum_top1.all_reduce()
            kingdom_top1.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        a, b = run_validate(aux_val_loader, len(val_loader))
        confusion_matrix += a


    progress.display_summary()

    if args.evaluate:
        pass
    else:
        if args.rank in {-1, 0} and summary_writer:
            summary_writer.add_scalar('val_loss', losses.avg, epoch)
            if args.loss == "tree_layer_loss":
                summary_writer.add_scalars('val_Layer_Acc@1', {'order_top1':top1.avg, 
                    'class_top1':class_top1.avg, 'phylum_top1':phylum_top1.avg, 'kingdom_top1':kingdom_top1.avg}, epoch)

                # summary_writer.add_scalar('train_tree_loss', tree_losses.avg, epoch)
                # summary_writer.add_scalars('val_Layer_Acc@1', {'family_top1':family_top1.avg, 'order_top1':order_top1.avg, 
                #     'class_top1':class_top1.avg, 'phylum_top1':phylum_top1.avg, 'kingdom_top1':kingdom_top1.avg}, epoch)
    if plot_cm:
        plot_confusion_matrix(confusion_matrix, range(len(confusion_matrix)), save_dir=args.output_dir)
        # plt.savefig(args.output_dir + '/confusion_matix.png', dpi=600)
        if args.loss == "tree_layer_loss":
            for cm in layers_cm:
                plot_confusion_matrix(cm, range(len(cm)), save_dir=args.output_dir, 
                    filename=str(len(cm))+'_'+'confusion_matrix.png')
                # plt.savefig(args.output_dir + '/' +str(len(cm)) + '_confusion_matix.png', dpi=400)


    return top1.avg

#计算混淆矩阵
def cal_confusion_matrix(targets, preds, labels):
    preds = preds.to('cpu').numpy()
    targets = targets.to('cpu').numpy()
    confusion_matrix = metrics.confusion_matrix(targets, preds, labels=labels)
    return confusion_matrix

#绘制混淆矩阵
def plot_confusion_matrix(cm, names,
                          normalize=True, save_dir='', filename='confusion_matrix.png'):
    # try:
    import seaborn as sn
    from pathlib import Path

    matrix = np.array(cm)
    array = matrix / ((matrix.sum(0).reshape(1, -1) + 1E-9) if normalize else 1)  # normalize columns
    array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

    fig = plt.figure(figsize=(12, 9), tight_layout=True)
    nc, nn = len(names), len(names)  # number of classes, names
    sn.set(font_scale=1.0 if nc < 50 else 0.8)  # for label size
    labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
        sn.heatmap(array,
                   annot=nc < 30,
                   annot_kws={
                       "size": 8},
                   cmap='Blues',
                   fmt='.2f',
                   square=True,
                   vmin=0.0,
                   xticklabels=names if labels else "auto",
                   yticklabels=names if labels else "auto").set_facecolor((1, 1, 1))
    fig.axes[0].set_xlabel('True')
    fig.axes[0].set_ylabel('Predicted')
    fig.savefig(Path(save_dir) / filename, dpi=250)
    plt.close()
    # except Exception as e:
    #     print(f'WARNING: ConfusionMatrix plot failure: {e}')

    # """
    # This function prints and plots the confusion matrix.
    # Normalization can be applied by setting `normalize=True`.
    # """
    # if normalize:
    #     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #     print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix, without normalization')

    # print(cm)

    # plt.switch_backend('agg') # AVOID RuntimeError('Invalid DISPLAY variable') when using ssh
    # plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    # plt.colorbar()
    # tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=45)
    # plt.yticks(tick_marks, classes)

    # fmt = '.2f' if normalize else 'd'
    # thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     plt.text(j, i, format(cm[i, j], fmt),
    #              horizontalalignment="center",
    #              color="white" if cm[i, j] > thresh else "black")

    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    # plt.tight_layout()


def save_checkpoint(state, is_best, dir_path, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(dir_path, filename))
    if is_best:
        shutil.copyfile(os.path.join(dir_path, filename), os.path.join(dir_path, 'model_best.pth.tar'))

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()