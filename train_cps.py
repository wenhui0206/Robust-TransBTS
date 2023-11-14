# python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 20003 train_spup3.py
# this one is the ultimate one used for general training, because we did cross validation experiments.
# this one also has the robust loss function part.

import argparse
import os
import random
import logging
import numpy as np
import time
# import setproctitle

import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F
from models.TransBTS.TransBTS_downsample8x_skipconnection_lw import TransBTS
import torch.distributed as dist
from models import criterions

from data.BraTS import BraTS, BraTS_noisy
from torch.utils.data import DataLoader
from utils.tools import all_reduce_tensor
from losses import BetaCrossEnropyError, GeneralizedCrossEntropy
import losses
# from tensorboardX import SummaryWriter
from torch import nn


local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

parser = argparse.ArgumentParser()

# Basic Information
parser.add_argument('--user', default='name of user', type=str)

parser.add_argument('--experiment', default='TransBTS', type=str)

parser.add_argument('--date', default=local_time.split(' ')[0], type=str)

parser.add_argument('--description',
                    default='TransBTS,'
                            'training on train.txt!',
                    type=str)

# DataSet Information
parser.add_argument('--root', default='/scratch1/wenhuicu/robust_seg/brats_preprocessed/', type=str)

parser.add_argument('--ckpoint_root', default='/scratch1/wenhuicu/robust_seg/TransBTS_outputs/')

parser.add_argument('--train_dir', default='', type=str)

parser.add_argument('--valid_dir', default='', type=str)

parser.add_argument('--mode', default='train', type=str)

parser.add_argument('--train_file', default='train_list.txt', type=str)

parser.add_argument('--valid_file', default='val_list.txt', type=str)

parser.add_argument('--test_file', default='test_list.txt', type=str)

parser.add_argument('--dataset', default='brats', type=str)

parser.add_argument('--model_name', default='TransBTS', type=str)

parser.add_argument('--input_C', default=4, type=int)

parser.add_argument('--input_H', default=240, type=int)

parser.add_argument('--input_W', default=240, type=int)

parser.add_argument('--input_D', default=160, type=int)

parser.add_argument('--crop_H', default=128, type=int)

parser.add_argument('--crop_W', default=128, type=int)

parser.add_argument('--crop_D', default=128, type=int)

parser.add_argument('--output_D', default=155, type=int)

# Training Information
parser.add_argument('--lr', default=0.0002, type=float)

parser.add_argument('--weight_decay', default=1e-5, type=float)

parser.add_argument('--amsgrad', default=True, type=bool)

parser.add_argument('--criterion', default='BetaCrossEnropyError', type=str)

parser.add_argument('--num_class', default=4, type=int)

parser.add_argument('--seed', default=1000, type=int)

parser.add_argument('--no_cuda', default=False, type=bool)

parser.add_argument('--gpu', default='0,1,2,3', type=str)

parser.add_argument('--num_workers', default=8, type=int)

parser.add_argument('--batch_size', default=4, type=int)

parser.add_argument('--start_epoch', default=0, type=int)

parser.add_argument('--end_epoch', default=1000, type=int)

parser.add_argument('--save_freq', default=100, type=int)

parser.add_argument('--resume', default='', type=str)

parser.add_argument('--load', default=True, type=bool)

parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')

parser.add_argument('--corrupt_r', default=0.0, type=float, help='rate of corrupted labels to be used during training')

parser.add_argument('--beta', default=0.0, type=float, help='hyper-parameter beta for robust loss')

parser.add_argument('--train_partial', default=False, type=bool, help='only train on a percentage of the training data, used during training baseline. only train on samples with gt labels.')

parser.add_argument('--start_i', default=1, type=int, help='train wth differen folds.')
parser.add_argument('--end_i', default=1, type=int, help='train wth differen folds.')
parser.add_argument('--fold', default=1, type=int, help='train wth differen folds.')
parser.add_argument('--lamda', default=1.5, type=float, help='train wth differen folds.')
args = parser.parse_args()


cv_fold_inds = {0.3: {0:(0, 60), 1:(61, 121), 2:(121, 181)}, 0.5 : {0:(0, 100), 1:(50, 150), 2:(100, 200)}, 0.7 : {0:(0, 140), 1:(30, 170), 2:(60, 200)}}

pct_train = 1.0 - args.corrupt_r
start_i = cv_fold_inds[args.corrupt_r][args.fold][0]
end_i = cv_fold_inds[args.corrupt_r][args.fold][1]


def main_worker():
    if args.local_rank == 0:
        log_dir = os.path.join(args.ckpoint_root, 'log', args.experiment + args.date)
        log_file = log_dir + '.txt'
        log_args(log_file)
        logging.info('--------------------------------------This is all argsurations----------------------------------')
        for arg in vars(args):
            logging.info('{}={}'.format(arg, getattr(args, arg)))
        logging.info('----------------------------------------This is a halving line----------------------------------')
        logging.info('{}'.format(args.description))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.distributed.init_process_group('nccl')
    torch.cuda.set_device(args.local_rank)

    _, model = TransBTS(dataset='brats', _conv_repr=True, _pe_type="learned")
    _, model2 = TransBTS(dataset='brats', _conv_repr=True, _pe_type="learned")
    model.cuda(args.local_rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    model.train()

    model2.cuda(args.local_rank)
    model2 = nn.parallel.DistributedDataParallel(model2, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    model2.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.amsgrad)

    optimizer2 = torch.optim.Adam(model2.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.amsgrad)
    
    # criterion = getattr(criterions, args.criterion)
    
    #======================Define Loss Function here==========
    flatten_func = getattr(criterions, 'flatten')
    if args.beta > 0:
        criterion = getattr(losses, args.criterion)(args.num_class)
        print(criterion)
        # criterion = BetaCrossEnropyError(args.num_class, scale=1.0, beta=args.beta)
        # criterion = GeneralizedCrossEntropy(args.num_class, q=args.beta)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if args.local_rank == 0:
        checkpoint_dir = os.path.join(args.ckpoint_root, 'checkpoint', args.experiment)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    resume = args.resume
    
    # writer = SummaryWriter()

    if os.path.isfile(resume) and args.load:
        logging.info('loading checkpoint {}'.format(resume))
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)

        model.load_state_dict(checkpoint['state_dict'])

        logging.info('Successfully loading checkpoint {} and training from epoch: {}'
                     .format(args.resume, args.start_epoch))
    else:
        logging.info('re-training!!!')

    train_list = os.path.join(args.root, args.train_dir, args.train_file)
    train_root = os.path.join(args.root, args.train_dir)
    
    ##==================================Create Dataset===========================
    names = []
    with open(train_list) as f:
        for line in f:
            line = line.strip()
            name = line.split('/')[-1]
            names.append(name)
    total_num = len(names)

    if args.train_partial==False:
        # random.shuffle(names) # should I random shuffle here? So each training, the noisy labels would be different. You shouldn't, in that way, after adding robust loss, it would be different training noisy labels.
        # num_crpt = int(total_num * args.corrupt_r)
        if args.corrupt_r > 0:
            train_noisy_set = BraTS_noisy(names[start_i : end_i], train_root, args.mode, args.corrupt_r, args.fold)
            train_gt_set = BraTS(names[:start_i] + names[end_i:], train_root, args.mode)
            # print(len(train_noisy_set))
            train_noisy_sampler = torch.utils.data.distributed.DistributedSampler(train_noisy_set)

            train_noisy_loader = DataLoader(dataset=train_noisy_set, sampler=train_noisy_sampler, batch_size=args.batch_size//4, drop_last=True, num_workers=args.num_workers, pin_memory=True)

            train_gt_sampler = torch.utils.data.distributed.DistributedSampler(train_gt_set)

            train_gt_loader = DataLoader(dataset=train_gt_set, sampler=train_gt_sampler, batch_size=args.batch_size//4, drop_last=True, num_workers=args.num_workers, pin_memory=True)
            train_set = torch.utils.data.ConcatDataset([train_gt_set, train_noisy_set])
        else:
            train_set = BraTS(names, train_root, args.mode)
    else:
        train_set = BraTS(names[:start_i] + names[end_i:], train_root, args.mode) # the second part training set is gt when training.


    ##===============================
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    # train_sampler = torch.utils.data.BatchSampler(train_set)
    logging.info('Samples for train = {}'.format(len(train_set)))

    # num_gpu = (len(args.gpu)+1) // 2
    num_gpu = 2

    train_loader = DataLoader(dataset=train_set, sampler=train_sampler, batch_size=args.batch_size // num_gpu, drop_last=True, num_workers=args.num_workers, pin_memory=True)
    

    start_time = time.time()

    torch.set_grad_enabled(True)
    i = 0
    for epoch in range(args.start_epoch, args.end_epoch):
        # train_sampler.set_epoch(epoch)  # shuffle
        # setproctitle.setproctitle('{}: {}/{}'.format(args.user, epoch+1, args.end_epoch))
        train_gt_sampler.set_epoch(epoch)
        train_noisy_sampler.set_epoch(epoch)
        start_epoch = time.time()

        for data_gt, data_ns in zip(train_gt_loader, train_noisy_loader):
        # for i, data in enumerate(train_loader):
            adjust_learning_rate(optimizer, epoch, args.end_epoch, args.lr)
            # adjust_learning_rate(optimizer2, epoch, args.end_epoch, args.lr)
            
            x_gt, target_gt = data_gt
            x_ns, target_ns = data_ns
           
            x = torch.cat((x_gt, x_ns), dim=0)
            target = torch.cat((target_gt, target_ns), dim=0)
            # x, target = data
            x = x.cuda(args.local_rank, non_blocking=True)
            target = target.cuda(args.local_rank, non_blocking=True)
            
            output = model(x)
            output2 = model2(x)
            # loss, loss1, loss2, loss3 = criterion(output, target)
            ##=========================================
            target[target == 4] = 3
            loss_sup = criterion(flatten_func(output[0:1]).permute((1, 0)), torch.flatten(target[0:1])) + criterion(flatten_func(output2[0:1]).permute((1, 0)), torch.flatten(target[0:1]))

            pseudo_map1 = torch.argmax(F.softmax(output, dim=1), dim=1)
            pseudo_map2 = torch.argmax(F.softmax(output2, dim=1), dim=1)
            
            loss_cps = criterion(flatten_func(output).permute((1, 0)), torch.flatten(pseudo_map2)) + criterion(flatten_func(output2).permute((1, 0)), torch.flatten(pseudo_map1))
            loss = loss_sup + args.lamda * loss_cps
            ##==========================================
            reduce_loss = all_reduce_tensor(loss, world_size=num_gpu).data.cpu().numpy()
            # reduce_loss1 = all_reduce_tensor(loss1, world_size=num_gpu).data.cpu().numpy()
            # reduce_loss2 = all_reduce_tensor(loss2, world_size=num_gpu).data.cpu().numpy()
            # reduce_loss3 = all_reduce_tensor(loss3, world_size=num_gpu).data.cpu().numpy()
            # print(torch.cuda.memory_summary(device=None, abbreviated=False))
            if args.local_rank == 0:
                logging.info('Epoch: {}_Iter:{}  loss: {:.5f}'
                             .format(epoch, i, reduce_loss))
            i = i + 1
            optimizer.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer2.step()

        end_epoch = time.time()
        if args.local_rank == 0:
            if (epoch + 1) % int(args.save_freq) == 0 \
                    or (epoch + 1) % int(args.end_epoch - 1) == 0 \
                    or (epoch + 1) % int(args.end_epoch - 2) == 0 \
                    or (epoch + 1) % int(args.end_epoch - 3) == 0:
                file_name = os.path.join(checkpoint_dir, 'model_epoch_{}.pth'.format(epoch))
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optim_dict': optimizer.state_dict(),
                },
                    file_name)

            # writer.add_scalar('lr:', optimizer.param_groups[0]['lr'], epoch)
            # writer.add_scalar('loss:', reduce_loss, epoch)
            # writer.add_scalar('loss1:', reduce_loss1, epoch)
            # writer.add_scalar('loss2:', reduce_loss2, epoch)
            # writer.add_scalar('loss3:', reduce_loss3, epoch)

        if args.local_rank == 0:
            epoch_time_minute = (end_epoch-start_epoch)/60
            remaining_time_hour = (args.end_epoch-epoch-1)*epoch_time_minute/60
            logging.info('Current epoch time consumption: {:.2f} minutes!'.format(epoch_time_minute))
            logging.info('Estimated remaining training time: {:.2f} hours!'.format(remaining_time_hour))

    if args.local_rank == 0:
        # writer.close()

        final_name = os.path.join(checkpoint_dir, 'model_epoch_last.pth')
        torch.save({
            'epoch': args.end_epoch,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
        },
            final_name)
    end_time = time.time()
    total_time = (end_time-start_time)/3600
    logging.info('The total training time is {:.2f} hours'.format(total_time))
    # torch.cuda().empty_cache()
    logging.info('----------------------------------The training process finished!-----------------------------------')


def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(init_lr * np.power(1-(epoch) / max_epoch, power), 8)


def log_args(log_file):

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s ===> %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # args FileHandler to save log file
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # args StreamHandler to print log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # add the two Handler
    logger.addHandler(ch)
    logger.addHandler(fh)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main_worker()
