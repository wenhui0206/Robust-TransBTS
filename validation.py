import argparse
import os
import time
import random
import numpy as np
import setproctitle

import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torch.optim
from torch.utils.data import DataLoader

from data.BraTS import BraTS
from predict import validate_performance
from models.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS


parser = argparse.ArgumentParser()

parser.add_argument('--user', default='name of user', type=str)

parser.add_argument('--root', default='/home/wenhuicu/brats_preprocessed/', type=str)

parser.add_argument('--valid_dir', default='data/', type=str)

parser.add_argument('--valid_file', default='val_list.txt', type=str)

parser.add_argument('--output_dir', default='output', type=str)

parser.add_argument('--submission', default='submission', type=str)

parser.add_argument('--visual', default='visualization', type=str)

parser.add_argument('--experiment', default='TransBTS', type=str)

parser.add_argument('--test_date', default='2022-02-09', type=str)

parser.add_argument('--test_file', default='model_epoch_249.pth', type=str)

parser.add_argument('--use_TTA', default=False, type=bool)

parser.add_argument('--post_process', default=True, type=bool)

parser.add_argument('--save_format', default='nii', choices=['npy', 'nii'], type=str)

parser.add_argument('--crop_H', default=128, type=int)

parser.add_argument('--crop_W', default=128, type=int)

parser.add_argument('--crop_D', default=128, type=int)

parser.add_argument('--seed', default=1000, type=int)

parser.add_argument('--model_name', default='TransBTS', type=str)

parser.add_argument('--num_class', default=4, type=int)

parser.add_argument('--no_cuda', default=False, type=bool)

parser.add_argument('--gpu', default='0', type=str)

parser.add_argument('--num_workers', default=4, type=int)

parser.add_argument('--output_root', default="../TransBTS_outputs/", type=str)

args = parser.parse_args()


def main():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    _, model = TransBTS(dataset='brats', _conv_repr=True, _pe_type="learned")

    model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()
    prefix = args.output_root
    load_file = os.path.join(prefix,
                             'checkpoint', args.experiment+args.test_date, args.test_file)

    if os.path.exists(load_file):
        print('Loading trained model.......')
        checkpoint = torch.load(load_file)
        model.load_state_dict(checkpoint['state_dict'])
        args.start_epoch = checkpoint['epoch']
        print('Successfully load checkpoint {}'.format(os.path.join(args.experiment+args.test_date, args.test_file)))
    else:
        print('There is no resume file to load!')

    valid_list = os.path.join(args.valid_dir, args.valid_file)
    valid_root = os.path.join(args.root)
    valid_set = BraTS(valid_list, valid_root, mode='valid')
    print('Samples for valid = {}'.format(len(valid_set)))

    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    submission = os.path.join(prefix, args.output_dir,
                              args.submission, args.experiment+args.test_date)
    visual = os.path.join(prefix, args.output_dir,
                          args.visual, args.experiment+args.test_date)

    if not os.path.exists(submission):
        os.makedirs(submission)
    if not os.path.exists(visual):
        os.makedirs(visual)

    start_time = time.time()

    with torch.no_grad():
        validate_performance(valid_loader=valid_loader,
                         model=model,
                         load_file=load_file,
                         multimodel=False,
                         savepath="",
                         visual=visual,
                         names=valid_set.names,
                         use_TTA=args.use_TTA, # test time augmentation. set false as default
                         save_format=args.save_format,
                         snapshot=False,
                         postprocess=True,
                         valid_in_train=True
                         )

    end_time = time.time()
    full_test_time = (end_time-start_time)/60
    average_time = full_test_time/len(valid_set)
    print('{:.2f} minutes!'.format(average_time))


if __name__ == '__main__':
    # config = opts()
    setproctitle.setproctitle('{}: Testing!'.format(args.user))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    main()

