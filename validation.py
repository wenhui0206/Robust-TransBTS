import argparse
import os
import time
import random
import numpy as np
# import setproctitle

import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torch.optim
from torch.utils.data import DataLoader

from data.BraTS import BraTS
from predict import validate_performance, validate_softmax, compare_performance
from models.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS


parser = argparse.ArgumentParser()

parser.add_argument('--user', default='name of user', type=str)

parser.add_argument('--root', default='/scratch1/wenhuicu/robust_seg/brats_preprocessed/', type=str)

parser.add_argument('--valid_dir', default='/scratch1/wenhuicu/robust_seg/TransBTS/data/', type=str)

parser.add_argument('--valid_file', default='val_list.txt', type=str)

parser.add_argument('--output_dir', default='output', type=str)

parser.add_argument('--submission', default='submission', type=str)

parser.add_argument('--visual', default='visualization', type=str)

parser.add_argument('--experiment', default='TransBTS', type=str)

parser.add_argument('--test_date', default='2022-02-10', type=str)

parser.add_argument('--test_file', default='model_epoch_999.pth', type=str)

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

parser.add_argument('--output_root', default="/scratch1/wenhuicu/robust_seg/TransBTS_outputs/", type=str)

parser.add_argument('--pred_pct', default=1.0, type=float)

parser.add_argument('--csv_name', default='', type=str)

parser.add_argument('--start_i', default=0, type=int)

parser.add_argument('--end_i', default=1, type=int)

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
                             'checkpoint', args.experiment, args.test_file)

    if os.path.exists(load_file):
        print('Loading trained model.......')
        checkpoint = torch.load(load_file)
        model.load_state_dict(checkpoint['state_dict'])
        args.start_epoch = checkpoint['epoch']
        print('Successfully load checkpoint {}'.format(os.path.join(args.experiment, args.test_file)))
    else:
        print('There is no resume file to load!')

    valid_list = os.path.join(args.valid_dir, args.valid_file)
    valid_root = os.path.join(args.root)
    #=====================================
    names = []
    with open(valid_list) as f:
        for line in f:
            line = line.strip()
            name = line.split('/')[-1]
            names.append(name)

    #======================Predict the part for noisy labels=========================
    if args.pred_pct < 1:
        num_all = len(names)
        # names = names[: int(args.pred_pct * num_all)]
        names = names[args.start_i : args.end_i]


    valid_set = BraTS(names, valid_root, mode='valid')
    print('Samples for valid = {}'.format(len(valid_set)))

    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    if args.submission != '':
	    submission = os.path.join(prefix, args.output_dir,
                              args.submission, args.experiment)
    else:
	    submission = ''
    visual = os.path.join(prefix, args.output_dir,
                          args.visual, args.experiment)

    if (not os.path.exists(submission)) and args.submission != '':
        os.makedirs(submission)
    if not os.path.exists(visual):
        os.makedirs(visual)

    start_time = time.time()

    with torch.no_grad():
        # compare_performance(valid_loader=valid_loader,
        #                  model=model,
        #                  load_file=load_file,
        #                  multimodel=False,
        #                  savepath=submission,
        #                  visual=visual,
        #                  names=valid_set.names,
        #                  use_TTA=args.use_TTA, # test time augmentation. set false as default
        #                  save_format=args.save_format,
        #                  snapshot=False,
        #                  postprocess=True,
        #                  valid_in_train=True,
        #                  fname=args.experiment[6:]
        #                  )
        validate_performance(valid_loader=valid_loader,
                         model=model,
                         load_file=load_file,
                         multimodel=False,
                         savepath=submission,
                         visual=visual,
                         names=valid_set.names,
                         use_TTA=args.use_TTA, # test time augmentation. set false as default
                         save_format=args.save_format,
                         snapshot=False,
                         postprocess=True,
                         valid_in_train=True,
                         csv_name=args.csv_name,
                        fp=args.output_root + 'csv_results/')
        # validate_softmax(valid_loader=valid_loader,
        #                  model=model,
        #                  load_file=load_file,
        #                  multimodel=False,
        #                  savepath=submission,
        #                  visual=visual,
        #                  names=valid_set.names,
        #                  use_TTA=args.use_TTA, # test time augmentation. set false as default
        #                  save_format=args.save_format,
        #                  snapshot=True,
        #                  postprocess=True,
        #                  valid_in_train=True
        #                  )

    end_time = time.time()
    full_test_time = (end_time-start_time)/60
    average_time = full_test_time/len(valid_set)
    print('{:.2f} minutes!'.format(average_time))


if __name__ == '__main__':
    # config = opts()
    # setproctitle.setproctitle('{}: Testing!'.format(args.user))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    main()

