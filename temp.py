import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import dataset
import gvt
import numpy as np
from torchvision import transforms
import nni
from nni.utils import merge_parameter
import os
import logging
import math
import time

from utils import save_checkpoint, setup_seed
from image import load_data
from config import args, return_args
from timm.models import create_model

setup_seed(args.seed)
logger = logging.getLogger('counting task')


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt



def main(args):
    train_file = './npydata/train.npy'
    test_file = './npydata/test.npy'

    with open(train_file,'rb') as outfile:
        train_list = np.load(outfile).tolist()
    with open(test_file,'rb') as outfile:
        val_list = np.load(outfile).tolist()

    print(len(train_list), len(val_list))

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu_id']

    model = create_model(
        args['model_type'],
        pretrained=args['pretrained'],
        drop_rate=args['drop'],
        drop_path_rate=args['drop_path'],
        drop_block_rate=None,
    )
    cudnn.benchmark = True
    model = nn.DataParallel(model, device_ids=[0])
    model = model.cuda()
    torch.set_num_threads(args['workers'])
    nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of params: {nparam}")

    criterion = nn.L1Loss(size_average=None).cuda()

    optimizer = torch.optim.Adam(
        [
            {'params': model.parameters(), 'lr': args['lr']},
        ], lr=args['lr'], weight_decay=args['weight_decay']
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200], gamma=0.1, last_epoch=-1)

    print(f"save path {args['save_path']}")
    if not os.path.exists(args['save_path']):
        os.makedirs(args['save_path'])

    print(args['start_epoch'])
    train_data = pre_data(train_list, args, train=True)
    val_data = pre_data(val_list, args, train=False)

    for epoch in range(args['start_epoch'], args['epochs']):
        start = time.time()
        train(train_data, model, criterion, optimizer, epoch, args, scheduler)
        end1 = time.time()

        if epoch % 5 == 0 and epoch >=5:
            prec1 = validate(val_data, model, args)
            end2 = time.time()
            if prec1 < args['best_pred']:
                is_best = True
                args['best_pred'] = prec1
            print(' * best MAE {mae:.3f} '.format(mae=args['best_pred']), args['save_path'], end1 - start, end2 - end1)

            # save_checkpoint({
            #         'epoch': epoch + 1,
            #         'arch': args['pre'],
            #         'state_dict': model.state_dict(),
            #         'best_prec1': args['best_pred'],
            #         'optimizer': optimizer.state_dict(),
            # }, is_best, args['save_path'])

def pre_data(datalist, args, train):
    print('loading dataset')
    data_keys = {}
    cnt = 0
    for i in range(len(datalist)):
        imgpath = datalist[i]
        fname = os.path.basename(imgpath)
        img, gt = load_data(imgpath, args, train)

        blob = {}
        blob['img'] = img
        blob['gt'] = gt
        blob['fname'] = fname
        data_keys[cnt] = blob
        cnt += 1

    return data_keys


def train(data, model, criterion, optimizer, epoch, args, scheduler):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(data, args['save_path'],
                            shuffle=True,
                            transform=transforms.Compose([
                                # transforms.Resize(args['input_size'], interpolation=3),
                                # transforms.RandomCrop(args['input_size'], padding=12),
                                transforms.ToTensor(),
                                # transforms.ColorJitter(0.4),
                                # transforms.RandomHorizontalFlip(),
                                # transforms.RandomVerticalFlip(),
                                # transforms.RandomRotation(30),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225]),

                            ]),
                            train=True,
                            batch_size=args['batch_size'],
                            num_workers=args['workers'],
                            args=args
                            ),
        batch_size=args['batch_size'], drop_last=False
    )
    args['lr'] = optimizer.param_groups[0]['lr']
    print(f"epoch {epoch}, processed {epoch*len(train_loader.dataset)} samples, lr {args['lr']: .10f}")

    model.train()
    end = time.time()

    for i, (fname, img, gt) in enumerate(train_loader):
        data_time.update(time.time() - end)
        img = img.cuda()
        out1 = model(img)
        gt = gt.type(torch.FloatTensor).cuda().unsqueeze(1)

        # print(f'{out1}\n{gt}')

        loss = criterion(out1, gt)
        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args['print_freq'] == 0:
            print('4_Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                .format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))

    scheduler.step()


def validate(data, model, args):
    print('testing')
    batch_size = 1
    val_loader = torch.utils.data.DataLoader(
        dataset.listDataset(data, args['save_path'],
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225]),
                            ]),
                            args=args,
                            train=False
                            ),
        batch_size=batch_size
    )

    model.eval()

    mae = 0.0
    mse = 0.0
    vis = []
    inx = 0

    for i, (fname, img, gt) in enumerate(val_loader):
        img = img.cuda()
        if len(img.shape) == 5:
            img = img.squeeze(0)
        if len(img.shape) == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            out1 = model(img)
            cnt = torch.sum(out1).item()


        gt = torch.sum(gt).item()

        # print(f'{out1}\n{gt}')

        mae += abs(gt - cnt)
        mse += abs(gt - cnt) * abs(gt - cnt)

        if i % 15 == 0:
            print('{fname} Gt {gt:.2f} Pred {pred}'.format(fname=fname[0], gt=gt, pred=cnt))

    mae = mae * 1.0 / (len(val_loader) * batch_size)
    mse = math.sqrt(mse / (len(val_loader) * batch_size))

    nni.report_intermediate_result(mae)
    print(' \n* MAE {mae:.3f}\n'.format(mae=mae), '* MSE {mse:.3f}'.format(mse=mse))

    return mae


if __name__ == '__main__':
    tuner_params = nni.get_next_parameter()
    logger.debug(tuner_params)
    params = vars(merge_parameter(return_args, tuner_params))
    # print(params)

    # params = vars(return_args)
    print(params)
    # main(params)
    main(params)
