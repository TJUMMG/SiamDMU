import argparse
import shutil
import os
import cv2  # imread
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
from os.path import join, isdir, isfile
from os import makedirs

from updatenet.utils_upd import get_axis_aligned_bbox, cxy_wh_2_rect, overlap_ratio #loss get_axis_aligned_rect function
from updatenet.UpdateModel import UpdateModel, load_updatenet
import warnings

warnings.filterwarnings("ignore")

#from scipy import io

parser = argparse.ArgumentParser(description='Training  in Pytorch 0.4.0')
parser.add_argument('--input_sz', dest='input_sz', default=125, type=int, help='crop input size')
parser.add_argument('--padding', dest='padding', default=2.0, type=float, help='crop padding size')
parser.add_argument('--range', dest='range', default=10, type=int, help='select range')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-5, type=float,
                    metavar='W', help='weight decay (default: 5e-5)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--save_path', '-s', default='/home/wh/', type=str, help='directory for saving')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print(args)
best_loss = 1e6

dataram = dict()
 
#训练数据路径
tem_path ='/home/wh/update_set_templates_step1_std/'



video_npy_list = os.listdir(os.path.join(tem_path,'gt'))
video_npy_list.sort()
for npy_name in ['template','templatei','template0','init0','init','pre','gt']:
# for npy_name in ['init0', 'init', 'pre', 'gt']:
    tem_npy = np.load(os.path.join(tem_path, npy_name, video_npy_list[0]))
    for video_npy in video_npy_list[1:]:
        tem_npy_new = np.load(os.path.join(tem_path, npy_name, video_npy))
        tem_npy = np.append(tem_npy,tem_npy_new,axis=0)
    dataram[npy_name] = tem_npy
dataram['train'] = np.arange(len(dataram['gt']), dtype=np.int)

# optionally resume from a checkpoint
if args.resume:
    if isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

cudnn.benchmark = True

save_path = args.save_path


def adjust_learning_rate(optimizer, epoch, lr0):
    lr = np.logspace(-lr0[0], -lr0[1], num=args.epochs)[epoch]
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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

def save_checkpoint(state, epoch,lr, filename=join(save_path, 'checkpoint.pth.tar')):
    name0 = 'lr' + str(lr[0])+str(lr[1])
    name0 = name0.replace('.','_')
    epo_path = join(save_path, name0)
    if not isdir(epo_path):
        makedirs(epo_path)
    if (epoch+1) % 1 == 0:
        filename=join(epo_path, 'checkpoint{}.pth.tar'.format(epoch+1))
        torch.save(state, filename)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    #random.seed(seed)
    torch.backends.cudnn.deterministic = True

lrs=np.array([[6,7],[7,8],[8,9],[9,10]])

for ii in np.arange(0, lrs.shape[0]):

    setup_seed(20)

    # construct model
    model = UpdateModel()


    # 初始化参数
    model.apply(weights_init)
    
    #加载预训练模型
    # updatenet_path = '/media/HardDisk_new/lr67/checkpoint45.pth.tar'
    # model = load_updatenet(model, updatenet_path)


    model = nn.DataParallel(model,device_ids=[0])#开启多GPU
    model.train().cuda()
    
    criterion = nn.MSELoss(size_average=False).cuda()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01,
                            weight_decay=args.weight_decay)

    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer, epoch, lrs[ii])

        losses = AverageMeter()
        subset = np.random.permutation(dataram['train'])

        for t in range(0, len(subset), args.batch_size):
            
            batchStart = t
            batchEnd = min(t+args.batch_size, len(subset))
            batch = subset[batchStart:batchEnd]
            init_index = dataram['init0'][batch]
            pre_index = dataram['pre'][batch]
            gt_index = dataram['gt'][batch]
            
            # reset diff T0
            for rr in range(len(init_index)):
                if init_index[rr] != 0:#不是初始帧
                    init_index[rr] = np.random.choice(init_index[rr],1)
            cur = dataram['templatei'][batch]
            init = dataram['template0'][batch-init_index]
            pre = dataram['template'][batch-pre_index]
            gt = dataram['template0'][batch+gt_index-1]
            
            #pdb.set_trace() 
            temp = np.concatenate((init, pre, cur), axis=1)
            input_up = torch.Tensor(temp)
            target = torch.Tensor(gt)
            init_inp = Variable(torch.Tensor(init)).cuda()#
            input_up = Variable(input_up).cuda()
            target = Variable(target).cuda()
            
            # compute output
            output = model(input_up, init_inp)
            loss = criterion(output, target)/target.size(0)

            # measure accuracy and record loss
            loss_data=loss.cpu().data.numpy().tolist()           
            losses.update(loss_data)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if t % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                       str(epoch).zfill(2), str(t).zfill(5), len(subset), loss=losses))     
        save_checkpoint({'state_dict': model.state_dict()}, epoch,lrs[ii])

        if epoch == 0 or epoch % 49 == 0:
            path = os.path.join(save_path, '1.txt')
            with open(path, 'a') as file:
                write_in = str(lrs[ii]) + ' ' + str(epoch) + ' ' + str(losses.avg)
                file.write(write_in)
                file.write('\n')



