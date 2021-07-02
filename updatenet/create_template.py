import sys
import cv2  # imread
import torch
import numpy as np
from os.path import realpath, dirname, join
import os
import pdb
import argparse

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker, build_tracker_for_train
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str

from FlowWrapping.models import FlowNet2C

from deepmask.tools.InferDeepMask import Infer
import deepmask.models as mask_models
from updatenet.utils_upd import get_axis_aligned_bbox, cxy_wh_2_rect, overlap_ratio #loss get_axis_aligned_rect function
from updatenet.UpdateModel import UpdateModel
import warnings

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='siamese model tracking')
parser.add_argument('--config', default='../experiments/siamrpn_r50_l234_dwxcorr/config.yaml', type=str,
        help='config file')
parser.add_argument('--snapshot', default='../experiments/siamrpn_r50_l234_dwxcorr/model.pth', type=str,
        help='snapshot of models to eval')

# warpping model parameters
parser.add_argument('--fp16', action='store_true',
                    help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
parser.add_argument("--rgb_max", type=float, default=255.)
parser.add_argument('--model_path', type=str,
                    default='../FlowWrapping/pretrained_model/FlowNet2-C_checkpoint.pth.tar')

# deepmask model parameters
mask_model_names = sorted(name for name in mask_models.__dict__
                     if not name.startswith("__") and callable(mask_models.__dict__[name]))

parser.add_argument('--arch', '-a', metavar='ARCH', default='DeepMask', choices=mask_model_names,
                    help='model architecture: ' + ' | '.join(mask_model_names) + ' (default: DeepMask)')
parser.add_argument('--resume', default='../deepmask/pretrained/DeepMask.pth.tar',
                    type=str, metavar='PATH', help='path to checkpoint')
parser.add_argument('--nps', default=10, type=int,
                    help='number of proposals to save in test')
parser.add_argument('--si', default=-2.5, type=float, help='initial scale')
parser.add_argument('--sf', default=.5, type=float, help='final scale')
parser.add_argument('--ss', default=.5, type=float, help='scale step')
args = parser.parse_args()



os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# load config
cfg.merge_from_file(args.config)

# create model
model = ModelBuilder()

# load model
model = load_pretrain(model, args.snapshot)
torch.nn.DataParallel(model, device_ids=[0])
model.eval().to(device)

# FlowNet
flownet = FlowNet2C(args)
flownet.load_state_dict(torch.load(args.model_path)["state_dict"])
torch.nn.DataParallel(flownet, device_ids=[0])
flownet.eval().to(device)

# DeepMask
from collections import namedtuple

Mask_Config = namedtuple('Config', ['iSz', 'oSz', 'gSz', 'batch'])
mask_config = Mask_Config(iSz=160, oSz=56, gSz=112, batch=1)

deepmask_model = (mask_models.__dict__[args.arch](mask_config))
deepmask_model = load_pretrain(deepmask_model, args.resume)
deepmask_model = deepmask_model.eval().to(device)


def range_end(start, stop, step=1):
    return np.arange(start, stop + step, step)


scales = [2 ** i for i in range_end(args.si, args.sf, args.ss)]
meanstd = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
infer = Infer(nps=args.nps, scales=scales, meanstd=meanstd, model=deepmask_model, device=device)

# UpdateNet
updatenet = UpdateModel()

# build tracker
tracker = build_tracker_for_train(model, flownet, infer, updatenet.cuda())


step=1
if step>1:
    update_model=torch.load('/media/HardDisk_new/wh/code/pysot-master/pysot_20210505/updatenet/train_sample10/2_2/lr78/checkpoint50.pth.tar')['state_dict']

    update_model_fix = dict()
    for i in update_model.keys():
       update_model_fix['.'.join(i.split('.')[1:])] = update_model[i]

    tracker.updatenet_model.load_state_dict(update_model_fix)
    tracker.updatenet_model.eval().cuda()
    updatenet = 'yes'
else:
    updatenet=''
    tracker.model.cuda()
reset = 1; frame_max = 300
setfile = 'update_set'
temp_path ='/home/wh/update_set_templates_step'+str(step)+'_std'  #step=1,2,3

for i in ['template','templatei','template0','init0','init','pre','gt']:
    if not os.path.isdir(join(temp_path,i)):
        os.makedirs(join(temp_path,i))

video_path = '/media/HardDisk_new/wh/code/lasot_small/'
lists = open('/media/HardDisk_new/wh/code/SiamDMU/updatenet/'+setfile+'.txt','r')
list_file = [line.strip() for line in lists]
category = os.listdir(video_path)
category.sort()



for video in category[:]:

    # video = 'drone-1'

    if video not in list_file:
        continue
    print(video)

    template_acc = []
    template_cur = []
    template_gt = []

    init0 = []
    init = []
    pre = []
    gt = []

    gt_path = join(video_path,video, 'groundtruth.txt')

    ground_truth = np.loadtxt(gt_path, delimiter=',')

    num_frames = len(ground_truth)

    img_path = join(video_path,video, 'img')

    imgFiles = [join(img_path,'%08d.jpg') % i for i in range(1,num_frames+1)]

    frame = 0


    while frame < num_frames:

        Polygon = ground_truth[frame]
        cx, cy, w, h = get_axis_aligned_bbox(Polygon)

        if w*h!=0:

            image_file = imgFiles[frame]
            target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
            im = cv2.imread(image_file)  # HxWxC


            try:
                state = tracker.init(im, Polygon)
            except:
                print(video,' init ', frame)
                frame = frame + 1
                continue


            template_acc.append(state['z_f'])

            template_cur.append(state['z_f_cur'])

            template_gt.append(state['gt_f_cur'])

            init0.append(0); init.append(frame); frame_reset=0; pre.append(0);  gt.append(1)

            #初始化结束,开始跟踪
            while frame < (num_frames-1):

                frame = frame + 1; frame_reset=frame_reset+1

                image_file = imgFiles[frame]

                if not image_file:
                    break 
                
                Polygon = ground_truth[frame]
                cx, cy, w, h = get_axis_aligned_bbox(Polygon)
                gt_pos, gt_sz = np.array([cx, cy]), np.array([w, h])
                state['gt_pos']=gt_pos
                state['gt_sz']=gt_sz

                im = cv2.imread(image_file)

                idx=frame


                state = tracker.track(im, Polygon, updatenet)

                template_acc.append(state['z_f'])#累积模板

                template_cur.append(state['z_f_cur'])#检测模板

                template_gt.append(state['gt_f_cur'])#当前帧gt框对应的特征图

                init0.append(frame_reset); init.append(frame); pre.append(1); 

                if frame==(num_frames-1):
                    gt.append(0)
                else:
                    gt.append(1)

                res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
                #计算当前检测的框和gt框的iou，如果不相交则丢失目标
                if reset:
                    rect=ground_truth[frame]
                    gt_rect=np.array([rect[0]-1,rect[1]-1,rect[2],rect[3]])
                    iou = overlap_ratio(gt_rect, res)
                    if iou<=0.5:
                        break    
        else:
            template_acc.append(torch.zeros([1, 3, 127, 127], dtype=torch.float32))
            template_cur.append(torch.zeros([1, 3, 127, 127], dtype=torch.float32))
            template_gt.append(torch.zeros([1, 3, 127, 127], dtype=torch.float32))
            init0.append(0); init.append(frame); pre.append(1)

            if frame==(num_frames-1):
                gt.append(0)
            else:
                gt.append(1)  

        frame = frame + 1

    template_acc = np.concatenate(template_acc)  # 累积模板
    template_cur = np.concatenate(template_cur)  # 当前检测模板
    template_gt = np.concatenate(template_gt)  # gt特征图

    np.save(temp_path + '/template/' + video, template_acc)  # 累积特征图
    np.save(temp_path + '/templatei/' + video, template_cur)  # 当前的检测模板
    np.save(temp_path + '/template0/' + video, template_gt)  # gt模板
    np.save(temp_path + '/init0/' + video, init0)  # 第一帧
    np.save(temp_path + '/init/' + video, init)  # 累积模板编号
    np.save(temp_path + '/pre/' + video, pre)  # 每一帧对应的检测模板 =1
    np.save(temp_path + '/gt/' + video, gt)  # 每一帧对应的gt,一般=1
