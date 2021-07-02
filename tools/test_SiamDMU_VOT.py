# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker, build_tracker_for_test
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str

from FlowWrapping.models import FlowNet2C
import warnings

from deepmask.tools.InferDeepMask import Infer
import deepmask.models as mask_models

from updatenet.UpdateModel import UpdateModel, load_updatenet

parser = argparse.ArgumentParser(description='siamese model tracking')
parser.add_argument('--dataset', type=str,
        help='datasets', default='VOT2016')
parser.add_argument('--config', default='../experiments/siamrpn_r50_l234_dwxcorr/config.yaml', type=str,
        help='config file')
parser.add_argument('--snapshot', default='../experiments/siamrpn_r50_l234_dwxcorr/model.pth', type=str,
        help='snapshot of models to eval')
parser.add_argument('--video', default='', type=str,
        help='eval one special video')
parser.add_argument('--vis', action='store_true',
        help='whether visualzie result')

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

parser.add_argument('--resultsave', default='./', type=str,
        help="result save path")
parser.add_argument('--updatepath', default= '/media/HardDisk_new/wh/code/models_results/',
                    type=str, help="result save path")
parser.add_argument('--checkpoint', default='SiamDMU.pth.tar',
                    type=str, help="result save path")
parser.add_argument('--savename', default='SiamDMU',
                    type=str, help="result save name")
args = parser.parse_args()

# torch.set_num_threads(1)

# os.environ["CUDA_VISIBLE_DEVICES"] = "1, 0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(checkpoint):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load config
    cfg.merge_from_file(args.config)

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    # dataset_root = os.path.join('/media/HardDisk_new/wh/DataSet/for_pysot/Dataset_UAV123/UAV123/data_seq/', args.dataset)
    dataset_root = os.path.join('/media/HardDisk_new/wh/DataSet/for_pysot/', args.dataset)

    # dataset_root = '-' + args.dataset

    # create model
    model = ModelBuilder()

    # load model
    model = load_pretrain(model, args.snapshot)
    torch.nn.DataParallel(model, device_ids=[0])
    model.eval().to(device)

    # tracker = build_tracker(model)

    flownet = FlowNet2C(args)
    flownet.load_state_dict(torch.load(args.model_path)["state_dict"])
    torch.nn.DataParallel(flownet, device_ids=[0])
    flownet.eval().to(device)


    from collections import namedtuple
    Mask_Config = namedtuple('Config', ['iSz', 'oSz', 'gSz', 'batch'])
    mask_config = Mask_Config(iSz=160, oSz=56, gSz=112, batch=1)  # default for training

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
    updatenet = load_updatenet(updatenet, os.path.join(args.updatepath, checkpoint)).cuda().eval()

    # build tracker
    tracker = build_tracker_for_test(model, flownet, infer, updatenet)

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)


    model_name = args.savename
    total_lost = 0
    fps = 0
    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019']:
        # restart tracking
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue

            # if video.name != 'basketball':
            #     continue

            frame_counter = 0
            lost_number = 0
            toc = 0
            pred_bboxes = []
            for idx, (img, gt_bbox) in enumerate(video):
                if len(gt_bbox) == 4:
                    gt_bbox = [gt_bbox[0], gt_bbox[1],
                       gt_bbox[0], gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]]
                tic = cv2.getTickCount()
                if idx == frame_counter:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    tracker.init(img, gt_bbox_, args.dataset)
                    pred_bbox = gt_bbox_
                    pred_bboxes.append(1)
                elif idx > frame_counter:

                    try:
                        # outputs = tracker.track(img, float(alph), float(beta), float(lamb), float(gamma),
                        #                         video.name,idx)
                        outputs = tracker.track(img)
                    except:
                        print(idx)
                    pred_bbox = outputs['bbox']
                    if cfg.MASK.MASK:
                        pred_bbox = outputs['polygon']
                    overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))
                    if overlap > 0:
                        # not lost
                        pred_bboxes.append(pred_bbox)
                    else:
                        # lost object
                        pred_bboxes.append(2)
                        frame_counter = idx + 5 # skip 5 frames
                        lost_number += 1
                else:
                    pred_bboxes.append(0)
                toc += cv2.getTickCount() - tic
                if idx == 0:
                    cv2.destroyAllWindows()
                if args.vis and idx > frame_counter:
                    cv2.polylines(img, [np.array(gt_bbox, np.int).reshape((-1, 1, 2))],
                            True, (0, 255, 0), 3)
                    if cfg.MASK.MASK:
                        cv2.polylines(img, [np.array(pred_bbox, np.int).reshape((-1, 1, 2))],
                                True, (0, 255, 255), 3)
                    else:
                        bbox = list(map(int, pred_bbox))
                        cv2.rectangle(img, (bbox[0], bbox[1]),
                                      (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(img, str(lost_number), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)
            toc /= cv2.getTickFrequency()
            # save results
            # video_path = os.path.join('results', args.dataset, model_name,
            #         'baseline', video.name)
            video_path = os.path.join(args.resultsave, 'results', args.dataset, model_name,
                    'baseline', video.name)

            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    if isinstance(x, int):
                        f.write("{:d}\n".format(x))
                    else:
                        f.write(','.join([vot_float2str("%.4f", i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
                    v_idx+1, video.name, toc, idx / toc, lost_number))
            total_lost += lost_number
            fps += idx / toc
        print("{:s} total lost: {:d}".format(model_name, total_lost))
        print("{:s} fps: {:3.1f} fps".format(model_name, fps/(v_idx + 1)))
    else:
        # OPE tracking
        fps = 0
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            toc = 0
            pred_bboxes = []
            scores = []
            track_times = []
            for idx, (img, gt_bbox) in enumerate(video):
                tic = cv2.getTickCount()
                if idx == 0:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    tracker.init(img, gt_bbox_, args.dataset)
                    pred_bbox = gt_bbox_
                    scores.append(None)
                    if 'VOT2018-LT' == args.dataset:
                        pred_bboxes.append([1])
                    else:
                        pred_bboxes.append(pred_bbox)
                else:
                    try:
                        outputs = tracker.track(img)
                    except:
                        print(idx)
                    pred_bbox = outputs['bbox']
                    pred_bboxes.append(pred_bbox)
                    scores.append(outputs['best_score'])
                toc += cv2.getTickCount() - tic
                track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
                if idx == 0:
                    cv2.destroyAllWindows()
                if args.vis and idx > 0:
                    gt_bbox = list(map(int, gt_bbox))
                    pred_bbox = list(map(int, pred_bbox))
                    cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                  (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)
                    cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                  (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)
            toc /= cv2.getTickFrequency()
            fps += idx / toc
            # save results
            if 'VOT2018-LT' == args.dataset:
                video_path = os.path.join('results', args.dataset, model_name,
                        'longterm', video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path,
                        '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
                result_path = os.path.join(video_path,
                        '{}_001_confidence.value'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in scores:
                        f.write('\n') if x is None else f.write("{:.6f}\n".format(x))
                result_path = os.path.join(video_path,
                        '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            elif 'GOT-10k' == args.dataset:
                video_path = os.path.join('results', args.dataset, model_name, video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
                result_path = os.path.join(video_path,
                        '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            else:
                model_path = os.path.join(args.resultsave, 'results', args.dataset, model_name)
                if not os.path.isdir(model_path):
                    os.makedirs(model_path)
                result_path = os.path.join(model_path, '{}.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
                v_idx+1, video.name, toc, idx / toc))
    print("{:s} fps: {:3.1f} fps".format(model_name, fps / (v_idx + 1)))


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    main(args.checkpoint)
