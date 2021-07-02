from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
import cv2

from pysot.core.config import cfg
from pysot.utils.bbox import get_axis_aligned_bbox
from FlowWrapping.warpping_function import WarppingFunction


def get_subwindow(im, pos, model_sz, original_sz, avg_chans):
    if isinstance(pos, float):
        pos = [pos, pos]
    # print(isinstance(pos, float))
    sz = original_sz
    im_sz = im.shape
    c = (original_sz + 1) / 2
    # context_xmin = round(pos[0] - c) # py2 and py3 round
    context_xmin = np.floor(pos[0] - c + 0.5)
    context_xmax = context_xmin + sz - 1
    # context_ymin = round(pos[1] - c)
    context_ymin = np.floor(pos[1] - c + 0.5)
    context_ymax = context_ymin + sz - 1
    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    r, c, k = im.shape
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
        te_im = np.zeros(size, np.uint8)
        te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad:, :] = avg_chans
        im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                   int(context_xmin):int(context_xmax + 1), :]
    else:
        im_patch = im[int(context_ymin):int(context_ymax + 1),
                   int(context_xmin):int(context_xmax + 1), :]

    if not np.array_equal(model_sz, original_sz):
        # im_patch = svd_background(im_patch, 0.01, 1)
        im_patch = cv2.resize(im_patch, (model_sz, model_sz))
    im_patch = im_patch.transpose(2, 0, 1)
    im_patch = im_patch[np.newaxis, :, :, :]
    im_patch = im_patch.astype(np.float32)
    im_patch = torch.from_numpy(im_patch)
    if cfg.CUDA:
        im_patch = im_patch.cuda()
    return im_patch


def get_new_template(img, bbox):
    center_positon = np.array([bbox[0] + (bbox[2] - 1) / 2,
                               bbox[1] + (bbox[3] - 1) / 2])
    ori_size = np.array([bbox[2], bbox[3]])

    # calculate z crop size
    w_z = ori_size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(ori_size)
    h_z = ori_size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(ori_size)
    s_z = round(np.sqrt(w_z * h_z))

    # calculate channle average
    channel_average_vaule = np.mean(img, axis=(0, 1))

    # get crop
    current_z_crop = get_subwindow(img, center_positon,
                                   cfg.TRACK.EXEMPLAR_SIZE,
                                   s_z, channel_average_vaule)
    return current_z_crop


def restore(sigma, u, v, K):
    m = len(u)
    n = len(v[0])
    a = np.zeros((m, n))
    for k in range(K):
        uk = u[:, k].reshape(m, 1)
        vk = v[k].reshape(1, n)
        a += sigma[k] * np.dot(uk, vk)
    a = a.clip(0, 255)
    return np.rint(a)


def svd_normalize_(img, alph, k):
    img = img.cpu().detach().squeeze(0).numpy().transpose(2, 1, 0)
    # img_grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    u, sigma, v = [0, 0, 0], [0, 0, 0], [0, 0, 0]
    img_re = np.zeros(img.shape)
    for i in range(img.shape[2]):
        u[i], sigma[i], v[i] = np.linalg.svd(img[:, :, i])
        img_re[:, :, i] = alph * restore(sigma[i], u[i], v[i], k)

    img_re = torch.from_numpy(img_re.astype(np.float32).transpose(2, 1, 0))
    img_re = img_re.unsqueeze(0)
    return img_re.cuda()


def generate_attention_im(frame, bbox, infer):

    '''
    (1)只用masks中对应scores最大的一个（第0个）
    (2)输入为bbox的w+p,h+p,p=(w+h)/43
    '''

    frame_ = frame.copy()
    im = frame_[:, :, ::-1].copy().astype(np.float32)
    frame_1 = np.zeros(list(frame.shape))

    cx, cy, w, h = get_axis_aligned_bbox(np.array(bbox))
    p = (w + h) / 6
    #p = 0
    w_ = w + p
    h_ = h + p
    bbox_ = [cx - (w_) / 2, cy - (h_) / 2, w_, h_]

    im_ = im[max(0, int(bbox_[1])):min(int(bbox_[1] + bbox_[3]), im.shape[0]),
             max(0, int(bbox_[0])):min(int(bbox_[0] + bbox_[2]), im.shape[1]), :]
    im_gt = im[max(0, int(bbox[1])):min(int(bbox[1] + bbox[3]), im.shape[0]),
               max(0, int(bbox[0])):min(int(bbox[0] + bbox[2]), im.shape[1]), :]



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img = np.expand_dims(np.transpose(im_, (2, 0, 1)), axis=0).astype(np.float32)
    img = torch.from_numpy(img / 255.).to(device)
    with torch.no_grad():
        infer.forward(img)
        masks, scores = infer.getTopProps(.2, im_.shape[0], im_.shape[1])

    masks_x = (masks.shape[1] - w)/2
    masks_y = (masks.shape[0] - h)/2

    masks_ = masks[max(0, int(masks_y)):int(masks_y) + im_gt.shape[0], max(0, int(masks_x)):int(masks_x) + im_gt.shape[1], :]


    # import matplotlib.pyplot as plt
    # plt.figure(1)
    # plt.imshow(masks[:, :, 0].astype(np.int))
    # plt.show()
    # plt.figure(2)
    # plt.imshow(im_gt.astype(np.int))
    # plt.show()
    # plt.figure(3)
    # plt.imshow(masks_[:, :, 0].astype(np.int))
    # plt.show()
    # plt.figure(4)
    # plt.imshow(im_.astype(np.int))
    # plt.show()

    for i in range(im_gt.shape[2]):
        # for j in range(masks.shape[2]):
        #     im_[:, :, i] = masks[:, :, j] * 255. + (1. - masks[:, :, j]) * im_[:, :, i]
        # im_gt[:, :, i] = masks_[:, :, 0] * 255. + (1. - masks_[:, :, 0]) * im_gt[:, :, i]
        im_gt[:, :, i] = masks_[:, :, 0] * 255.

    frame_1[max(0, int(bbox[1])):int(bbox[1] +bbox[3]), max(0, int(bbox[0])):int(bbox[0] +bbox[2]), :] = im_gt[:, :, ::-1]
    # from matplotlib import pyplot as plt
    # plt.figure()
    # plt.imshow(frame_)
    # plt.show()
    return frame_1

# def generate_attention_im(frame, bbox, infer):
#     frame_ = frame.copy()
#     im = frame_[:, :, ::-1].copy().astype(np.float32)
#
#     im_ = im[max(0, int(bbox[1])):int(bbox[1] + bbox[3]), max(0, int(bbox[0])):int(bbox[0] + bbox[2]), :]
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     h, w = im_.shape[:2]
#     img = np.expand_dims(np.transpose(im_, (2, 0, 1)), axis=0).astype(np.float32)
#     img = torch.from_numpy(img / 255.).to(device)
#     infer.forward(img)
#     masks, scores = infer.getTopProps(.2, h, w)
#
#     for i in range(im_.shape[2]):
#         # for j in range(masks.shape[2]):
#             # im_[:, :, i] = masks[:, :, j] * 255. + (1. - masks[:, :, j]) * im_[:, :, i]
#         im_[:, :, i] = masks[:, :, 0] * 255. + (1. - masks[:, :, 0]) * im_[:, :, i]
#
#     # from matplotlib import pyplot as plt
#     # plt.figure()
#     # plt.imshow(im_.astype(np.uint8))
#     # plt.axis('off')
#     # plt.show()
#
#     frame_[max(0, int(bbox[1])):int(bbox[1] + bbox[3]), max(0, int(bbox[0])):int(bbox[0] + bbox[2]), :] = im_[:, :, ::-1]
#     # from matplotlib import pyplot as plt
#     # plt.figure()
#     # plt.imshow(frame_)
#     # plt.show()
#     return frame_

def show_template(template, name, idx):
    import os
    save_path = '/media/ubuntu/hd_330/users/hzh/pysot-master/figures'
    idx_str = '{} th'.format(idx)
    filename = os.path.join(save_path, name + '{}'.format(idx_str) + '.png')
    if len(template.shape) == 4:
        print(template.shape)
        print('{}th frame {}'.format(idx, name))
        template = template.cpu().detach().squeeze(0).numpy().transpose(1, 2, 0).astype(np.int)
        from matplotlib import pyplot as plt
        plt.figure()
        plt.imshow(template[:, :, ::-1])
        plt.show()
        cv2.imwrite(filename, template)
    elif len(template.shape) == 3:
        print(template.shape)
        if not isinstance(template, np.ndarray):
            template = template.cpu().detach().numpy().astype(np.int)
        from matplotlib import pyplot as plt
        plt.figure()
        plt.imshow(template)
        plt.show()
        cv2.imwrite(filename, template)
    else:
        print("error")



