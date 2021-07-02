# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch.nn.functional as F
import torch
import cv2
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cvbase as cvb

from pysot.core.config import cfg
from pysot.utils.anchor import Anchors
from pysot.tracker.base_tracker import SiameseTracker
from pysot.tracker.siamrpn_utils import get_new_template, generate_attention_im,  show_template, svd_normalize_
from pioneer.function_mask import function_mask,function_flow
'''
202102082308
'''

# SiameseRPN++ baseline
# class SiamRPNTracker(SiameseTracker):
#     def __init__(self, model):
#         super(SiamRPNTracker, self).__init__()
#         self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
#             cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
#         self.anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
#         hanning = np.hanning(self.score_size)
#         window = np.outer(hanning, hanning)
#         self.window = np.tile(window.flatten(), self.anchor_num)
#         self.anchors = self.generate_anchor(self.score_size)
#         self.model = model
#         self.model.eval()
#
#     def generate_anchor(self, score_size):
#         anchors = Anchors(cfg.ANCHOR.STRIDE,
#                           cfg.ANCHOR.RATIOS,
#                           cfg.ANCHOR.SCALES)
#         anchor = anchors.anchors
#         x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
#         anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)
#         total_stride = anchors.stride
#         anchor_num = anchor.shape[0]
#         anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
#         ori = - (score_size // 2) * total_stride
#         xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
#                              [ori + total_stride * dy for dy in range(score_size)])
#         xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
#             np.tile(yy.flatten(), (anchor_num, 1)).flatten()
#         anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
#         return anchor
#
#     def _convert_bbox(self, delta, anchor):
#         delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
#         delta = delta.data.cpu().numpy()
#
#         delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
#         delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
#         delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
#         delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
#         return delta
#
#     def _convert_score(self, score):
#         score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
#         score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
#         return score
#
#     def _bbox_clip(self, cx, cy, width, height, boundary):
#         cx = max(0, min(cx, boundary[1]))
#         cy = max(0, min(cy, boundary[0]))
#         width = max(10, min(width, boundary[1]))
#         height = max(10, min(height, boundary[0]))
#         return cx, cy, width, height
#
#     def init(self, img, bbox):
#         """
#         args:
#             img(np.ndarray): BGR image
#             bbox: (x, y, w, h) bbox
#         """
#         self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
#                                     bbox[1]+(bbox[3]-1)/2])
#         self.size = np.array([bbox[2], bbox[3]])
#
#         # calculate z crop size
#         w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
#         h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
#         s_z = round(np.sqrt(w_z * h_z))
#
#         # calculate channle average
#         self.channel_average = np.mean(img, axis=(0, 1))
#
#         # get crop
#         z_crop = self.get_subwindow(img, self.center_pos,
#                                     cfg.TRACK.EXEMPLAR_SIZE,
#                                     s_z, self.channel_average)
#         self.z_crop = z_crop
#         self.model.template(z_crop)
#
#     def track(self, img):
#         """
#         args:
#             img(np.ndarray): BGR image
#         return:
#             bbox(list):[x, y, width, height]
#         """
#         w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
#         h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
#         s_z = np.sqrt(w_z * h_z)
#         scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
#         s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
#         x_crop = self.get_subwindow(img, self.center_pos,
#                                     cfg.TRACK.INSTANCE_SIZE,
#                                     round(s_x), self.channel_average)
#
#         outputs = self.model.track(x_crop)
#
#         score = self._convert_score(outputs['cls'])
#         pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)
#
#         def change(r):
#             return np.maximum(r, 1. / r)
#
#         def sz(w, h):
#             pad = (w + h) * 0.5
#             return np.sqrt((w + pad) * (h + pad))
#
#         # scale penalty
#         s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
#                      (sz(self.size[0]*scale_z, self.size[1]*scale_z)))
#
#         # aspect ratio penalty
#         r_c = change((self.size[0]/self.size[1]) /
#                      (pred_bbox[2, :]/pred_bbox[3, :]))
#         penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
#         pscore = penalty * score
#
#         # window penalty
#         pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
#             self.window * cfg.TRACK.WINDOW_INFLUENCE
#         best_idx = np.argmax(pscore)
#
#         bbox = pred_bbox[:, best_idx] / scale_z
#         lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR
#
#         cx = bbox[0] + self.center_pos[0]
#         cy = bbox[1] + self.center_pos[1]
#
#         # smooth bbox
#         width = self.size[0] * (1 - lr) + bbox[2] * lr
#         height = self.size[1] * (1 - lr) + bbox[3] * lr
#
#         # clip boundary
#         cx, cy, width, height = self._bbox_clip(cx, cy, width,
#                                                 height, img.shape[:2])
#
#         # udpate state
#         self.center_pos = np.array([cx, cy])
#         self.size = np.array([width, height])
#
#         bbox = [cx - width / 2,
#                 cy - height / 2,
#                 width,
#                 height]
#         best_score = score[best_idx]
#         return {
#                 'bbox': bbox,
#                 'best_score': best_score
#                }


# Chinese paper + English paper + zl(2 templates) + zl(mask + update)
class SiamRPNTracker(SiameseTracker):
    def __init__(self, model, warpping_model, deepmask_model):
        super(SiamRPNTracker, self).__init__()
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        self.anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.anchors = self.generate_anchor(self.score_size)
        self.model = model
        self.model.eval()
        self.warpping_model = warpping_model
        self.deepmask_model = deepmask_model
        self.track_idx = 1

        # zl(2 templates)
        # self.motion_pool = []

        self.z_crop = None
        self.z_crop_ = None
        self.current_zrop = None
        self.new_template = None

        # Chinese paper idea 2
        self.svd_nor = None

    def generate_anchor(self, score_size):
        anchors = Anchors(cfg.ANCHOR.STRIDE,
                          cfg.ANCHOR.RATIOS,
                          cfg.ANCHOR.SCALES)
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
            np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    def _convert_bbox(self, delta, anchor):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    def _convert_score(self, score):
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """

        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        # show_template(img, 'img111', self.track_idx)

        img_att = generate_attention_im(img, bbox, self.deepmask_model)

        self.channel_average_ = np.mean(img_att, axis=(0, 1))
        #
        # # zl2 fill and crop
        z_crop_att = self.get_subwindow(img_att, self.center_pos,
                                        cfg.TRACK.EXEMPLAR_SIZE,
                                        s_z, self.channel_average_)

        # show_template(z_crop_att, 'initial', self.track_idx)

        # Chinese paper idea 2
        self.svd_nor = svd_normalize_(z_crop, 0, 100)

        self.z_crop = z_crop
        self.z_crop_ = 0.004 * z_crop_att
        # self.z_crop_ = z_crop_att
        self.model.template(self.z_crop)
        self.track_idx = 1
        self.template_att_pool = [0]

    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """

        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)

        # update
        if self.track_idx > 1:
            self.model.template(self.new_template)

        outputs = self.model.track(x_crop)

        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0]/self.size[1]) /
                     (pred_bbox[2, :]/pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]

        self.current_zrop = get_new_template(img, bbox)

        # show wrapped template of i th frame
        # motion_info, wrapped_img = self.warpping_model(torch.stack([self.z_crop, self.current_zrop], 2))
        # if self.track_idx == 250:
        #     show_template(wrapped_img, 'T_wrap', self.track_idx)

        # Chinese paper + English paper idea 1
        motion_info, flow2_up = self.warpping_model(torch.stack([self.z_crop, self.current_zrop], 2))
        motion_zero = torch.zeros(motion_info.shape).cuda()
        motion_info = torch.where(motion_info < 0, motion_zero, motion_info)

        # I0指导A0生成At‘
        att_ia = self.warpping_model.warp(self.z_crop_, flow2_up)

        # English paper idea 2
        if self.track_idx % 80 == 0:
            # show_template(img, 'img', self.track_idx)
            recent_att = generate_attention_im(img, bbox, self.deepmask_model)
            w_z_re = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
            h_z_re = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
            s_z_re = round(np.sqrt(w_z_re * h_z_re))
            channel_average_ = np.mean(recent_att, axis=(0, 1))
            recent_att_ = self.get_subwindow(recent_att, self.center_pos,
                                             cfg.TRACK.EXEMPLAR_SIZE,
                                             s_z_re, channel_average_)
            self.template_att_pool.append(recent_att_)

        # zl(mask + update)
        # if self.track_idx % 30 == 0:
        #     # show_template(img, 'dalunwen', self.track_idx)
        #     recent_att = generate_attention_im(img, bbox, self.deepmask_model)
        #     # show_template(recent_att, 'dalunwen', self.track_idx)
        #     w_z_re = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        #     h_z_re = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        #     s_z_re = round(np.sqrt(w_z_re * h_z_re))
        #     channel_average_ = np.mean(recent_att, axis=(0, 1))
        #     recent_att_ = self.get_subwindow(recent_att, self.center_pos,
        #                                      cfg.TRACK.EXEMPLAR_SIZE,
        #                                      s_z_re, channel_average_)
        #     self.template_att_pool.append(recent_att_)

        # English paper, 公式4-5
        def get_motion_template(T0, T_motion, att, recent_att):
            T_new = T0 + 0.05 * (T_motion + att) + 0.0005 * recent_att
            return T_new

        # Chinese paper
        def get_motion_template_Chinese(T0, T_motion, svd_nor):
            T_new = T0 + 0.01 * (T_motion + svd_nor)
            return T_new

        # zl(2 templates)
        def get_motion_template_zl(T0, track_idx, motion_pool):
            if track_idx == 1:
                T_new = T0 + 0.05 * motion_pool[0]
                return T_new
            else:
                T_new = T0 + 0.05 * motion_pool[1] + 0.02 * motion_pool[0]
                return T_new

        # zl(mask + update)
        def get_motion_template_zl2(T0, att, recent_att):
            T_new = T0 + att + 0.03 * recent_att
            return T_new

        # 公式4-5
        # def get_motion_template_new(T0, T_motion, att, recent_att):
        #     #T_new = T0 + 0.05 * (T_motion + att) + 0.0005 * recent_att
        #     T_new = T0 + 0.0002 * att + 0.0005 * recent_att + 0.05 * T_motion
        #     return T_new

        # English paper
        self.new_template = get_motion_template(self.z_crop, motion_info, self.z_crop_, self.template_att_pool[-1])

        # Chinese paper
        # self.new_template = get_motion_template_Chinese(self.z_crop, motion_info, self.svd_nor)

        # zl(2 templates)
        # if self.track_idx < 3:
        #     self.motion_pool.append(motion_info)
        #     self.new_template = get_motion_template_zl(self.z_crop, self.track_idx, self.motion_pool)
        # else:
        #     self.motion_pool[0] = self.motion_pool[1]
        #     self.motion_pool[1] = motion_info
        #     self.new_template = get_motion_template_zl(self.z_crop, self.track_idx, self.motion_pool)

        # zl(mask + update)
        # self.new_template = get_motion_template_zl2(self.z_crop, self.z_crop_, self.template_att_pool[-1])

        self.track_idx += 1

        return {
                'bbox': bbox,
                'best_score': best_score
               }

class SiamRPNTrackerUpdate(SiameseTracker):
    def __init__(self, model, warpping_model, deepmask_model, updatenet):
        super(SiamRPNTrackerUpdate, self).__init__()
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        self.anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.anchors = self.generate_anchor(self.score_size)
        self.model = model
        self.model.eval()
        self.warpping_model = warpping_model
        self.deepmask_model = deepmask_model
        self.updatenet_model = updatenet
        self.track_idx = 1

        # zl(2 templates)
        # self.motion_pool = []

        self.z_crop = None
        self.z_crop_ = None
        self.current_zrop = None
        self.new_template = None

        # Chinese paper idea 2
        self.svd_nor = None

    def generate_anchor(self, score_size):
        anchors = Anchors(cfg.ANCHOR.STRIDE,
                          cfg.ANCHOR.RATIOS,
                          cfg.ANCHOR.SCALES)
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
            np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    def _convert_bbox(self, delta, anchor):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    def _convert_score(self, score):
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, img, bbox, alph):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """

        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        # show_template(img, 'img111', self.track_idx)

        # import matplotlib.pyplot as plt
        # a = z_crop.cpu().numpy()
        # a = np.transpose(a, [0,2,3,1])
        # plt.imshow(a[0].astype(np.int))
        # plt.show()
        # plt.imsave('/media/HardDisk_new/wh/code/flownet2-pytorch-master/data/test4.jpg',a[0].astype(np.uint8))

        img_att = generate_attention_im(img, bbox, self.deepmask_model)

        self.channel_average_ = np.mean(img_att, axis=(0, 1))
        #
        # # zl2 fill and crop
        z_crop_att = self.get_subwindow(img_att, self.center_pos,
                                        cfg.TRACK.EXEMPLAR_SIZE,
                                        s_z, self.channel_average_)

        # show_template(z_crop_att, 'initial', self.track_idx)

        # Chinese paper idea 2
        # self.svd_nor = svd_normalize_(z_crop, 0, 100)

        self.z_crop = z_crop
        self.z_crop_att = z_crop_att
        self.model.template(self.z_crop)
        self.track_idx = 1

        self.idx_for_flow = 0
        self.current_zrop_memory_for_flow = torch.zeros([10,3,127,127]).cuda()
        self.current_zrop_memory_for_flow[0,:,:,:] = z_crop
        self.new_template = self.z_crop * function_mask(alph, self.z_crop_att)

    def track(self, img, alph, beta, lamb, gamma):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """

        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)

        # update
        if self.track_idx > 1:
            self.model.template(self.new_template.type(torch.float32))

        outputs = self.model.track(x_crop)

        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0]/self.size[1]) /
                     (pred_bbox[2, :]/pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]

        # import matplotlib.pyplot as plt
        # plt.imshow(img)
        # plt.gca().add_patch(plt.Rectangle(xy=(bbox[0], bbox[1]), width=bbox[2], height=bbox[3],
        #                                       color='red',
        #                                       fill=False, linewidth=2))
        # plt.show()

        self.current_zrop = get_new_template(img, bbox)
        self.idx_for_flow += 1

        if self.idx_for_flow == 1 or self.idx_for_flow % 1 == 0:  # self.idx_for_flow % 2 == 0:
            if self.idx_for_flow <= 9:
                # 1到9帧，都和初始帧(0帧)做光流
                crop_for_flow = torch.unsqueeze(self.current_zrop_memory_for_flow[0, :, :, :], 0)
            else:
                crop_for_flow = torch.unsqueeze(self.current_zrop_memory_for_flow[self.idx_for_flow % 10, :, :, :], 0)

            flow2_up = self.warpping_model(torch.stack([crop_for_flow, self.current_zrop], 2))
            flow2_up_rgb = flow2_up.data.cpu().numpy().squeeze().transpose(1, 2, 0)
            flow2_up_rgb = cvb.flow2rgb(flow2_up_rgb) * 255
            flow2_up_gray = np.dot(flow2_up_rgb, [0.299, 0.587, 0.114])
            self.flow2_up_gray = torch.tensor(flow2_up_gray).cuda()

        # 更新记忆存储
        self.current_zrop_memory_for_flow[self.idx_for_flow % 10, :, :, :] = self.current_zrop

        if self.idx_for_flow % 80 == 1:
            current_att = generate_attention_im(img, bbox, self.deepmask_model)
            w_z_re = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
            h_z_re = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
            s_z_re = round(np.sqrt(w_z_re * h_z_re))
            channel_average_ = np.mean(current_att, axis=(0, 1))
            self.current_att = self.get_subwindow(current_att, self.center_pos,
                                                  cfg.TRACK.EXEMPLAR_SIZE,
                                                  s_z_re, channel_average_)

        # import matplotlib.pyplot as plt
        # a = flow2_up_rgb.cpu().numpy()
        # a = np.transpose(a, [0, 2, 3, 1])
        # plt.imshow(a[0].astype(np.int))
        # plt.show()

        # updatenet
        t_0 = self.z_crop * function_mask(alph, self.z_crop_att)
        t_t = (self.current_zrop * (function_mask(beta, self.current_att) +
                                   function_flow(lamb, self.flow2_up_gray))).float()
        temp = torch.cat((Variable(t_0), Variable(self.new_template), Variable(t_t)), 1)
        self.new_template = self.updatenet_model(temp, (Variable(t_0)))

        # print('max:{max},min:{min}'.format(max=torch.max(unout),min=torch.min(unout)))

        # self.new_template = self.z_crop

        # liner
        # self.new_template = self.z_crop * function_mask(alph, self.z_crop_att) + \
        #                     gamma * self.current_zrop * (function_mask(beta, self.current_att) + function_flow(lamb, self.flow2_up_gray))
        #
        # plt.subplot(4,3,1)
        # c = self.z_crop.cpu().numpy()
        # c = np.transpose(c, [0, 2, 3, 1])
        # plt.imshow(c[0].astype(np.int))
        # plt.title('z_crop')
        #
        # plt.subplot(4, 3, 2)
        # c = self.current_zrop.cpu().numpy()
        # c = np.transpose(c, [0, 2, 3, 1])
        # plt.imshow(c[0].astype(np.int))
        # plt.title('current_zrop')
        #
        # plt.subplot(4, 3, 4)
        # c = self.z_crop_att.cpu().numpy()
        # c = np.transpose(c, [0, 2, 3, 1])
        # plt.imshow(c[0].astype(np.int))
        # plt.title('z_crop_att')
        #
        # plt.subplot(4, 3, 5)
        # a = self.current_att.cpu().numpy()
        # a = np.transpose(a, [0,2,3,1])
        # plt.imshow(a[0].astype(np.int))
        # plt.title('current_att')
        #
        # plt.subplot(4, 3, 6)
        # a = self.flow2_up_gray.cpu().numpy()
        # plt.imshow(a.astype(np.int))
        # plt.title('flow2_up_gray')
        #
        # plt.subplot(4, 3, 7)
        # a = (self.z_crop * function_mask(alph, self.z_crop_att)).cpu().numpy()
        # a = np.transpose(a, [0,2,3,1])
        # plt.imshow(a[0].astype(np.int))
        # plt.title('t0*mask_a0')
        #
        # plt.subplot(4, 3, 8)
        # a = (self.current_zrop * function_mask(beta, self.current_att)).cpu().numpy()
        # a = np.transpose(a, [0,2,3,1])
        # plt.imshow(a[0].astype(np.int))
        # plt.title('t_t*mask_at')
        #
        # plt.subplot(4, 3, 9)
        # a = (self.current_zrop * function_flow(lamb, self.flow2_up_gray)).cpu().numpy()
        # a = np.transpose(a, [0,2,3,1])
        # plt.imshow(a[0].astype(np.int))
        # plt.title('t_t*flow')
        #
        # plt.subplot(4, 3, 11)
        # a = self.current_zrop * (function_mask(beta, self.current_att) + function_flow(lamb, self.flow2_up_gray))
        # a = a.cpu().numpy()
        # a = np.transpose(a, [0, 2, 3, 1])
        # plt.imshow(a[0].astype(np.int))
        # plt.title('t_t*(mask_at+flow)')
        #
        # plt.subplot(4, 3, 12)
        # a = self.new_template.cpu().detach().numpy()
        # a = np.transpose(a, [0, 2, 3, 1])
        # plt.imshow(a[0].astype(np.int))
        # plt.title('new_template')

        # plt.subplot(4, 3, 10)
        # a = (unout).cpu().detach().numpy()
        # a = np.transpose(a, [0, 2, 3, 1])
        # plt.imshow(a[0].astype(np.int))
        # plt.title('U.N.output')

        # plt.show()

        self.track_idx += 1

        return {
                'bbox': bbox,
                'best_score': best_score
               }

class SiamRPNTrackerForTrain(SiameseTracker):
    def __init__(self, model, warpping_model, deepmask_model, updatenet):
        super(SiamRPNTrackerForTrain, self).__init__()
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        self.anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.anchors = self.generate_anchor(self.score_size)
        self.model = model
        self.model.eval()
        self.warpping_model = warpping_model
        self.deepmask_model = deepmask_model
        self.updatenet_model = updatenet
        self.track_idx = 1

        # zl(2 templates)
        # self.motion_pool = []

        self.z_crop = None
        self.z_crop_ = None
        self.current_zrop = None
        self.new_template = None

        # Chinese paper idea 2
        self.svd_nor = None

    def generate_anchor(self, score_size):
        anchors = Anchors(cfg.ANCHOR.STRIDE,
                          cfg.ANCHOR.RATIOS,
                          cfg.ANCHOR.SCALES)
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
            np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    def _convert_bbox(self, delta, anchor):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    def _convert_score(self, score):
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, img, bbox, alph):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """

        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        # show_template(img, 'img111', self.track_idx)

        # import matplotlib.pyplot as plt
        # a = z_crop.cpu().numpy()
        # a = np.transpose(a, [0,2,3,1])
        # plt.imshow(a[0].astype(np.int))
        # plt.show()
        # plt.imsave('/media/HardDisk_new/wh/code/flownet2-pytorch-master/data/test4.jpg',a[0].astype(np.uint8))

        img_att = generate_attention_im(img, bbox, self.deepmask_model)

        self.channel_average_ = np.mean(img_att, axis=(0, 1))
        #
        # # zl2 fill and crop
        z_crop_att = self.get_subwindow(img_att, self.center_pos,
                                        cfg.TRACK.EXEMPLAR_SIZE,
                                        s_z, self.channel_average_)

        # show_template(z_crop_att, 'initial', self.track_idx)

        # Chinese paper idea 2
        self.svd_nor = svd_normalize_(z_crop, 0, 100)

        self.z_crop = z_crop
        self.z_crop_att = z_crop_att
        self.model.template(self.z_crop)
        self.track_idx = 1

        self.idx_for_flow = 0
        self.current_zrop_memory_for_flow = torch.zeros([10,3,127,127]).cuda()
        self.current_zrop_memory_for_flow[0,:,:,:] = z_crop
        self.new_template = self.z_crop * function_mask(alph, self.z_crop_att)

        self.state = dict()
        self.state['z_f_cur'] = (self.z_crop * function_mask(alph, self.z_crop_att)).cpu().data  # 当前检测特征图
        self.state['z_f'] = (self.z_crop * function_mask(alph, self.z_crop_att)).cpu().data  # 累积的特征图
        self.state['gt_f_cur'] = (self.z_crop * function_mask(alph, self.z_crop_att)).cpu().data  # gt框对应的特征图
        return self.state

    def track(self, img, gt, updatenet, alph, beta, lamb, gamma):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """

        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)

        # update
        if self.track_idx > 1:
            self.model.template(self.new_template.type(torch.float32))

        outputs = self.model.track(x_crop)

        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0] * scale_z, self.size[1] * scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0] / self.size[1]) /
                     (pred_bbox[2, :] / pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                 self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]

        # if self.track_idx % 50 == 0:
        #     import matplotlib.pyplot as plt
        #     plt.imshow(img)
        #     plt.gca().add_patch(plt.Rectangle(xy=(bbox[0], bbox[1]), width=bbox[2], height=bbox[3],
        #                                       color='red',
        #                                       fill=False, linewidth=2))
        #     plt.show()

        # 直接使用groundtruth作为跟踪结果
        # bbox = gt
        # self.size = np.array([gt[2], gt[3]])
        # self.center_pos = np.array([(2*gt[0]+gt[2])/2, (2*gt[1]+gt[3])/2])

        self.current_zrop = get_new_template(img, bbox)
        self.idx_for_flow += 1

        if self.idx_for_flow == 1 or self.idx_for_flow % 2 == 0:
            if self.idx_for_flow <= 9:
                # 1到9帧，都和初始帧(0帧)做光流
                crop_for_flow = torch.unsqueeze(self.current_zrop_memory_for_flow[0, :, :, :], 0)
            else:
                crop_for_flow = torch.unsqueeze(self.current_zrop_memory_for_flow[self.idx_for_flow % 10, :, :, :], 0)

            flow2_up = self.warpping_model(torch.stack([crop_for_flow, self.current_zrop], 2))
            flow2_up_rgb = flow2_up.data.cpu().numpy().squeeze().transpose(1, 2, 0)
            flow2_up_rgb = cvb.flow2rgb(flow2_up_rgb) * 255
            flow2_up_gray = np.dot(flow2_up_rgb, [0.299, 0.587, 0.114])
            self.flow2_up_gray = torch.tensor(flow2_up_gray).cuda()

        # 更新记忆存储
        self.current_zrop_memory_for_flow[self.idx_for_flow % 10, :, :, :] = self.current_zrop

        if self.idx_for_flow % 80 == 1:
            current_att = generate_attention_im(img, bbox, self.deepmask_model)
            w_z_re = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
            h_z_re = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
            s_z_re = round(np.sqrt(w_z_re * h_z_re))
            channel_average_ = np.mean(current_att, axis=(0, 1))
            self.current_att = self.get_subwindow(current_att, self.center_pos,
                                                  cfg.TRACK.EXEMPLAR_SIZE,
                                                  s_z_re, channel_average_)




        # updatenet
        t_0 = self.z_crop * function_mask(alph, self.z_crop_att)
        t_t = (self.current_zrop * (function_mask(beta, self.current_att) +
                                   function_flow(lamb, self.flow2_up_gray))).float()
        if updatenet == '':
            self.new_template = t_0 + gamma * t_t
        else:
            temp = torch.cat((Variable(t_0), Variable(self.new_template), Variable(t_t)),1)
            self.new_template = self.updatenet_model(temp, (Variable(t_0)))

        self.state['z_f'] = self.new_template.cpu().data
        self.state['z_f_cur'] = t_t.cpu().data

        # 生成当前帧的gt
        gt_size = np.array([gt[2], gt[3]])
        gt_center_pos = np.array([(2*gt[0]+gt[2])/2, (2*gt[1]+gt[3])/2])
        gt_crop = get_new_template(img, gt)

        gt_sizew = gt_size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(gt_size)
        gt_sizeh = gt_size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(gt_size)
        gt_s_z = round(np.sqrt(gt_sizew * gt_sizeh))
        gt_img_att = generate_attention_im(img, gt, self.deepmask_model)
        gt_channel_average_ = np.mean(gt_img_att, axis=(0, 1))
        gt_crop_att = self.get_subwindow(gt_img_att, gt_center_pos,
                                        cfg.TRACK.EXEMPLAR_SIZE,
                                        gt_s_z, gt_channel_average_)

        self.state['gt_f_cur'] = (gt_crop * function_mask(alph, gt_crop_att)).cpu().data # 当前帧gt框对应的特征模板

        self.state['target_pos'] = self.center_pos
        self.state['target_sz'] = self.size

        self.track_idx += 1

        return self.state

