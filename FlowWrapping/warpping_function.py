from FlowWrapping.models import FlowNet2C
from FlowWrapping.utils.frame_utils import read_gen

import torch
import numpy as np
import argparse
import os
from matplotlib import pyplot as plt
import cv2


def restore(sigma, u, v, K):  # 奇异值、左特征向量、右特征向量
    m = len(u)
    n = len(v[0])
    a = np.zeros((m, n))
    for k in range(K):
        uk = u[:, k].reshape(m, 1)
        vk = v[k].reshape(1, n)
        a += sigma[k] * np.dot(uk, vk)  # 前 k 个奇异值的加和
    a = a.clip(0, 255)
    return np.rint(a).astype('uint8')


def svd_background(img, alph, k):
    img_grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    u, sigma, v = np.linalg.svd(img_grey)
    mask = alph * restore(sigma, u, v, k) / 255
    for i in range(3):
        img[:, :, i] = np.rint(mask + img[:, :, i])
    return img, mask


def WarppingFunction(input1, input2, flownet):
    input_template = torch.stack([input1, input2], 2).cuda()
    warpping_tempate, diff_img = flownet(input_template)
    return warpping_tempate, diff_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true', help='Run siamrpnpp_base_1 in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)
    parser.add_argument('--model_path', type=str,
                        default='/media/ubuntu/hd_330/users/hzh/pysot-master/FlowWrapping/pretrained_model/FlowNet2-C_checkpoint.pth.tar')
    parser.add_argument('--data_dir', type=str, default='/media/ubuntu/hd_330/users/hzh/pysot-master/FlowWrapping/data')
    args = parser.parse_args()

    flownet = FlowNet2C(args).cuda()
    # load the state_dict
    flownet.load_state_dict(torch.load(args.model_path)["state_dict"])

    test_pair1 = '0img0.ppm'
    test_pair2 = '0img1.ppm'

    pim1 = read_gen(os.path.join(args.data_dir, test_pair1)).transpose(2, 0, 1)
    pim2 = read_gen(os.path.join(args.data_dir, test_pair2)).transpose(2, 0, 1)

    pim1 = torch.from_numpy(pim1.astype(np.float32)).unsqueeze(0).cuda()
    pim2 = torch.from_numpy(pim2.astype(np.float32)).unsqueeze(0).cuda()

    warpping_img, diff_img = WarppingFunction(pim1, pim2, flownet)

    warpping_img = warpping_img.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)

    warpping_img, mask = svd_background(warpping_img, 0.01, 20)


    diff_img = diff_img.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
    diff_img = np.where(diff_img < 0, 0, diff_img)
    pim1_ = read_gen(os.path.join(args.data_dir, test_pair1)).astype(np.int)

    # warpping_mask = np.expand_dims(warpping_mask, 2).repeat(3, axis=2)
    # warpping_img += warpping_mask
    plt.figure()
    plt.imshow(np.array(mask, dtype=np.float))
    # plt.imshow(np.array(warpping_img.clip(0, 255), dtype=np.int))
    # plt.imshow(np.array(diff_img, dtype=np.int))
    # plt.imshow(np.array(pim1_new))
    plt.show()