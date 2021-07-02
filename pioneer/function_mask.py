import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def function_mask(alpha, mask):
    # 注意输入mask的type

    mask_alpha = torch.tensor(alpha).cuda()
    mask_one = torch.tensor(1.).cuda()
    mask_update = torch.where(mask==torch.tensor(0.), mask_alpha, mask_one)
    return mask_update

def histeq(im,nbr_bins = 256):
    """对一幅灰度图像进行直方图均衡化"""
    #计算图像的直方图
    #在numpy中，也提供了一个计算直方图的函数histogram(),第一个返回的是直方图的统计量，第二个为每个bins的中间值
    imhist,bins = np.histogram(im.flatten(),nbr_bins,normed= True)
    cdf = imhist.cumsum()   #
    cdf = 255.0 * cdf / cdf[-1]
    #使用累积分布函数的线性插值，计算新的像素值
    im2 = np.interp(im.flatten(),bins[:-1],cdf)
    return im2.reshape(im.shape)

def function_flow(lamda, flow):
    flow_lamda = torch.tensor(lamda).cuda()
    #flow_histeq = histeq(flow)
    flow= flow/torch.max(flow)
    flow_update = flow * flow_lamda
    return flow_update[:127, :127]


if __name__ == '__main__':
    # test function_mask
    mask = torch.zeros([7, 7])
    mask[2:5, 2:5] = torch.tensor(1.)
    alpha = 0.8

    mask_update = function_mask(alpha, mask)
    print(mask_update)

    # test function_flow
    # im = np.array(Image.open('/media/HardDisk_new/wh/code/pysot-master/pysot_20201123/flow.jpg').convert('L'))
    # im= histeq(im)
    im = torch.randint(0,255,[128,128,3])
    lamda = 0.8

    result = function_flow(lamda, im)
