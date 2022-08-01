# SiamDMU

This is the official implementation with training code for 'SiamDMU: Siamese Dual Mask Update Networkfor Visual Object Tracking'. 

Introduction
--------------------------------
We propose a novel tracker named Siamese Dual Mask Update (SiamDMU), which utilizes motion and semantic information to generate the enhanced tracking results for updating the template.</br>

[The Results of SiamDMU are here.](https://pan.baidu.com/s/1Lx_YF8nSzommRv8igQZuqw) Baidu:2333</br>

Requirements
--------------------------
1. Ubuntu 20.04
2. Pytorch 1.3.1
3. Python 3.7

Installation
--------------------------
Please refer to [PySOT_INSTALL.md](https://github.com/STVIR/pysot/blob/master/INSTALL.md), [FlowNet_README.md](https://github.com/NVIDIA/flownet2-pytorch#readme) and [DeepMask_README.md](https://github.com/foolwood/deepmask-pytorch#readme) for installation.

You can also download the code of SiamDMU from [BaiduYun](https://pan.baidu.com/s/15jwMPICD2UZ7S6CARpvGDw) password: ofed. This file has included the model of SiamRPN++, FlowNet and DeepMask in the dirctory.</br>

Usage
--------------------------
### Download models
1. Please download the SiamRPN++ and SiamRPN++_otb [model.pth](https://github.com/STVIR/pysot/blob/master/MODEL_ZOO.md) to the path './experiments/siamrpn_r50_l234_dwxcorr/' and './experiments/siamrpn_r50_l234_dwxcorr_otb/', respectively. 
2. Download the [FlowNet2-C_checkpoint.pth.tar](https://drive.google.com/file/d/1BFT6b7KgKJC8rA59RmOVAXRM_S7aSfKE/view) to the './FlowWrapping/pretrained_model/'. 
3. Download the [DeepMask.pth.tar](http://www.robots.ox.ac.uk/~qwang/DeepMask.pth.tar) to the './deepmask/pretrained/'.
### Test
1. Modify the dataset path 'dataset_root'.
2. run the './tools/test_SiamDMU_VOT.py'.
### Train 
1. run the './updatenet/create_template.py'.
3. run the './updatenet/train_upd.py'.


Acknowledgments
------------------------------
1. [PySOT](https://github.com/STVIR/pysot)
2. [deepmask-pytorch](https://github.com/foolwood/deepmask-pytorch)
3. [flownet2-pytorch](https://github.com/NVIDIA/flownet2-pytorch)
4. [SiamTrackers](https://github.com/HonglinChu/SiamTrackers)


