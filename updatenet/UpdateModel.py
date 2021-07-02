import torch.nn as nn
import torch

class UpdateModel(nn.Module):
    def __init__(self, feature_in=256, feature_out=256, anchor=1):
        super(UpdateModel,self).__init__()
        self.update = nn.Sequential(
            nn.Conv2d(3*3, 96, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(96, 3, 1),
        )


    def forward(self, x, x0):
        response = self.update(x)
        output = response + x0
        return output
        # return output, response

class UpdateModel_(nn.Module):
    def __init__(self, feature_in=256, feature_out=256, anchor=1):
        super(UpdateModel,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3*3, 96, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(96)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 192, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(192)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(192, 96, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(96)
        )
        self.upsample = nn.Upsample(mode='bilinear',size=[127,127])
        self.conv4 = nn.Conv2d(192,3,1)

    def forward(self, x, x0):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        upsample_out = self.upsample(conv3_out)
        concat = torch.cat((conv1_out, upsample_out), 1)
        conv4_out = self.conv4(concat)
        response = conv4_out + x0
        return response

def load_updatenet(model,update_path):

    # update_model = torch.load(update_path, map_location={'cuda:0': 'cuda:2'})['state_dict']
    update_model = torch.load(update_path)['state_dict']
    update_model_fix = dict()
    for i in update_model.keys():
        if i.split('.')[0] == 'module':  # 多GPU模型去掉开头的'module'
            update_model_fix['.'.join(i.split('.')[1:])] = update_model[i]
        else:
            update_model_fix[i] = update_model[i]  # 单GPU模型直接赋值

    model.load_state_dict(update_model_fix)
    return model