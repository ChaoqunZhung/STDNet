import torch
from torch import  nn
import numpy as np
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

def fill_up_weights(up):
    w = up.weight.data
    f = np.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - np.fabs(i / f - c)) * (1 - np.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class CenternetHead(nn.Module):
    def __init__(self,heads,in_channel,head_conv=128,final_kernel=1):
        super(CenternetHead,self).__init__()
        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0: 
                fc = nn.Sequential(
                    nn.Conv2d(in_channel, head_conv,
                              kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes,
                              kernel_size=final_kernel, stride=1,
                              padding=final_kernel // 2, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(in_channel, classes,
                               kernel_size=final_kernel, stride=1,
                               padding=final_kernel // 2, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward(self,x):
        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(x)
            
        return [z]