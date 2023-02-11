import torch
import torch.nn as nn
from torch.autograd import Variable as v
from utils.Hilbert_transform import HilbertTransform
import math


class wave_kur(nn.Module):
    """
    compute the wavelet coefficient kurtosis
    """
    def __init__(self,epoch):
        super(wave_kur, self).__init__()
        self.num_epochs=epoch
    def forward(self, x, meta) :
        a_2=torch.pow(x,2)
        a_4 = torch.sum(torch.pow(a_2, 2),dim=2) / a_2.size()[2]
        a_3 = torch.pow(torch.sum(a_2,dim=2) / a_2.size()[2], 2)
        wave_kur=torch.div(a_4,a_3)
        wave_kur=torch.sum(wave_kur-3)/(x.size()[0]*x.size()[1])

        p = meta['epoch'] / (0.33 * self.num_epochs)
        progress = math.cos(min(max(p, 0), 1) * (math.pi / 2)) ** 2
        wave_kur = progress*wave_kur
        return -wave_kur

class wave_spec_kur(nn.Module):
    """
    compute the wavelet coefficient kurtosis
    """
    def __init__(self,epoch):
        super(wave_spec_kur, self).__init__()
        self.hilbert = HilbertTransform()
        self.num_epochs = epoch
    def forward(self, x, meta) :
        x_hilbert=self.hilbert(x)
        a_2=torch.pow(x_hilbert.real,2)+torch.pow(x_hilbert.imag,2)#hilbert 变换
        a_4 = torch.sum(torch.pow(a_2, 2),dim=2) / a_2.size()[2]
        a_3 = torch.pow(torch.sum(a_2,dim=2) / a_2.size()[2], 2)
        wave_kur=a_4/a_3
        wave_kur=torch.sum(wave_kur)/(x.size()[0]*x.size()[1])

        p = meta['epoch'] / (0.33 * self.num_epochs)
        progress = math.cos(min(max(p, 0), 1) * (math.pi / 2)) ** 2
        wave_kur = wave_kur*progress
        return -wave_kur
#峭度
# a = v(torch.full((2,2,3),5.0),requires_grad=True)
# a.retain_grad()
# a_9 = a*3.0
# a_9.retain_grad()
# # b = v(torch.randn(12,2,512),requires_grad=True).view(-1, 1)
#
# a_2=torch.pow(a_9,2)
#
# a_4=torch.sum(torch.pow(a_2,2))/a_2.size()[2]
# a_3=torch.pow(torch.sum(a_2)/a_2.size()[2],2)
#
# kur=a_4/a_3*10000000.0
# print(kur)
# kur.backward(retain_graph=True)
# print(a.grad)
# print(a_9.grad)
