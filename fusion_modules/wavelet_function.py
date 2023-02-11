import torch
from math import pi
import torch.nn as nn

class Laplace(nn.Module):
    def __init__(self):
        super(Laplace,self).__init__()
        self.A = 0.08*5.5  # 归一化小波函数
        ep = 0.03  # 粘滞阻尼比，[0，1)
        self.tal = 0.1  # tal时间参数
        f = 50 # 小波频率
        self.w = 2 * pi * f  # 小波圆频率
        q = torch.tensor(1 - pow(ep, 2))
        self.temp=(-ep / (torch.sqrt(q)))*self.w
    def forward(self,p):
        y = self.A * torch.exp(self.temp * (p - self.tal)) * (-torch.sin(self.w * (p - self.tal)))  # 定义laplace小波函数
        return y

class Mexh(nn.Module):
    def __init__(self):
        super(Mexh,self).__init__()
    def forward(self,p):
        y = (1.0 - 2.0 * torch.pow(p, 2.0)) * torch.exp(-torch.pow(p, 2.0) / 2.0)
        return y


class Morlet(nn.Module):
    def __init__(self):
        super(Morlet,self).__init__()
        self.C=pow(pi, 0.25)

    def forward(self,p):
        y = self.C * torch.exp(-torch.pow(p, 2.0) / 2.0) * torch.cos(2.0 * pi * p)
        return y
#
# def Morlet(p):
#     C = pow(pi, 0.25)
#     # p = 0.03 * p
#     y = C * torch.exp(-torch.pow(p, 2) / 2) * torch.cos(2 * pi * p)
#     return y
# def Laplace(p):
#     A = 0.08#归一化小波函数
#     ep = 0.03#粘滞阻尼比，[0，1)
#     tal = 0.1#tal时间参数
#     f = 50#小波频率
#     w = 2 * pi * f#小波圆频率
#     q = torch.tensor(1 - pow(ep, 2))
#     y = A * torch.exp((-ep / (torch.sqrt(q))) * (w * (p - tal))) * (-torch.sin(w * (p - tal)))#定义laplace小波函数
#     return y

# def Mexh(p):
#     # p = 0.04 * p  # 将时间转化为在[-5,5]这个区间内
#     y = (1 - 2.0*torch.pow(p, 2)) * torch.exp(-torch.pow(p, 2) / 2)
#     return y
#
# def Morlet(p):
#     C = pow(pi, 0.25)
#     # p = 0.03 * p
#     y = C * torch.exp(-torch.pow(p, 2) / 2) * torch.cos(2 * pi * p)
#     return y