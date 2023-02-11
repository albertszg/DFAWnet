import torch
import torch.nn as nn
import torch.nn.functional as F
from fusion_modules.wavelet_function import Laplace,Mexh,Morlet
import logging
import matplotlib.pyplot as plt
'''
real_cout 实际输出的channel数量
Parameter:( nn.Parameter(torch.FloatTensor(64, 10).normal_(mean=0, std=0.01)) ) 
buffer:     self.register_buffer('mybuffer', mybuffer_tmp)

optimizer 优化 parameter 不更新 buffer

都在 model.state_dict(), 随model.to(device)而移动
buffer 需要得到梯度信息同时，不会因为optimizer而更新，也可以通过 parameter+requred_grad=False 实现
'''
#dfw_unit(wave_use,wave_choose,fused_use,fused_channel_number)
class dfw_unit(nn.Module):
    def __init__(self, wave_choose,fused_use,out_channels, real_cout, kernel_size, in_channels=1):
        super(dfw_unit, self).__init__()
        if in_channels != 1:
            msg = "MexhConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size - 1
        self.wave_function=[]
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1#使得kenel为偶数个，并且使向下取偶数

        #所有的小波核使用的kernel_size一样长，所以时间长度应该是一致的，Morlet和Mexhat的time_step一致，只是尺度不同
        #尺度在卷积通道范围上展开
        if wave_choose=='L':#Laplace
            logging.info('using Laplace as dfw_unit kernel')
            self.wave_function=Laplace()
            self.time_step = torch.linspace(0, 1, steps=int((self.kernel_size)))
            self.a_ = nn.Parameter(torch.linspace(0.1, 2, out_channels), requires_grad=True)  # 尺度scale-频率  1到10，outchannel个等距点，划分到outchannel个频带去

        elif wave_choose=='M':#Mexhat
            logging.info('using Mexhat as dfw_unit kernel')
            self.wave_function=Mexh()
            time_disc_left,time_disc_right = torch.linspace(-(self.kernel_size / 2) + 1, -1,steps=int((self.kernel_size / 2))),torch.linspace(0, (self.kernel_size / 2) - 1,steps=int((self.kernel_size / 2)))
            self.time_step = torch.cat([time_disc_left, time_disc_right], dim=0)  # 40x1x250
            self.a_ = nn.Parameter(torch.linspace(0.1,3, out_channels), requires_grad=True) #0.1-2

        elif wave_choose=='m':#Morlet
            logging.info('using Morlet as dfw_unit kernel')
            self.wave_function =Morlet()
            time_disc_left, time_disc_right = torch.linspace(-(self.kernel_size / 2) + 1, -1,steps=int((self.kernel_size / 2))), torch.linspace(0, (self.kernel_size / 2) - 1, steps=int((self.kernel_size / 2)))
            self.time_step = torch.cat([time_disc_left, time_disc_right], dim=0)  # 40x1x250
            self.a_ = nn.Parameter(torch.linspace(0.1, 4.5, steps=int(out_channels)), requires_grad=True)
        else:
            msg = "only one wavelet are use in dfw_unit,more wavelet please use dfw_unit_2,dfw_unit_3"
            raise ValueError(msg)

        #小波核融合
        self.fused_use=fused_use
        layeri_softmaxP = torch.zeros(real_cout, out_channels)#[32,64]
        self.register_buffer('layeri_softmaxP', layeri_softmaxP)
        fused_weight = torch.zeros(real_cout, 1, self.kernel_size)#[32,1,32]
        self.register_buffer('fused_weight', fused_weight)
        self.filters =torch.randn(self.out_channels, 1, self.kernel_size)  # 形式改变，原本是kernel*outchannel个值 [64,64]


    def forward(self, waveforms):


        p1 = self.time_step.cuda()/self.a_.cuda().view(-1,1)#kernel*outchannel个值

        wave_filter = self.wave_function(p1)#获得laplace小波族

        self.filters = (wave_filter).view(self.out_channels, 1, self.kernel_size).cuda()#形式改变，原本是kernel*outchannel个值 [64,64]

        if self.fused_use:
            self.fused_weight = torch.mm(self.layeri_softmaxP, self.filters.reshape(self.out_channels, -1)).reshape(-1,1, self.kernel_size)# [32,64]*[64,64]
            # print(self.fused_weight.size())
        else:
            self.fused_weight = self.filters
            # print(self.fused_weight.size())
        # ####################################################画小波图
        # for i in range(self.out_channels):
        #     x = range(0, int((self.kernel_size)))
        #     y=self.filters[i,0,:].detach().cpu()
        #     plt.plot(x, y, color='black', label='wavelet')
        #     plt.show()
        #
        # exit(0)
        # # ##############################################
        return F.conv1d(waveforms, self.fused_weight, stride=1, padding=1, dilation=1, bias=None, groups=1)

        #可视化
        # ####################################################画小波图
        # for i in range(self.out_channels):
        #     x = range(0, int((self.kernel_size)))
        #     y=self.filters[i,0,:].detach().cpu()
        #     plt.plot(x, y, color='black', label='wavelet')
        #     plt.show()
        # ##############################################

class dfw_unit_2(nn.Module):
    def __init__(self, wave_choose,fused_use,out_channels, real_cout, kernel_size, in_channels=1):
        super(dfw_unit_2, self).__init__()
        if in_channels != 1:
            msg = "MexhConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size - 1

        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1##使得kenel为偶数个，并且使向下取偶数

        #所有的小波核使用的kernel_size一样长，所以时间长度应该是一致的，其中Morlet和Mexhat的time_step一致，只是尺度不同
        #尺度在卷积通道范围上展开,一个小波占一半的卷积通道
        self.wave_function = []
        self.time_step=[]
        time_step_L = torch.linspace(0, 1, steps=int((self.kernel_size)))
        time_disc_left, time_disc_right = torch.linspace(-(self.kernel_size / 2) + 1, -1,steps=int((self.kernel_size / 2))), torch.linspace(0, (self.kernel_size / 2) - 1, steps=int((self.kernel_size / 2)))
        time_step_Mm = torch.cat([time_disc_left, time_disc_right], dim=0)  # 40x1x250

        #laplace: 0.1 , 2
        #Mexhat: 0.1 , 3
        #Morlet: 0.1, 4.5
        if wave_choose=='LM':
            logging.info('using Laplace&Mexhat as dfw_unit kernel')
            self.wave_function = [Laplace(), Mexh()]#小波函数
            self.time_step=[time_step_L,time_step_Mm]#小波时间
            scale_step_M = int(out_channels/2)
            logging.info('Mexhat channel number:{}'.format(scale_step_M))
            # logging.info('morlet channel number:{}'.format(scale_step_m))
            scale_L = nn.Parameter(torch.linspace(0.1, 2, steps=out_channels-scale_step_M),requires_grad=True)
            scale_M = nn.Parameter(torch.linspace(0.1,3, steps=scale_step_M),requires_grad=True)
            self.a_ = [scale_L, scale_M]#小波尺度

        elif wave_choose=='Lm':
            logging.info('using Laplace&Morlet as dfw_unit kernel')
            self.wave_function = [Laplace(), Morlet()]
            self.time_step = [time_step_L, time_step_Mm]
            scale_step_m=int(out_channels/2)
            logging.info('morlet channel number:{}'.format(scale_step_m))
            scale_L = nn.Parameter(torch.linspace(0.1, 2, steps=out_channels-scale_step_m), requires_grad=True)
            scale_m = nn.Parameter(torch.linspace(0.1, 4.5, steps= scale_step_m), requires_grad=True)
            self.a_ = [scale_L, scale_m]  # 小波尺度
        elif wave_choose == 'Mm':
            logging.info('using Mexhat&Morlet as dfw_unit kernel')
            self.wave_function = [Mexh(), Morlet()]
            self.time_step = [time_step_Mm, time_step_Mm]
            scale_step_m = int(5*out_channels / 10.0)
            logging.info('morlet channel number:{}'.format(scale_step_m))
            scale_M = nn.Parameter(torch.linspace(0.1,3, steps=out_channels-scale_step_m), requires_grad=True)
            scale_m = nn.Parameter(torch.linspace(0.1,4.5, steps=scale_step_m), requires_grad=True)
            self.a_ = [scale_M, scale_m]  # 小波尺度
        else:
            msg = "only 2 wavelets are use in dfw_unit_2"
            raise ValueError(msg)

        #小波核融合
        self.fused_use=fused_use
        layeri_softmaxP = torch.zeros(real_cout, out_channels)#[32,64]
        self.register_buffer('layeri_softmaxP', layeri_softmaxP)
        fused_weight = torch.zeros(real_cout, 1, self.kernel_size)#[32,1,32]
        self.register_buffer('fused_weight', fused_weight)
        self.filters =torch.randn(self.out_channels, 1, self.kernel_size)  # 形式改变，原本是kernel*outchannel个值 [64,64]


    def forward(self, waveforms):


        p1 = self.time_step[0].cuda()/self.a_[0].cuda().view(-1,1)#kernel*outchannel个值
        wave_filter_1 = self.wave_function[0](p1)#获得小波族1

        p2 = self.time_step[1].cuda() / self.a_[1].cuda().view(-1,1)  # kernel*outchannel个值
        wave_filter_2 = self.wave_function[1](p2)  # 获得小波族2

        wave_filter=torch.cat((wave_filter_1,wave_filter_2),0)#拼接

        self.filters = (wave_filter).view(self.out_channels, 1, self.kernel_size).cuda()#形式改变，原本是kernel*outchannel个值 [64,64]
        # print(self.filters.size())

        #组装卷积核


        if self.fused_use:
            self.fused_weight = torch.mm(self.layeri_softmaxP, self.filters.reshape(self.out_channels, -1)).reshape(-1,1, self.kernel_size)# [32,64]*[64,64]
        else:
            self.fused_weight = self.filters
        return F.conv1d(waveforms, self.fused_weight, stride=1, padding=1, dilation=1, bias=None, groups=1)


class dfw_unit_3(nn.Module):
    def __init__(self, wave_choose,fused_use,out_channels, real_cout, kernel_size, in_channels=1):
        super(dfw_unit_3, self).__init__()
        if in_channels != 1:
            msg = "MexhConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size - 1
        self.wave_function=[]
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1  # 使得kenel为偶数个，并且使向下取偶数

        channels_M = int(out_channels / 3)
        channels_m = int(out_channels / 3)
        channels_L = out_channels - channels_M -channels_m

        self.wave_function = []
        self.time_step = []
        time_step_L = torch.linspace(0, 1, steps=int((self.kernel_size)))
        time_disc_left, time_disc_right = torch.linspace(-(self.kernel_size / 2) + 1, -1,steps=int((self.kernel_size / 2))), torch.linspace(0, (self.kernel_size / 2) - 1, steps=int((self.kernel_size / 2)))
        time_step_Mm = torch.cat([time_disc_left, time_disc_right], dim=0)  # 40x1x250

        if wave_choose == 'LMm':
            logging.info('using Laplace & Mexhat & Morlet as dfw_unit kernel')
            self.wave_function = [Laplace(), Mexh(), Morlet()]  # 小波函数
            self.time_step = [time_step_L, time_step_Mm, time_step_Mm]  # 小波时间
            self.scale_L = nn.Parameter(torch.linspace(0.1, 2, steps=channels_L),
                                        requires_grad=True)
            self.scale_M = nn.Parameter(torch.linspace(0.1, 3, steps=channels_M ),
                                   requires_grad=True)
            self.scale_m = nn.Parameter(torch.linspace(0.1, 4.5, steps=channels_m),
                                   requires_grad=True)
            # self.a_ = [scale_L, scale_M, scale_m]  # 小波尺度

        else:
            msg = "only 3 wavelets are use in dfw_unit_3, other wavelets are not implemented!"
            raise ValueError(msg)
        logging.info('channels: Laplace: {}, Mexhat {}, Morlet {}'.format(channels_L,channels_M,channels_m))
        # 小波核融合
        self.fused_use = fused_use
        layeri_softmaxP = torch.zeros(real_cout, out_channels)  # [32,64]
        self.register_buffer('layeri_softmaxP', layeri_softmaxP)
        fused_weight = torch.zeros(real_cout, 1, self.kernel_size)  # [32,1,32]
        self.register_buffer('fused_weight', fused_weight)
        self.filters = torch.randn(self.out_channels, 1, self.kernel_size)  # 形式改变，原本是kernel*outchannel个值 [64,64]

    def forward(self, waveforms):

        p1 = self.time_step[0].cuda() / self.scale_L.cuda().view(-1,1)# kernel*outchannel个值
        wave_filter_1 = self.wave_function[0](p1)  # 获得小波族1

        p2 = self.time_step[1].cuda() / self.scale_M.cuda().view(-1,1) # kernel*outchannel个值
        wave_filter_2 = self.wave_function[1](p2)  # 获得小波族2

        p3 = self.time_step[2].cuda() / self.scale_m.cuda().view(-1,1) # kernel*outchannel个值
        wave_filter_3 = self.wave_function[2](p3)  # 获得小波族3

        wave_filter = torch.cat((wave_filter_1, wave_filter_2,wave_filter_3), 0)  # 拼接

        self.filters = (wave_filter).view(self.out_channels, 1,
                                          self.kernel_size).cuda()  # 形式改变，原本是kernel*outchannel个值 [64,64]
        # print(self.filters.size())

        # 组装卷积核

        if self.fused_use:
            self.fused_weight = torch.mm(self.layeri_softmaxP, self.filters.reshape(self.out_channels, -1)).reshape(-1,
                                                                                                                    1,
                                                                                                                    self.kernel_size)  # [32,64]*[64,64]
            # print(self.fused_weight.size())
        else:
            self.fused_weight = self.filters
            # print(self.fused_weight.size())
        return F.conv1d(waveforms, self.fused_weight, stride=1, padding=1, dilation=1, bias=None, groups=1)


class conv1d_no_bias(nn.Module):
    def __init__(self, out_channels, kernel_size,fused_use=False,real_cout=32, in_channels=1,stride=1, padding=0, dilation=1, bias=False, groups=1):
        super(conv1d_no_bias, self).__init__()
        if in_channels != 1:
            msg = "conv1d_no_bias only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)
        self.out_channels = out_channels
        self.kernel_size = kernel_size - 1
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1  # 使得kenel为偶数个，并且使向下取偶数
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        if bias:
            raise ValueError('conv1d_no_bias does not support bias.')
        if groups > 1:
            raise ValueError('conv1d_no_bias does not support groups.')

        self.weight_ori = torch.Tensor(self.out_channels,self.kernel_size)
        torch.nn.init.kaiming_uniform_(self.weight_ori, mode='fan_in')
        self.weight = nn.Parameter(self.weight_ori,requires_grad = True)

        # 卷积核融合
        self.fused_use = fused_use
        layeri_softmaxP = torch.zeros(real_cout, out_channels)  # [32,64]
        self.register_buffer('layeri_softmaxP', layeri_softmaxP)
        fused_weight = torch.zeros(real_cout, 1, self.kernel_size)  # [32,1,32]
        self.register_buffer('fused_weight', fused_weight)
        self.filters = torch.randn(self.out_channels, 1, self.kernel_size)  # 形式改变，原本是kernel*outchannel个值 [64,64]

    def forward(self, waveforms):
        # print('filter is not ok')
        self.filters =self.weight.view(self.out_channels, 1, self.kernel_size)#形式改变，原本是kernel*outchannel个值
        # print('filter is ok')

        if self.fused_use:
            self.fused_weight = torch.mm(self.layeri_softmaxP, self.filters.reshape(self.out_channels, -1)).reshape(-1,1, self.kernel_size)# [32,64]*[64,64]
            # print(self.fused_weight.size())
        else:
            self.fused_weight = self.filters

        return F.conv1d(waveforms, self.fused_weight, stride=1, padding=1, dilation=1, bias=None, groups=1)
