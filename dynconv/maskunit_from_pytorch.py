import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as v
import sys
# from utils import logger_dyconv as logger
import logging
import dynconv.apply_mask_function as apply_mask_function
class Mask():
    '''
    Class that holds the mask properties
    x:b*c*l
    hard: the hard/binary mask (1 or 0), 3-dim tensor
    soft (optional): the float mask, same shape as hard
    active_positions: the amount of positions where hard == 1
    total_positions: the total amount of positions 
                        (typically batch_size * output_width * output_height)
    '''
    def __init__(self, hard, soft=None):
        assert hard.dim() == 3
        assert hard.shape[1] == 1#one mask for all channel
        assert soft is None or soft.shape == hard.shape

        self.hard = hard
        # self.active_positions = torch.sum(torch.sum(hard,dim=2))
        self.active_positions = torch.sum(hard)# this must be kept backpropagatable!
        self.total_positions = hard.numel()#int类型
        self.soft = soft
        
        self.flops_per_position = 0
    
    def size(self):
        return self.hard.shape

    def __repr__(self):
        return f'Mask with {self.active_positions}/{self.total_positions} positions, and {self.flops_per_position} accumulated FLOPS per position'

class MaskUnit(nn.Module):
    '''
    Generates the mask and applies the gumbel softmax trick
    '''

    def __init__(self, channels, stride=1, dilate_stride=1):
        super(MaskUnit, self).__init__()
        self.maskconv = Squeeze(channels=channels, stride=stride)
        self.gumbel = Gumbel()
        # self.gumbel = F.gumbel_softmax()
        self.expandmask = ExpandMask(stride=dilate_stride)

    def forward(self, x, meta):
        soft = self.maskconv(x)
        hard = self.gumbel(soft, meta['gumbel_temp'], meta['gumbel_noise'])
        mask = Mask(hard, soft[:,0,:].unsqueeze(dim=1))
        # print(mask.__repr__())
        # print(mask.hard)
        # print(mask.hard.size())
        # hard_dilate = self.expandmask(mask.hard)
        # mask_dilate = Mask(hard_dilate)
        # m = {'std': mask, 'dilate': mask_dilate}


        # m = {'std': mask}
        # meta['masks'].appedn(m)
        meta['masks'].append(hard.detach().cpu().numpy())
        return mask


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature=1,gumbel_noise=True):

    logits=torch.softmax(logits,dim=-2)
    logits=logits.log()
    if gumbel_noise:
        y = logits + sample_gumbel(logits.size())
    else:
        y = logits
    return F.softmax(y / temperature, dim=-2)


class Gumbel(nn.Module):
    '''
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    '''
    def __init__(self, eps=1e-8):
        super(Gumbel, self).__init__()

    def forward(self,logits, temperature=1, gumbel_noise=True):
        if not self.training:  # no Gumbel noise during inference
            # print(self.training)
            a=torch.argmax(logits,dim=1)
            activation=(1.0-a).unsqueeze(dim=1)
            b=(logits >= 0).float()[:,0,:].unsqueeze(dim=1)
            return activation
        y = gumbel_softmax_sample(logits, temperature,gumbel_noise)

        shape = y.size()
        _, ind = y.max(dim=-2)
        y_hard = torch.zeros_like(y)
        y_hard.scatter_(1, ind.unsqueeze(dim=1), 1)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y_hard = (y_hard - y).detach() + y
        assert not torch.any(torch.isnan(y_hard))
        hard_mask=y_hard[:,0,:]
        return hard_mask.unsqueeze(dim=1)

## Mask convs
class Squeeze(nn.Module):
    """ 
    Squeeze module to predict masks
    output:
    mask.size()=x.size()
    """

    def __init__(self, channels, stride=1):
        super(Squeeze, self).__init__()
        self.norm=nn.BatchNorm1d(64)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels, 2, bias=True)
        self.conv = nn.Conv1d(channels, out_channels=2,kernel_size=3, stride=stride,padding=1, bias=True)#all channels sharing the same mask

    def forward(self, x):
        x = self.norm(x)
        b, c, _ = x.size()#batch*channel*length
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b,2,1)

        z = self.conv(x)
        return z + y.expand_as(z)#全局与局部结合


class ExpandMask(nn.Module):#用于inference的时候使用，暂时不用，他把
    def __init__(self, stride, padding=1): 
        super(ExpandMask, self).__init__()
        self.stride=stride
        self.padding = padding
        
    def forward(self, x):#输入 hard mask  输出
        # assert x.shape[1] == 1

        if self.stride > 1:
            self.pad_kernel = torch.zeros( (1,1,self.stride), device=x.device)
            self.pad_kernel[0,0,0] = 1
        self.dilate_kernel = torch.ones((1,1,1+2*self.padding), device=x.device)

        x = x.float()
        if self.stride > 1:
            x = F.conv_transpose1d(x, self.pad_kernel, stride=self.stride, groups=x.size(1))
        x = F.conv1d(x, self.dilate_kernel, padding=self.padding, stride=1)
        return x > 0.5

def main():

    meta={'gumbel_temp':1,'gumbel_noise':True,'masks':[]}

    mask_generate=MaskUnit(2)
    a = v(torch.full((2, 2, 10), 1.0), requires_grad=True)
    a.retain_grad()
    a_9 = a * 3.0
    m=mask_generate(a_9,meta)
    mask_dilate, mask = m['dilate'], m['std']
    out= apply_mask_function.apply_mask(a_9, mask)
    out_1=torch.sum(out)

    print(out_1.size())
    out_1.backward()
    print(a.grad)
    # print(mask_generate.__repr__)
    # print(mask_generate.size)


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
    #

if __name__ == '__main__':
    main()