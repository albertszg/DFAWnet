from torch import optim
import torch.nn as nn
import torch
import torch.nn.functional as F

class conv1d_no_bias(nn.Module):
    def __init__(self, out_channels, kernel_size, in_channels=1,stride=1, padding=0, dilation=1, bias=False, groups=1):
        super(conv1d_no_bias, self).__init__()
        if in_channels != 1:
            msg = "conv1d_no_bias only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)
        self.out_channels = out_channels
        self.kernel_size = kernel_size - 1
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

    def forward(self, waveforms):
        # print('filter is not ok')
        self.filters =self.weight.view(self.out_channels, 1, self.kernel_size)#形式改变，原本是kernel*outchannel个值
        # print('filter is ok')
        return F.conv1d(waveforms, self.filters, stride=1, padding=1, dilation=1, bias=None, groups=1)

class zero_layer_with_extra(nn.Module):
    def __init__(self):
        super(zero_layer_with_extra, self).__init__()
    def forward(self, x,meta):
        return 0

class identity_layer(nn.Module):
    def __init__(self):
        super(identity_layer, self).__init__()
    def forward(self,x):
        return x

class zero_layer(nn.Module):
    def __init__(self):
        super(zero_layer, self).__init__()
    def forward(self,x):
        return 0

def lr_scheduler_choose(lr_scheduler_way='fix', optimizer=None, steps='2', gamma=0.1):
    if lr_scheduler_way == 'step':
        steps = [int(step) for step in steps.split(',')]
        # print(steps)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, steps, gamma=gamma)
    elif lr_scheduler_way == 'exp':
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma)
        # lr = base_lr * gamma^epoch
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    elif lr_scheduler_way == 'stepLR':
        steps = int(steps)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, steps, gamma)#lr*gamma
        # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    elif lr_scheduler_way == 'cos':
        steps = int(steps)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, 0)
        #lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100,eta_min=0.001)
    elif lr_scheduler_way == 'fix':
        lr_scheduler = None
    else:
        raise Exception("lr schedule not implement")
    return lr_scheduler

class LayerActivations:#对某一层（layer_num)层插入钩子，记录其输出。
    features = None
    def __init__(self, model,Sequential=False,layer_num=None):
        if Sequential:
            self.hook = model[layer_num].register_forward_hook(self.hook_fn)
        else:
            self.hook = model.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = output
    def remove(self):
        self.hook.remove()


#============画折线图=======
# y2 = train_loss_list
# y4 = val_loss_list
# plt.subplot(212)
# plt.plot(x, y2,color='black',label='train_loss')
# plt.plot(x, y4,color='red',label='val_loss')
# plt.ylabel('Loss')
# plt.xlabel('epochs')
# plt.legend()
# plt.show()


# #============画混淆矩阵图=======

# y_pred=[]
# y_true=[]

#y_pred+=pred.cpu().numpy().tolist()

# sns.set()
# f,ax=plt.subplots()
# C2= confusion_matrix(y_true, y_pred, labels=[0,1, 2,3,4,5,6,7,8,9])
# print(C2) #打印出来看看
# sns.heatmap(C2,annot=True,ax=ax,cmap='Blues') #画热力图
#
# ax.set_title('confusion matrix') #标题
# ax.set_xlabel('predict') #x轴
# ax.set_ylabel('true') #y轴
# plt.show()

##============查看梯度========
# a = model.parameters().__next__().grad

