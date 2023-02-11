import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
from math import pi
import torch.nn.functional as F
import logging
from torchsummary import summary
import time
import utils.tools as tool
#小波融合卷积
from fusion_modules.fuse_conv_wavelet import dfw_unit,dfw_unit_2,dfw_unit_3,conv1d_no_bias
#稀疏降噪层
from dynconv.maskunit_from_pytorch import MaskUnit
# from dynconv.maskunit_from_pytorch_multi import MaskUnit
from dynconv.apply_mask_function import apply_mask

#峭度loss层
from physics_loss.kurtosis import wave_kur,wave_spec_kur

from paper_model.weighting_layer import ECAAttention


# -----------------------input size>=32---------------------------------
class DFWnet(nn.Module):
    def __init__(self, in_channel=1, out_channel=10,kernel_size=251,
                            wave_use=False,wave_choose='L',wave_channel_number=64,#第一层输出通道
                           fused_use=False,fused_channel_number=32,#小波融合
                           denoising_use=False,#降噪层
                           sparse_use=False,kurtosis_use=False,kurtosis_type='1',epoch=110,#稀疏层
                           weighting_use=False):
        super(DFWnet, self).__init__()
        if fused_use:
            realchannel=fused_channel_number
        else:
            realchannel=wave_channel_number

        if wave_use:
            if wave_choose=='L'or wave_choose=='M'or wave_choose=='m':
                self.waveconv = dfw_unit(wave_choose=wave_choose,fused_use=fused_use,
                                     out_channels=wave_channel_number,real_cout=fused_channel_number,kernel_size=kernel_size)
            elif wave_choose == 'LM' or wave_choose=='Mm' or wave_choose=='Lm':
                self.waveconv = dfw_unit_2(wave_choose=wave_choose, fused_use=fused_use,
                                         out_channels=wave_channel_number, real_cout=fused_channel_number,
                                         kernel_size=kernel_size)
            else:
                self.waveconv = dfw_unit_3(wave_choose=wave_choose, fused_use=fused_use,
                                         out_channels=wave_channel_number, real_cout=fused_channel_number,
                                         kernel_size=kernel_size)
        else:
            self.waveconv = conv1d_no_bias(out_channels=wave_channel_number, kernel_size=kernel_size,fused_use=fused_use,real_cout=fused_channel_number)

        self.sparse_use=sparse_use
        if self.sparse_use:
            self.mask_generate = MaskUnit(realchannel)

        if kurtosis_use:
            if kurtosis_type==1:
                self.temp_loss=wave_kur(epoch)
            else:
                self.temp_loss = wave_spec_kur(epoch)
        else:
            self.temp_loss= tool.zero_layer_with_extra()#回传0层

        if weighting_use:

            self.weighting_layer=ECAAttention(kernel_size=3)
        else:
            self.weighting_layer = tool.identity_layer()#直连

        self.identity=tool.identity_layer() #记录降噪
        self.identity1=tool.identity_layer() #记录加权权重
        self.conv1 = nn.Sequential(
            nn.BatchNorm1d(realchannel),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(realchannel, 16, 5),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(25)  # adaptive change the outputsize to (16,5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 25, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, out_channel)

    def forward(self, x,meta):

        x=self.waveconv(x)
        # x = self.denoising(x)

        if self.sparse_use:
            m = self.mask_generate(x, meta)
            current = torch.div(m.active_positions, m.total_positions)
            target = meta['sparsity_target']
            loss1 = torch.pow(current - target, 2)#稀疏度约束
            # logging.info(current)
            masked_x=apply_mask(x, m)
            x = x+masked_x
        else:
            loss1=0
        #X=X*2.0
        loss=self.temp_loss(x,meta)#小波谱约束
        x = self.identity(x)
        x,weight = self.weighting_layer(x)
        weight=self.identity1(weight)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x,loss,loss1

def main():
    run_model = False
    print_model_parameter = True
    print_model_size = False
    time_print=False

    if run_model == True:
        strat_time=time.time()
        input = torch.randn(64, 1, 1024).cuda()
        model = DFWnet().to('cuda')
        output = model(input)
        end_time=time.time()

    if print_model_parameter == True:
        model = DFWnet()
        parm = {}
        # for name, parameters in model.named_parameters():
        #     print(name, ":", parameters.size())
        #     parm[name] = parameters.detach().cpu().numpy()
        print(model.named_modules())
        for module in model.named_modules():
            print(module)
        #named_ 带名字的打印
        # modules 遍历所有 迭代器
        # children 只遍历第二层 迭代器
        # parameters 所有参数 迭代器
        # state_dict 返回字典
    if print_model_size == True:
        device = torch.device('cuda')
        model = DFWnet().to(device)
        summary(model, input_size=(1, 1024))

    if time_print==True:
        print('cost time: {} s'.format(end_time-strat_time))

if __name__ == '__main__':
    main()