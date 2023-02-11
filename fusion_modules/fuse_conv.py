import torch
import torch.nn as nn
import torch.nn.functional as F


class FuseConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, real_cout, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(FuseConv1d, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
        )#父类有 outchannel个卷积核 ->fuse-> real_cout 个卷积核

        cout, cin, k = self.weight.shape#outchannel*inchannel*kernelsize

        layeri_softmaxP = torch.zeros(real_cout, cout)
        self.register_buffer('layeri_softmaxP', layeri_softmaxP)

        fused_weight = torch.zeros(real_cout, cin, k)
        self.register_buffer('fused_weight', fused_weight)

        if self.bias is not None:
            fused_bias = torch.zeros(real_cout)
            self.register_buffer('fused_bias', fused_bias)

    def forward(self, inputs):
        cout, cin, k = self.weight.shape

        self.fused_weight = torch.mm(self.layeri_softmaxP, self.weight.reshape(cout, -1)).reshape(-1, cin, k)
        if self.bias is not None:
            self.fused_bias = torch.mm(self.layeri_softmaxP, self.bias.unsqueeze(1)).squeeze(1)
        else:
            self.fused_bias = self.bias

        output = F.conv1d(input=inputs, weight=self.fused_weight, bias=self.fused_bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        return output



class FuseConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, real_cout, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(FuseConv2d, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
        )#父类有 outchannel个卷积核 ->fuse-> real_cout 个卷积核

        cout, cin, k, _ = self.weight.shape

        layeri_softmaxP = torch.zeros(real_cout, cout)
        self.register_buffer('layeri_softmaxP', layeri_softmaxP)

        fused_weight = torch.zeros(real_cout, cin, k, k)
        self.register_buffer('fused_weight', fused_weight)

        if self.bias is not None:
            fused_bias = torch.zeros(real_cout)
            self.register_buffer('fused_bias', fused_bias)

    def forward(self, inputs):
        cout, cin, k, _ = self.weight.shape

        self.fused_weight = torch.mm(self.layeri_softmaxP, self.weight.reshape(cout, -1)).reshape(-1, cin, k, k)
        if self.bias is not None:
            self.fused_bias = torch.mm(self.layeri_softmaxP, self.bias.unsqueeze(1)).squeeze(1)
        else:
            self.fused_bias = self.bias

        output = F.conv2d(input=inputs, weight=self.fused_weight, bias=self.fused_bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        return output