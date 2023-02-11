#!/usr/bin/python
# -*- coding:utf-8 -*-

import argparse
import os
from datetime import datetime
from utils.logger import setlogger
import logging
from utils.train_utils import train_utils
from utils.torch_random_seed import seed_torch

args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train')

    # basic parameters
    model_name = ['backbone', 'Laplace', 'Mexhat','Morlet', 'Laplace+Mexhat', 'denosing_DRSN','best_model_conference', 'ablation_no_multiple', 'ablation_no_weighting', 'laplace_fine_tuning']
    parser = argparse.ArgumentParser(description='Train')

    ###训练的参数
    parser.add_argument('--model_name', type=str, default='L', help='DFW,L,M,m,RSN,WCNN,MK,CNN,sinc')
    parser.add_argument('--data_choose', type=str, default='HYFJ_v', help='1-HYFJ,2=XJTU,3-MFPT,HYFJ_v')#1不同工况 2.不同噪声 3.消融实验
    parser.add_argument('--HYFJ_speed', type=int, choices=[500,1000,1500,2000,3900],default=1500, help='1-HYFJ,2=XJTU,3-MFPT')
    parser.add_argument('--HYFJ_test_set_ratio', type=float, default=0.5, help='ratio of HYFJ data')
    parser.add_argument('--snr', type=float, default=-4, help='signal to noise ratio')

    parser.add_argument('--extra_info', type=bool, default=True, help='extra description in save file name')
    parser.add_argument('--info', type=str, default='seed666', help='description')

    parser.add_argument('--max_epoch', type=int, default=30, help='max number of epoch')#110for XJTU MFPT 30 fro HYFJ

    parser.add_argument('--batch_size', type=int, default=64, help='batchsize of the training process')
    parser.add_argument('--print_step', type=int, default=100, help='the interval of log training information')
    parser.add_argument('--normlizetype', type=str, choices=['0-1', '1-1', 'mean-std'], default='1-1',help='data normalization methods')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of training process')
    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')

    parser.add_argument('--diff_lr', type=bool, default=False, help='use of different learning rate for the wave layer')
    parser.add_argument('--lr', type=float, default=0.0001, help='the initial learning rate')
    parser.add_argument('--opt', type=str, default='adam', help='adam ,sgd')#XJTU/MFPT adm
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='the weight decay')
    parser.add_argument('--lr_scheduler', type=str, default='exp', help='fix,step,exp,stepLR,cos')
    parser.add_argument('--lr_scheduler_step', type=str, default='10,20', help='fix,step,exp,stepLR,cos')
    parser.add_argument('--gamma', type=float, default=0.99, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--momentum', type=float, default=0.9, help='learning rate scheduler parameter for step and exp')
    # parser.add_argument('--wave_lr', type=float, default=1, help='the wave learning rate more than normal')

    #论文可变参数
    #小波层
    parser.add_argument('--wave_use', type=bool, default=True, help='if use wavelet convolution')
    parser.add_argument('--wave_choose', type=str, choices=['L','M','m','LM','Lm','Mm','LMm'],default='LMm',
                        help='L:Laplace, M:Mexhat, m:Morlet,N:Normal')
    parser.add_argument('--wave_kernel_size', type=int, default=32,help='kernel_size of wavelet kernel')
    #小波核(通道)融合
    parser.add_argument('--fused_use', type=bool, default=True, help='if use fusion')
    parser.add_argument('--fused_by', type=str, default='kernel', help='kernel,coefficient')
    parser.add_argument('--first_layer_outchannel', type=int, default=128, help='original channel number')
    parser.add_argument('--fused_channel_number', type=int, default=64, help='after fusion,the channel left')

    #稀疏层
    parser.add_argument('--sparse_use', type=bool, default=True, help='if use sparse')
    parser.add_argument('--gumbel_temp', type=float, default=0.66, help='gumbel_temp,higher,more vanishing gradient ')#默认5 小于1利于收敛 93
    parser.add_argument('--sparsity_target', type=float, default=0.6, help='target')
    parser.add_argument('--sparsity_ratio', type=float, default=0.05, help='ratio to loss')#0.05
    parser.add_argument('--sparsity_stop_epoch_ratio', type=float, default=0.8, help='stop use gubel noise: ratio*max_epoch')
    #小波谱约束层
    parser.add_argument('--kurtosis_use', type=bool, default=True, help='if use kurtosis')
    parser.add_argument('--kurtosis_type', type=int, default=2, help='1：wavelet kurtosis，2：spectral kurtosis')
    parser.add_argument('--kurtosis_ratio', type=float, default=0.005, help='ratio to loss')
    # 数据层融合#0.9117903930131005
    parser.add_argument('--weighting_use', type=bool, default=True, help='if weighting')

    #保存及可视化
    parser.add_argument('--save_acc_loss_mat', type=bool, default=True, help='save acc loss as .mat')
    parser.add_argument('--checkpoint_dir', type=str, default='./log_hyfj', help='the directory to save the model')#保存log文件及其他东西
    parser.add_argument('--t_sne_plot', type=bool, default=False, help='plot t-sne')

    # 小波降噪
    parser.add_argument('--denoising_use', type=bool, default=False, help='if use denoising')  # not used any more
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()
    # Prepare the saving path for the model
    if args.extra_info:
        sub_dir = args.model_name + '_' + args.data_choose + '_' + args.wave_choose + '_'+ args.info+ '_'+ datetime.strftime(
            datetime.now(), '%m%d-%H%M%S')
    else:
        sub_dir = args.model_name+'_'+args.data_choose + '_'+args.wave_choose+ '_'+ datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    save_dir = os.path.join(args.checkpoint_dir, sub_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # set the logger
    setlogger(os.path.join(save_dir, 'training.log'))

    # save the args
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))
    seed_torch(666)#666 555 444
    trainer = train_utils(args, save_dir)
    trainer.setup()
    trainer.train()
    logging.info(args.wave_choose)






