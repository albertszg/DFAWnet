#!/usr/bin/python
# -*- coding:utf-8 -*-

import logging
import os
import time
import warnings
from CNN_Datasets.CWRU_data import CWRU
from CNN_Datasets.XJTU_data import XJTU
from CNN_Datasets.MFPT import MFPT
from CNN_Datasets.HYFJ_data import HYFJ
from CNN_Datasets.HYFJ_data_across_condition import HYFJ as HYFJ_v
from utils.tools import *
from utils.t_sne import T_SNE
import paper_model
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat
from scipy.spatial.distance import cdist
import math


class train_utils(object):
    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir

    def setup(self):
        """
        Initialize the datasets, model, loss and optimizer
        :return:
        """
        args = self.args
        # Consider the gpu or cpu condition
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))

        '1=CWRU, 2=XJTU，3-MFPT,4-HYFJ, 5'
        if args.data_choose == 'CWRU':
            datadir = r'D:\data\西储大学轴承数据中心网站'
            Dataset = CWRU(datadir, args.normlizetype)
            logging.info('using CWRU data')
        elif args.data_choose == 'XJTU':
            # datadir = r'D:\data\XJTU-SY_Bearing_Datasets\XJTU-SY_Bearing_Datasets'
            datadir = '/media/ubuntu/Data/hcy/XJTU-SY_Bearing_Datasets'
            Dataset = XJTU(datadir, args.normlizetype,args.snr)
            logging.info('using XJTU data')
        elif args.data_choose == 'MFPT':
            datadir = r'D:\data\MFPT数据集\Fault-Data-Sets\Fault Data Sets - Revised by Rick'
            # datadir = '/media/ubuntu/Data/hcy/Mttp/MFPT Fault Data Sets'
            # datadir='/media/ubuntu/4T-disk/Shangzuogang/data/MFPT数据集/Fault-Data-Sets/Fault Data Sets - Revised by Rick'
            Dataset = MFPT(datadir, args.normlizetype)
            logging.info('using MFPT data')
        elif args.data_choose == 'HYFJ':
            # datadir=os.path.join('D:\data\HYFJ','data_'+str(args.HYFJ_speed)+'.pkl')
            datadir = os.path.join('/media/ubuntu/Data/dataset_PHM/HYFJ', 'data_' + str(args.HYFJ_speed) + '.pkl')
            Dataset = HYFJ(datadir, args.normlizetype, args.HYFJ_test_set_ratio)
            logging.info('using HYFJ data')
        elif args.data_choose == 'HYFJ_v':
            datadir='/usr/data_disk1/szg/data/HYFJ'
            # datadir = r'D:\data\HYFJ'
            Dataset = HYFJ_v(args.HYFJ_speed, args.normlizetype, args.HYFJ_test_set_ratio)
            logging.info('using HYFJ across condition data')

        logging.info(Dataset)

        self.datasets = {}
        if args.data_choose=='HYFJ_v':
            self.datasets['train'], self.datasets['val'],self.datasets['test'] = Dataset.data_preprare()
            logging.info('训练集样本数量{}'.format(self.datasets['train'].__len__()))
            logging.info('测试集样本数量{}'.format(self.datasets['val'].__len__()))
            logging.info('验证集样本数量{}'.format(self.datasets['test'].__len__()))
            self.dataloaders_v=torch.utils.data.DataLoader(self.datasets['test'], batch_size=args.batch_size,
                                                           shuffle=(False),
                                                           num_workers=args.num_workers,
                                                           pin_memory=(True if self.device == 'cuda' else False))

        else:
            self.datasets['train'], self.datasets['val'] = Dataset.data_preprare()
            logging.info('训练集样本数量{}'.format(self.datasets['train'].__len__()))
            logging.info('测试集样本数量{}'.format(self.datasets['val'].__len__()))
        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x], batch_size=args.batch_size,
                                                           shuffle=(True if x == 'train' else False),
                                                           num_workers=args.num_workers,
                                                           pin_memory=(True if self.device == 'cuda' else False))
                            for x in ['train', 'val']}
        # Define the model
        self.model = getattr(paper_model, args.model_name)(in_channel=Dataset.inputchannel, out_channel=Dataset.num_classes,kernel_size=args.wave_kernel_size,
                           wave_use=args.wave_use,wave_choose=args.wave_choose,wave_channel_number=args.first_layer_outchannel,#小波使用
                           fused_use=args.fused_use,fused_channel_number=args.fused_channel_number,#小波融合
                           denoising_use=args.denoising_use,#降噪层
                           sparse_use=args.sparse_use,kurtosis_use=args.kurtosis_use,kurtosis_type=args.kurtosis_type,epoch=args.max_epoch,#稀疏层
                           weighting_use=args.weighting_use)#加权层
        logging.info('模型分类数{}'.format(Dataset.num_classes))
        if self.device_count > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Define the optimizer
        if args.opt == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr,
                                       momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.opt == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr,
                                        weight_decay=args.weight_decay)
        else:
            raise Exception("optimizer not implement")

        self.lr_scheduler=lr_scheduler_choose(lr_scheduler_way=args.lr_scheduler, optimizer=self.optimizer, steps=args.lr_scheduler_step, gamma=args.gamma)
        # Load the checkpoint
        self.start_epoch = 0

        # Invert the model and define the loss
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()


    def train(self):
        """
        Training process
        :return:
        """
        time_start = time.time()
        args = self.args

        step = 0
        best_acc = 0.0
        batch_count = 0
        batch_loss = 0.0
        batch_acc = 0
        step_start = time.time()
        print_grad=False
        epoch_best = 0
        # =================精度/迭代次数图==============
        train_Accuracy_list = []
        val_Accuracy_list = []

        train_loss_s_list = []
        val_loss_s_list = []

        train_loss_k_list = []
        val_loss_k_list = []

        train_loss_list = []
        val_loss_list = []
        wave_shape_list = []#存第一层卷积核形状
        layer_score_list = []
        fuesd_wave_shape_before_list =[]
        fuesd_wave_shape_after_list = []

        # =================saveparameters==============
        if args.model_name=='DFW':
            if args.wave_use == True:
                with torch.set_grad_enabled(False):
                    # 取出第一层的小波形状
                    self.model.eval()
                    meta = {'gumbel_temp': args.gumbel_temp, 'gumbel_noise': False, 'masks': [], 'epoch': 0,
                            'sparsity_target': args.sparsity_target, 'device': self.device}
                    tmp = self.dataloaders['val'].__iter__().__next__()
                    _ = self.model(tmp[0].cuda(),meta)
                    #这里网络并没有经过数据，所以第一层的权重还是跟着nn.module初始化的，必须经过一次运行才可以
                    wave_shape_list.append(self.model.waveconv.filters.detach().cpu().numpy())
                    logging.info('取出第一层的小波形状')
                #取出第一层的输出量
                if args.sparse_use:
                    conv_out = LayerActivations(self.model.waveconv, Sequential=False,layer_num=0)
                    conv_out_deonising=LayerActivations(self.model.identity, Sequential=False,layer_num=0)
                    conv_out_weight=LayerActivations(self.model.identity1, Sequential=False,layer_num=0)
                    logging.info('第一层注入钩子获取输出')
        layers_iScore = []

        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
            # Update the learning rate
            if self.lr_scheduler is not None:
                # self.lr_scheduler.step(epoch)
                logging.info('current lr: {}'.format(self.lr_scheduler.get_last_lr()))
            else:
                logging.info('current lr: {}'.format(args.lr))

            # Each epoch has a training and val phase
            for phase in ['train', 'val']:
                # Define the temp variable
                epoch_start = time.time()
                epoch_acc = 0
                epoch_loss = 0.0
                epoch_loss_s=0.0
                epoch_loss_k = 0.0

                # Set model to train mode or test mode
                if phase == 'train':
                    '''小波层融合计算指标'''
                    if args.fused_use and args.model_name=='DFW':
                        logging.info(f"\nComputing epoch {epoch}'s all fused layer...")
                        start = time.time()

                        if args.fused_by == 'kernel':
                            module = self.model.waveconv
                            param = module.filters  # weight是原始的根据这个计算，那就是原始的小波核[64,1,64]
                            # * Compute layeri_param
                            layeri_param = torch.reshape(param.detach(), (param.shape[0],-1))  # * layeri_param.shape=[cout, cin, kernel_size], layeri_param[j] means filterj's weight.

                            '''直接用小波核计算'''
                            # * Compute layeri_Eudist
                            layeri_Eudist = cdist(layeri_param.cpu(), layeri_param.cpu(), metric='euclidean').astype(
                                np.float32)

                            # * Compute layeri_negaEudist
                            layeri_negaEudist = -torch.from_numpy(layeri_Eudist).to(self.device)

                            # * Compute layeri_softmaxP
                            softmax = nn.Softmax(dim=1)

                            # * Compute t
                            Ts = 1
                            Te = 10
                            e = epoch
                            E = args.max_epoch
                            pi = math.pi

                            k = 1
                            A = 2 * (Te - Ts) * (1 + math.exp(-k * E)) / (1 - math.exp(-k * E))
                            T = A / (1 + math.exp(-k * e)) + Ts - A / 2
                            t = 1 / T

                            layeri_softmaxP = softmax(
                                layeri_negaEudist / t)  # * layeri_softmaxP.shape=[cout, cout], layeri_softmaxP[j] means filterj's softmax vector P.计算概率

                            # * Compute layeri_KL
                            layeri_KL = torch.mean(layeri_softmaxP[:, None, :] * (
                                    layeri_softmaxP[:, None, :] / (layeri_softmaxP + 1e-7)).log(),
                                                   dim=2)  # * layeri_KL.shape=[cout, cout], layeri_KL[j, k] means KL divergence between filterj and filterk

                            # * Compute layers_iScore
                            layeri_iScore_kl = torch.sum(layeri_KL, dim=1)#layeri_KL(128,128)
                            if epoch == self.start_epoch:
                                if epoch == 0:
                                    layers_iScore.append(layeri_iScore_kl)
                                else:
                                    raise Exception('非断点')
                                    layers_iScore = layers_iScore_resume  # 断点续传
                            else:
                                if False not in torch.isfinite(layeri_iScore_kl):  # 如果非无穷则更新layer_iScore
                                    layers_iScore[
                                        0] = layeri_iScore_kl  # * layers_iScore.shape=[cout], layers_iScore[j] means jth layer‘s importance score
                                else:
                                    pass

                            # * setup conv_module.layeri_topm_filters_id
                            _, topm_ids = torch.topk(layers_iScore[0],
                                                     args.fused_channel_number)  # Returns the k largest elements of the given input tensor along a given dimension  layers_m：每一层实际通道数
                            # If dim is not given, the last dimension of the input is chosen.取出
                            _, topm_ids_order = torch.topk(layers_iScore[0], args.fused_channel_number,
                                                           sorted=False)  # 按原始顺序取出，不按排序后的，其实topk sorted没用，都是排序后的id

                            # 记录每一次更新权重值
                            layer_score_list.append(topm_ids.cpu().numpy())
                            # * Compute layeri_p
                            softmaxP = layeri_softmaxP[topm_ids_order, :]  # 根据top k重要的通道的序号去取出他们的系数，
                            onehotP = torch.eye(param.shape[0]).to(self.device)[topm_ids_order,
                                      :]  # Returns a 2-D tensor with ones on the diagonal and zeros elsewhere.对角阵

                            # * setup conv_module.layeri_softmaxP
                            module.layeri_softmaxP = softmaxP  # 更新融合小波层中的
                            # module.layeri_softmaxP = onehotP
                            del param, layeri_param, layeri_negaEudist, layeri_KL

                            if epoch==0:
                                self.model.eval()
                                meta = {'gumbel_temp': args.gumbel_temp, 'gumbel_noise': False, 'masks': [], 'epoch': 0,
                                        'sparsity_target': args.sparsity_target, 'device': self.device}
                                tmp = self.dataloaders['val'].__iter__().__next__()
                                _ = self.model(tmp[0].cuda(), meta)
                                fuesd_wave_shape_before_list.append(self.model.waveconv.fused_weight.detach().cpu().numpy())

                            if epoch==args.max_epoch-1:
                                fuesd_wave_shape_after_list.append(self.model.waveconv.fused_weight.detach().cpu().numpy())
                        else:
                            raise NotImplementedError("fused by coefficient: not implemented!")

                    self.model.train()
                    gumbel_noise = False if epoch > args.sparsity_stop_epoch_ratio * args.max_epoch else True
                else:
                    gumbel_noise = False
                    self.model.eval()
                for batch_idx, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # Do the learning process, in val, we do not care about the gradient for relaxing
                    with torch.set_grad_enabled(phase == 'train'):
                        meta = {'gumbel_temp': args.gumbel_temp, 'gumbel_noise': gumbel_noise, 'masks': [], 'epoch': epoch, 'sparsity_target': args.sparsity_target,'device':self.device}
                        # forward
                        if args.model_name=='DFW':
                            logits,data_for_sparse,sparsity = self.model(inputs,meta)
                        else:
                            logits= self.model(inputs)
                        '''稀疏加峭度'''
                        loss_base=self.criterion(logits, labels)

                        if args.sparse_use and args.model_name=='DFW':
                            # loss_s=10*sparsity
                            loss_s =args.sparsity_ratio* sparsity
                            epoch_loss_s_temp = loss_s.item() * inputs.size(0)
                        else:
                            loss_s=0
                            epoch_loss_s_temp = 0

                        if args.kurtosis_use and args.model_name=='DFW':
                            regularization_loss = data_for_sparse
                            loss_k = args.kurtosis_ratio * regularization_loss
                            epoch_loss_k_temp = loss_k.item() * inputs.size(0)
                        else:
                            loss_k = 0
                            epoch_loss_k_temp = 0
                        # loss =loss_k+loss_s
                        loss = loss_base+loss_s+loss_k
                        # loss = loss_base
                        # loss = loss_s
                        pred = logits.argmax(dim=1)
                        correct = torch.eq(pred, labels).float().sum().item()
                        loss_temp = loss.item() * inputs.size(0)
                        # loss_temp = loss.item()
                        epoch_loss += loss_temp
                        epoch_acc += correct

                        epoch_loss_k += epoch_loss_k_temp
                        epoch_loss_s += epoch_loss_s_temp

                        # Calculate the training information
                        if phase == 'train':
                            # backward
                            self.optimizer.zero_grad()
                            loss.backward()
                            """调试的时候看看梯度是否反传回去了"""
                            if print_grad == True:
                                a = self.model.parameters().__next__().grad
                                logging.info('Epoch: {} [{}/{}] grad of j:{} '.format(epoch, batch_idx * len(inputs),len(dataloaders[phase].dataset),a))
                            self.optimizer.step()

                            batch_loss += loss_temp
                            batch_acc += correct
                            batch_count += inputs.size(0)

                            # Print the training information
                            if step % args.print_step == 0:
                                batch_loss = batch_loss / batch_count
                                batch_acc = batch_acc / batch_count
                                temp_time = time.time()
                                train_time = temp_time - step_start
                                step_start = temp_time
                                batch_time = train_time / args.print_step if step != 0 else train_time
                                sample_per_sec = 1.0*batch_count/train_time
                                logging.info('Epoch: {} [{}/{}], Train Loss: {:.4f} Train Acc: {:.4f},'
                                             '{:.1f} examples/sec {:.2f} sec/batch'.format(
                                    epoch, batch_idx*len(inputs), len(self.dataloaders[phase].dataset),
                                    batch_loss, batch_acc, sample_per_sec, batch_time
                                ))
                                batch_acc = 0
                                batch_loss = 0.0
                                batch_count = 0
                            step += 1


                # Print the train and val information via each epoch
                epoch_loss = epoch_loss / len(self.dataloaders[phase].dataset)
                epoch_acc = epoch_acc / len(self.dataloaders[phase].dataset)
                epoch_loss_k = epoch_loss_k / len(self.dataloaders[phase].dataset)
                epoch_loss_s = epoch_loss_s / len(self.dataloaders[phase].dataset)
                if phase == 'train':
                    train_Accuracy_list.append(100 * epoch_acc)
                    train_loss_list.append(epoch_loss)
                    train_loss_k_list.append(epoch_loss_k)
                    train_loss_s_list.append(epoch_loss_s)
                    if epoch==args.max_epoch-1:#一个epoch里面的最后一个batch提取
                        if args.model_name == 'DFW':
                            mask=meta['masks']
                            hard_mask_train=mask#[19,1,995]
                            hard_input_train=inputs.cpu().numpy()#[19,1,1024]
                            hard_intermediate_train=conv_out.features.detach().cpu().numpy() #获取小波卷积层的输出     #[19,64,995]
                            #小波去噪层输出
                            hard_denosing_train=conv_out_deonising.features.detach().cpu().numpy()
                            #小波加权层输出
                            hard_weight_train=conv_out_weight.features.detach().cpu().numpy()
                            #网络分类层输出
                            output_train=logits.detach().cpu().numpy()
                            #实际标签
                            label_train=labels.cpu().numpy()
                        k=1

                else:
                    val_Accuracy_list.append(100 * epoch_acc)
                    val_loss_list.append(epoch_loss)
                    val_loss_k_list.append(epoch_loss_k)
                    val_loss_s_list.append(epoch_loss_s)
                    if epoch==args.max_epoch-1:
                        if args.model_name == 'DFW' and args.sparse_use:
                            mask=meta['masks']
                            hard_mask_val=mask
                            hard_input_val = inputs.cpu().numpy()
                            hard_intermediate_val =conv_out.features.detach().cpu().numpy()
                            # 小波去噪层输出
                            hard_denosing_val = conv_out_deonising.features.detach().cpu().numpy()
                            # 小波加权层输出
                            hard_weight_val = conv_out_weight.features.detach().cpu().numpy()
                            # 网络分类层输出
                            output_val = logits.detach().cpu().numpy()

                            label_val = labels.cpu().numpy()

                            conv_out.remove()
                            conv_out_deonising.remove()
                            conv_out_weight.remove()
                            logging.info('钩子去除')
                        k = 1

                logging.info('Epoch: {} {}-Loss: {:.4f} {}-Acc: {:.4f}, Cost {:.4f} sec, sparsity loss {:.4f}, kurtosis loss {:.4f}'.format(
                    epoch, phase, epoch_loss, phase, epoch_acc, time.time()-epoch_start,epoch_loss_s,loss_k
                ))

                # save the model
                if phase == 'val':
                    # save the checkpoint for other learning
                    # model_state_dic = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()
                    # save the best model according to the val accuracy
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        epoch_best = epoch
                        logging.info("best model epoch {}, acc {:.4f}".format(epoch, epoch_acc))
                        model_state_dic = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()
                        #torch.save(model_state_dic,os.path.join(self.save_dir, '{}-{:.4f}-best_model.pth'.format(epoch, best_acc)))

                    if epoch > args.max_epoch - 2 :
                        last_acc = epoch_acc
                        logging.info("======================================")
                        logging.info(" ")
                        logging.info("Last epoch model epoch {}, acc {:.4f}".format(epoch, last_acc))
                        logging.info("best model epoch {}, acc {:.4f}".format(epoch_best, best_acc))

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        logging.info("best model  epoch: {} acc: {:.4f}".format(epoch_best, best_acc))
        logging.info('train_over')
        if args.data_choose=='HYFJ_v':
            logging.info('val on another unseened domain')
            epoch_acc=0
            self.model.load_state_dict(model_state_dic)
            self.model.eval()
            if args.t_sne_plot:
                tsne_out = LayerActivations(self.model.conv2, Sequential=True, layer_num=-1)
                logging.info('特征提取层注入钩子获取tsne输出')
            with torch.no_grad():
                for batch_idx, (inputs, labels) in enumerate(self.dataloaders_v):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    if args.model_name == 'DFW':
                        logits, _, _ = self.model(inputs, meta)
                    else:
                        logits = self.model(inputs)
                    pred = logits.argmax(dim=1)
                    correct = torch.eq(pred, labels).float().sum().item()
                    epoch_acc += correct
                    if args.t_sne_plot:
                        tsne_temp = tsne_out.features.detach().view(tsne_out.features.size()[0], -1)
                        label_temp = labels.detach()
                        if batch_idx==0:
                            tsne_v=tsne_temp
                            label_v=label_temp
                        else:
                            tsne_v = torch.cat([tsne_v,tsne_temp],dim=0)
                            label_v = torch.cat([label_v,label_temp])
            epoch_acc = epoch_acc / len(self.dataloaders_v.dataset)
            logging.info("best model  epoch: {} acc: {:.4f}".format(epoch_best, best_acc))
            logging.info('val on another 4 conditions, the total acc:{:.4f}'.format(epoch_acc))
            if args.t_sne_plot:
                tsne_out.remove()
                logging.info('tsne钩子去除,画TSNE图')
                T_SNE(tsne_v.cpu().numpy(),label_v.cpu().numpy())

        # =================saveparameters==============
        if args.wave_use == True and args.model_name=='DFW':
            with torch.set_grad_enabled(False):
                # 取出第一层的小波形状
                self.model.eval()
                wave_shape_list.append(self.model.waveconv.filters.detach().cpu().numpy())
                logging.info('训练后取出第一层的小波形状')

        max_epoch=args.max_epoch
        #============画折线图=======
        time_end=time.time()
        time_spend=time_end-time_start
        logging.info('train_time:{}'.format(time_spend))
        x = range(self.start_epoch, args.max_epoch)
        y1 = train_Accuracy_list
        y3=val_Accuracy_list
        plt.subplot(211)
        plt.title('accuracy/Loss vs. epoches' + '_' + args.data_choose)
        plt.plot(x, y1,color='black',label='train_Accuracy')
        plt.plot(x, y3,color='red',label='val_Accuracy')
        plt.legend()
        plt.ylabel('accuracy')

        y2 = train_loss_list
        y4 = val_loss_list
        plt.subplot(212)
        plt.plot(x, y2,color='black',label='train_loss')
        plt.plot(x, y4,color='red',label='val_loss')
        plt.ylabel('Loss')
        plt.xlabel('epochs')
        plt.legend()
        plt.show()
        #plot sparsity
        x = range(self.start_epoch, args.max_epoch)
        y5 = train_loss_s_list
        y6 = val_loss_s_list
        plt.title('sparsity_Loss vs. epoches' + '_' + args.data_choose)
        plt.plot(x, y5, color='black', label='train_loss')
        plt.plot(x, y6, color='red', label='val_loss')
        plt.ylabel('Loss')
        plt.xlabel('epochs')
        plt.legend()
        plt.show()

        # plot kurtosis
        x = range(self.start_epoch, args.max_epoch)
        y7 = train_loss_k_list
        y8 = val_loss_k_list
        plt.title('kurtosis_Loss vs. epoches' + '_' + args.data_choose)
        plt.plot(x, y7, color='black', label='train_loss')
        plt.plot(x, y8, color='red', label='val_loss')
        plt.ylabel('Loss')
        plt.xlabel('epochs')
        plt.legend()
        plt.show()

        #===========acc_loss=======
        train_acc = np.array(train_Accuracy_list)
        val_acc = np.array(val_Accuracy_list)
        train_loss = np.array(train_loss_list)
        val_loss = np.array(val_loss_list)
        train_loss_s= np.array(train_loss_s_list)
        val_loss_s= np.array(val_loss_s_list)
        layer_score = np.array(layer_score_list)
        fuesd_wave_before_shape= np.squeeze(np.array(fuesd_wave_shape_before_list))
        fuesd_wave_after_shape = np.squeeze(np.array(fuesd_wave_shape_after_list))

        acc_last10 = np.mean(val_acc[max_epoch-10:max_epoch]/100)
        std_last10 = np.std(val_acc[max_epoch-10:max_epoch]/100)
        logging.info("last 10  epoch acc_mean : {} acc_std: {:.4f}".format(acc_last10,std_last10))

        if args.data_choose=='XJTU' or args.data_choose=='HYFJ_v':
            mask_train = np.array(hard_mask_train)  # [19,1,995]
            mask_val = np.array(hard_mask_val)
        else:
            mask_train = 0
            mask_val = 0

        hard_denosing_val = conv_out_deonising.features.detach().cpu().numpy()
        # 小波加权层输出
        hard_weight_val = conv_out_weight.features.detach().cpu().numpy()
        # 网络分类层输出
        output_val = logits.cpu().numpy()

        if args.save_acc_loss_mat == True:
            filename = os.path.join(self.save_dir, 'loss.mat')
            if args.wave_use and args.model_name=='DFW':
                savemat(filename,
                        {'before': wave_shape_list[0] if args.wave_use else 0,
                         'after': wave_shape_list[1] if args.wave_use else 0, 'train_acc': train_acc,
                         'val_acc': val_acc, 'train_loss': train_loss, 'val_loss': val_loss,
                         'train_loss_s': train_loss_s, 'val_loss_s': val_loss_s,
                         'layer_score': layer_score if args.fused_use else 0,
                         'fused_layer_before': fuesd_wave_before_shape if args.fused_use else 0,
                         'fused_layer_after': fuesd_wave_after_shape,
                         'time_spend': time_spend, 'last_10_acc_mean': acc_last10, 'last_10_acc_std': std_last10,

                         'input_train': hard_input_train,
                         'wave_train': hard_intermediate_train,
                         'denoising_train': hard_denosing_train,
                         'weight_train':hard_weight_train,
                         'output_train':output_train,
                         'label_train':label_train,

                         'mask_train': mask_train if args.sparse_use else 0,

                         'input_val': hard_input_val,
                         'wave_val': hard_intermediate_val,
                         'denoising_val':hard_denosing_val,
                         'weight_val':hard_weight_val ,
                         'output_val':output_val,
                         'label_val': label_val,

                         'mask_val': mask_val
                         })  # 专利出图
            else:
                savemat(filename,
                        { 'train_acc': train_acc,
                         'val_acc': val_acc, 'train_loss': train_loss, 'val_loss': val_loss,
                          'train_loss_s': train_loss_s, ' val_loss_s':  val_loss_s,
                         'time_spend': time_spend, 'last_10_acc_mean': acc_last10, 'last_10_acc_std': std_last10})







