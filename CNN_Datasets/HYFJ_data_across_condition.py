
from sklearn.model_selection import train_test_split

import pandas as pd
from datasets.SequenceDatasets import dataset
from datasets.sequence_aug import *
import pickle
import torch
import os
import matplotlib as plt
length = 1024




# data_BT500 = pd.read_csv('./HYFJ/BT_500.txt', names=['500'])
# data_NF500 = pd.read_csv('./HYFJ/NF_500.txt', names=['500'])
# data_TSW500 = pd.read_csv('./HYFJ/TSW_500.txt', names=['500'])
# data_XDBH500 = pd.read_csv('./HYFJ/XDBH_500.txt', names=['500'])
# data_BT1000 = pd.read_csv('./HYFJ/BT_1000.txt', names=['1000'])
# data_NF1000 = pd.read_csv('./HYFJ/NF_1000.txt', names=['1000'])
# data_TSW1000 = pd.read_csv('./HYFJ/TSW_1000.txt', names=['1000'])
# data_XDBH1000 = pd.read_csv('./HYFJ/XDBH_1000.txt', names=['1000'])
# data_BT1500 = pd.read_csv('./HYFJ/BT_1500.txt', names=['1500'])
# data_NF1500 = pd.read_csv('./HYFJ/NF_1500.txt', names=['1500'])
# data_TSW1500 = pd.read_csv('./HYFJ/TSW_1500.txt', names=['1500'])
# data_XDBH1500 = pd.read_csv('./HYFJ/XDBH_1500.txt', names=['1500'])
# data_BT2000 = pd.read_csv('./HYFJ/BT_2000.txt', names=['2000'])
# data_NF2000 = pd.read_csv('./HYFJ/NF_2000.txt', names=['2000'])
# data_TSW2000 = pd.read_csv('./HYFJ/TSW_2000.txt', names=['2000'])
# data_XDBH2000 = pd.read_csv('./HYFJ/XDBH_2000.txt', names=['2000'])
# data_BT3900 = pd.read_csv('./HYFJ/BT_3900.txt', names=['3900'])
# data_NF3900 = pd.read_csv('./HYFJ/NF_3900.txt', names=['3900'])
# data_TSW3900 = pd.read_csv('./HYFJ/TSW_3900.txt', names=['3900'])
# data_XDBH3900 = pd.read_csv('./HYFJ/XDBH_3900.txt', names=['3900'])
#
#
# data1, label1 = cut_the_data(data_BT500, 2)
# data2, label2 = cut_the_data(data_NF500, 0)
# data3, label3 = cut_the_data(data_TSW500, 1)
# data4, label4 = cut_the_data(data_XDBH500, 3)
#
# # data1, label1 = cut_the_data(data_BT3900, 2)
# # data2, label2 = cut_the_data(data_NF3900, 0)
# # data3, label3 = cut_the_data(data_TSW3900, 1)
# # data4, label4 = cut_the_data(data_XDBH3900, 3)
#
#
# data_pd = pd.DataFrame({'data':data1, 'label':label1})
# data_pd = data_pd.append(pd.DataFrame({'data':data2, 'label':label2}))
# data_pd = data_pd.append(pd.DataFrame({'data':data3, 'label':label3}))
# data_pd = data_pd.append(pd.DataFrame({'data':data4, 'label':label4}))


# with open('./data_500.pkl', 'wb') as f:
#     pickle.dump(data_pd, f)

def cut_the_data(data, label):
    data = data.values.tolist()
    data = np.array(data)
    start, end = 0, length
    d_data = []
    d_label = []
    while end < len(data):
        d_data.append(data[start:end])
        d_label.append(label)
        start += length
        end += length
    return d_data, d_label

# 数据预处理
def data_transforms(dataset_type="train", normlize_type="-1-1"):
    transforms = {
        'train': Compose([
            Reshape(),
            Normalize(normlize_type),
            wgn(snr=2),
            Retype()

        ]),
        'val': Compose([
            Reshape(),
            Normalize(normlize_type),
            Retype()
        ])
    }
    return transforms[dataset_type]


def train_test_split_order(data_pd, test_size=0.2, num_classes=4):
    train_pd = pd.DataFrame(columns=('data', 'label'))
    val_pd = pd.DataFrame(columns=('data', 'label'))
    for i in range(num_classes):
        data_pd_tmp = data_pd[data_pd['label'] == i].reset_index(drop=True)#drop=True 丢弃原索引
        train_pd = train_pd.append(data_pd_tmp.loc[:int((1-test_size)*data_pd_tmp.shape[0]), ['data', 'label']], ignore_index=True)
        val_pd = val_pd.append(data_pd_tmp.loc[int((1-test_size)*data_pd_tmp.shape[0]):, ['data', 'label']], ignore_index=True)
    return train_pd,val_pd

def test_data(test_path,num_classes=4):
    train_pd = pd.DataFrame(columns=('data', 'label'))
    for i in test_path:
        with open(i, 'rb') as f:
            list_data = pickle.load(f)
            # data_pd_tmp = list_data.reset_index(drop=True)
        # pd.concat([train_pd,list_data],axis=0,ignore_index=True)

        train_pd=train_pd.append(list_data.loc[:,['data', 'label']],ignore_index=True)
    return train_pd

class HYFJ(object):
    num_classes = 4
    inputchannel = 1

    def __init__(self, speed , normlizetype,HYFJ_test_set_ratio):
        Speeds=[500, 1000, 1500, 2000, 3900]
        self.path = os.path.join('/usr/data_disk1/szg/data/HYFJ', 'data_' + str(speed) + '.pkl')
        self.test_path=[]
        Speeds.remove(speed)
        for i in Speeds:
            self.test_path.append(os.path.join('/usr/data_disk1/szg/data/HYFJ', 'data_' + str(i) + '.pkl'))
        self.normlizetype = normlizetype
        self.test_size = HYFJ_test_set_ratio

    def data_preprare(self, test=False):

        with open(self.path, 'rb') as f:
            list_data = pickle.load(f)
        if test:
            test_dataset = dataset(list_data=list_data, test=True, transform=None)
            return test_dataset
        else:
            train_pd, val_pd = train_test_split_order(list_data, test_size=self.test_size, num_classes=4)
            test_pd= test_data(self.test_path,num_classes=4)
            train_dataset = dataset(list_data=train_pd, transform=data_transforms('train',self.normlizetype))
            val_dataset = dataset(list_data=val_pd, transform=data_transforms('val',self.normlizetype))
            test_dataset= dataset(list_data=test_pd, transform=data_transforms('val',self.normlizetype))
            return train_dataset, val_dataset, test_dataset



# data = HYFJ(r'D:\szg\Data\HYFJ\data_1000.pkl', 'mean-std')
# # print(max(train_dataset[600][0][0]), len(val_dataset))
# # plt.plot(train_dataset[600][0][0])
# # plt.show()
# datasets = {}
# datasets['train'], datasets['val'] = data.data_prepare()
# print(len(datasets['val']),len(datasets['train']))
# dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=64,
#                                                            shuffle=(True if x == 'train' else False),
#                                                            num_workers=0)
#                for x in ['train', 'val']}
# a = next(iter(dataloaders['train']))
# print(a[0].shape)

def main():
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)
    # device_count = 1
    # # save_dir=''
    # datadir = r'D:\szg\Data\HYFJ\data_3900.pkl'
    # # datadir=r'D:\文档\毕设\XJTU-SY_Bearing_Datasets'
    # normlizetype = '0-1'
    # Dataset = HYFJ(datadir, normlizetype)
    # datasets = {}
    # batch_size = 64
    # num_workers = 0
    # datasets['train'], datasets['val'] = Dataset.data_preprare()
    # print('HYFJ数据集：')
    # print('训练集样本个数：{}'.format(datasets['train'].__len__()))
    # print('测试集样本个数：{}'.format(datasets['val'].__len__()))
    # dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size,
    #                                               shuffle=(True if x == 'train' else False),
    #                                               num_workers=num_workers,
    #                                               pin_memory=(True if device == 'cuda' else False))
    #                for x in ['train', 'val']}
    # print('测试集batch个数：{}'.format(len(dataloaders['val'])))
    # tmp = dataloaders['val'].__iter__().__next__()
    # # print(tmp.numpy().size())
    # print('第一个样本batch_val：')
    # print(tmp[0])
    # print('#------------example-----------------------------')
    # tmp = dataloaders['val'].__iter__().__next__()
    # # print(tmp.numpy().size())
    # print(tmp[0])
    # c=dataloaders['val'].__iter__()
    # print('#------------c.__next__()-----------------------------')
    # print(c.__next__())
    # print('#------------c.__next__()-----------------------------')
    # print(c.__next__())
    Speed = [500, 1000, 1500, 2000, 3900]
    # datadir = os.path.join('/media/ubuntu/Data/dataset_PHM/HYFJ', 'data_' + str(speed[0]) + '.pkl')
    print(Speed)
    Speed.remove(500)
    print(len(Speed))



if __name__ == '__main__':
    main()