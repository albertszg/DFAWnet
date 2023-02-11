from datasets.SequenceDatasets import dataset
from datasets.sequence_aug import *
from scipy.io import loadmat
import os
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import numpy
#
noise=True#是否加入1500Hz频带的无用的信号
if noise == False:
    dataname = ["4000hz.mat","2000hz.mat","1000hz.mat","500hz.mat"]
else:
    dataname = ["e4000hz.mat","e2000hz.mat","e1000hz.mat","e500hz.mat"]

signal_size = 1024
label = [0,1, 2, 3]  # The failure data is labeled 1-9



# generate Training Dataset and Testing Dataset
def get_files(root, test=False):#获取数据转化成 [样本 标签]
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    normalname:List of normal data
    dataname:List of failure data
    '''

    data = []
    lab = []
    # data = {}
    # lab = {}
    for i in tqdm(range(4)):
        path2 = os.path.join(root, dataname[i])

        data1, lab1 = data_load(path2, label=label[i])
        data += data1
        lab += lab1
    return [data, lab]


def data_load(filename, label):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    axisname:Select which channel's data,---->"signal"
    '''

    realaxis = "s"
    fl = loadmat(filename)[realaxis]
    data = []
    lab = []
    start, end = 0, signal_size
    while end <= fl.shape[0]:
        data.append(fl[start:end])
        lab.append(label)
        start += signal_size
        end += signal_size

    return data, lab


def data_transforms(dataset_type="train", normlize_type="0-1"):
    transforms = {
        'train': Compose([
            Reshape(),
            Normalize(normlize_type),
            RandomAddGaussian(),
            RandomScale(),
            RandomStretch(),
            RandomCrop(),
            Retype()

        ]),
        'val': Compose([
            Reshape(),
            Normalize(normlize_type),
            Retype()
        ])
    }
    return transforms[dataset_type]

class IM(object):
    num_classes = 4
    inputchannel = 1

    def __init__(self, data_dir,normlizetype):
        self.data_dir = data_dir
        self.normlizetype = normlizetype

    def data_preprare(self, test=False):

        list_data = get_files(self.data_dir, test)
        if test:
            test_dataset = dataset(list_data=list_data, test=True, transform=None)
            return test_dataset
        else:
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            train_dataset = dataset(list_data=train_pd, transform=data_transforms('train',self.normlizetype))
            val_dataset = dataset(list_data=val_pd, transform=data_transforms('val',self.normlizetype))
            return train_dataset, val_dataset


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    device_count = 1
    # save_dir=''
    datadir = r'D:\szg\DL-based-Intelligent-Diagnosis-Benchmark-master\imitate_signal'
    normlizetype = '0-1'
    Dataset = IM(datadir, normlizetype)
    datasets = {}
    batch_size = 64
    num_workers = 0
    datasets['train'], datasets['val'] = Dataset.data_preprare()
    print('仿真实验:')
    print('训练集样本个数：{}'.format(datasets['train'].__len__()))
    print('测试集样本个数：{}'.format(datasets['val'].__len__()))
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size,
                                                  shuffle=(True if x == 'train' else False),
                                                  num_workers=num_workers,
                                                  pin_memory=(True if device == 'cuda' else False))
                   for x in ['train', 'val']}
    print('测试集batch个数：{}'.format(len(dataloaders['val'])))
    tmp = dataloaders['val'].__iter__().__next__()
    # print(tmp.numpy().size())
    print('第一个样本batch_val：')
    print(tmp[0])
    print('#------------example-----------------------------')
    tmp = dataloaders['val'].__iter__().__next__()
    # print(tmp.numpy().size())
    print(tmp[0])
    c=dataloaders['val'].__iter__()
    print('#------------c.__next__()-----------------------------')
    print(c.__next__())
    print('#------------c.__next__()-----------------------------')
    print(c.__next__())



if __name__ == '__main__':
    main()