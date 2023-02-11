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
signal_size = 12000
datasetname = ["12k Drive End Bearing Fault Data", "12k Fan End Bearing Fault Data", "48k Drive End Bearing Fault Data",
               "Normal Baseline Data"]
normalname = ["97.mat", "98.mat", "99.mat", "100.mat"]
# For 12k Drive End Bearing Fault Data
dataname1 = ["105.mat", "118.mat", "130.mat", "169.mat", "185.mat", "197.mat", "209.mat", "222.mat",
             "234.mat"]  # 1797rpm 内圈/滚动体/外圈顺序 * 故障尺寸0.007，0.014 0.021
dataname2 = ["106.mat", "119.mat", "131.mat", "170.mat", "186.mat", "198.mat", "210.mat", "223.mat",
             "235.mat"]  # 1772rpm
dataname3 = ["107.mat", "120.mat", "132.mat", "171.mat", "187.mat", "199.mat", "211.mat", "224.mat",
             "236.mat"]  # 1750rpm
dataname4 = ["108.mat", "121.mat", "133.mat", "172.mat", "188.mat", "200.mat", "212.mat", "225.mat",
             "237.mat"]  # 1730rpm

label = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # The failure data is labeled 1-9
axis = ["_DE_time", "_FE_time", "_BA_time"]


# generate Training Dataset and Testing Dataset
def get_files(root, test=False):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    normalname:List of normal data
    dataname:List of failure data
    '''
    data_root1 = os.path.join(root, datasetname[3])#正常数据
    data_root2 = os.path.join(root, datasetname[0])#故障数据

    path1 = os.path.join(data_root1, normalname[0])  # 0->1797rpm ;1->1772rpm;2->1750rpm;3->1730rpm
    data, lab = data_load(path1, axisname=normalname[0],label=0)  # nThe label for normal data is 0

    for i in tqdm(range(9)):
        path2 = os.path.join(data_root2, dataname1[i])

        data1, lab1 = data_load(path2, dataname1[i], label=label[i])
        data += data1
        lab += lab1
    return [data, lab]


def data_load(filename, axisname, label):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    axisname:Select which channel's data,---->"_DE_time","_FE_time","_BA_time"
    '''
    datanumber = axisname.split(".")
    if eval(datanumber[0]) < 100:
        realaxis = "X0" + datanumber[0] + axis[0]
    else:
        realaxis = "X" + datanumber[0] + axis[0]
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

class CWRU(object):
    num_classes = 10
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
    datadir = r'D:\文档\毕设\CWRU_classification\西储大学轴承数据中心网站'
    normlizetype = '0-1'
    Dataset = CWRU(datadir, normlizetype)
    datasets = {}
    batch_size = 64
    num_workers = 0
    datasets['train'], datasets['val'] = Dataset.data_preprare()
    print(datasets['train'].__len__())
    print(datasets['val'].__len__())
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size,
                                                  shuffle=(True if x == 'train' else False),
                                                  num_workers=num_workers,
                                                  pin_memory=(True if device == 'cuda' else False))
                   for x in ['train', 'val']}
    print(len(dataloaders['train']))
    tmp = dataloaders['val'].__iter__().__next__()
    # print(tmp.numpy().size())
    print(tmp[0])
    print('#------------example-----------------------------')
    tmp = dataloaders['val'].__iter__().__next__()
    # print(tmp.numpy().size())
    print(tmp[0])
    c=dataloaders['val'].__iter__()
    print('#------------c.__next__()-----------------------------')
    print(c.__next__())




if __name__ == '__main__':
    main()