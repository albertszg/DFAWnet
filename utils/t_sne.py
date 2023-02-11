# -*- coding:utf-8 -*-
#=================分类可视化==============
# following function (plot_with_labels) is for visualization, can be ignored if not interested
from matplotlib import cm
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
try: from sklearn.manifold import TSNE; HAS_SK = True
except: HAS_SK = False; print('Please install sklearn for layer visualization')
import seaborn as sns


def plot_with_labels_3d(lowDWeights, labels):
    plt.cla()# Clear axis即清除当前图形中的当前活动轴。其他轴不受影响
    X, Y ,Z= lowDWeights[:, 0], lowDWeights[:, 1], lowDWeights[:, 2]

    colors = [[107 / 255.0, 203 / 255.0, 119 / 255.0], [255 / 255.0, 107 / 255.0, 107 / 255.0]]
    # colors = [[77/255.0, 150/255.0, 255/255.0],[255/255.0, 107/255.0, 107/255.0],[107/255.0, 203/255.0, 119/255.0],[255 / 255.0,184 / 255.0, 48 / 255.0]]
    cmap, norm = matplotlib.colors.from_levels_and_colors(np.arange(0, labels.max() + 2), colors)
    # fig, ax = plt.subplots()
    # fig = plt.figure(dpi=600, figsize=(8, 8))
    # ax = fig.subplot()
    # ax = fig.add_subplot()

    # fig,ax=plt.subplots()
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    scatter = ax.scatter3D(X,Y,Z,marker='.',c=labels,cmap=cmap,norm=norm)
    legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
    ax.add_artist(legend1)
    #修改lengend标签
    # a, b = scatter.legend_elements()
    # print(b)
    # b = ['$\\mathdefault {type A}$',
    #      '$\\mathdefault{type B}$',
    #      '$\\mathdefault{type C}$',
    #      '$\\mathdefault{type D}$']
    # legend1 = ax.legend(a, b, title="Classes")
    # plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max());plt.zlim(Y.min(), Y.max())
    plt.title('hidden feature layer visualized by T-SNE')
    plt.show()
    plt.pause(0.01)
def plot_with_labels(lowDWeights, labels):
    # plt.cla()# Clear axis即清除当前图形中的当前活动轴。其他轴不受影响
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]

    colors = [[107 / 255.0, 203 / 255.0, 119 / 255.0], [255 / 255.0, 107 / 255.0, 107 / 255.0]]
    # colors = [[77/255.0, 150/255.0, 255/255.0],[255/255.0, 107/255.0, 107/255.0],[107/255.0, 203/255.0, 119/255.0],[255 / 255.0,184 / 255.0, 48 / 255.0]]
    cmap, norm = matplotlib.colors.from_levels_and_colors(np.arange(0, labels.max() + 2), colors)
    # fig, ax = plt.subplots()
    # fig = plt.figure(dpi=600, figsize=(8, 8))
    # ax = fig.subplot()
    # ax = fig.add_subplot()

    fig,ax=plt.subplots()
    scatter = ax.scatter(X,Y,marker='.',c=labels,cmap=cmap,norm=norm)
    legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
    ax.add_artist(legend1)
    #修改lengend标签
    # a, b = scatter.legend_elements()
    # print(b)
    # b = ['$\\mathdefault {type A}$',
    #      '$\\mathdefault{type B}$',
    #      '$\\mathdefault{type C}$',
    #      '$\\mathdefault{type D}$']
    # legend1 = ax.legend(a, b, title="Classes")
    plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max())
    plt.title('hidden feature layer visualized by T-SNE')
    plt.show()
    plt.pause(0.01)

#=================PCA==============
# plt.ion()# 打开交互模式在plt.show()之前一定不要忘了加plt.ioff()，如果不加，界面会一闪而过，并不会停留。
# Visualization of trained flatten layer (T-SNE)
    #
#
    #n_components默认为2，嵌入空间的维度（嵌入空间的意思就是结果空间）
    #default:1000, 最大迭代次数
    #init= string or numpy array, default:”random”, 可以是’random’, ‘pca’或者一个numpy数组(shape=(n_samples, n_components)。
'''
perplexity(困惑度)默认为30，数据集越大，需要参数值越大，建议值位5-50,
    低困惑度对应的是局部视角，要把自己想象成一只蚂蚁，在数据所在的流形上一个点一个点地探索。
    高困惑度对应的是全局视角，要把自己想象成上帝。
n_components默认为2，嵌入空间的维度（嵌入空间的意思就是结果空间）
default:1000, 最大迭代次数
init= string or numpy array, default:”random”, 可以是’random’, ‘pca’或者一个numpy数组(shape=(n_samples, n_components)。
    全局结构不能很清楚的保留。这个问题可以通过先用PCA降维到一个合理的维度（如50）后再用t-SNE来缓解，前置的PCA步骤也可以起到去除噪声等功能。（sklearn中可以直接使用参数init='pca'）
'''
def T_SNE(data,label,plot=True,dim=2):
    # data = np.asarray(data, dtype='float64')
    if plot:
        if dim==1:
            kdeplot(label, data)
            return
    tsne = TSNE(perplexity=50, n_components=dim, init='pca', learning_rate='auto', n_iter=2000,n_iter_without_progress=300)
    low_dim_embs = tsne.fit_transform(data)  # [:plot_only, :]降维成[261,2]
    if plot:
        if dim==3:
            plot_with_labels_3d(low_dim_embs, label)
        elif dim==2:
            plot_with_labels(low_dim_embs, label)
        else:
            raise NotImplementedError

def kdeplot(labels,scores,title='hidden feature Distribution (1D)',savedir=None):
    # sns.set()#切换到seaborn的默认运行配置
    index_normal = np.where(labels == 0)
    index_abnormal = np.where(labels == 1)
    # sns.distplot(scores[index_normal],kde=True,hist=False,kde_kws={"shade":True},color='g')
    # sns.distplot(scores[index_abnormal],kde=True,hist=False,kde_kws={"shade":True},color='r')

    sns.kdeplot(scores[index_normal],fill=True,color='green')
    sns.kdeplot(scores[index_abnormal],fill=True,color='red')
    plt.legend(labels=['Normal', 'Abnormal'])
    plt.title(title)
    plt.show()

