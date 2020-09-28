# coding=utf-8
# Reference:**********************************************
# @Time     : 2020/9/27 5:53 PM
# @Author   : huangyajian
# @File     : data_loader.py
# @Software : PyCharm
# @Comment  :
# Reference:**********************************************

import pickle
import numpy as np
import scipy.sparse as sp
import networkx as nx


def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def gcn_load_data():
    """
    ind.cora.x: train训练集实例的特征向量
    ind.cora.tx: test测试集实例的特征向量
    ind.cora.allx: 所有(有标签和无标签)的train训练实例特征向量
    ind.cora.y: train训练数据集的one-hot类型的标签向量
    ind.cora.ty: test测试数据集的one-hot类型的标签向量
    ind.cora.ally: 所有有标签数据的one-hot类型标签向量
    ind.cora.graph: 是一个字典{index:[index_of_neighbor_nodes]}
    ind.cora.test.index: 测试数据集的index

    :return:
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open('./data/ind.cora.{}'.format(names[i]), 'rb') as f:
            objects.append(pickle.load(f, encoding='latin1'))

    """
    x: (140, 1433)
    y: (140, 7)，一共7类，每类抽了20个case
    tx: (1000, 1433)
    ty: (1000, 7)
    allx: (1708, 1433)
    ally: (1708, 7)
    综上，训练集140，测试数据共1000条
    """
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx = []
    with open('./data/ind.cora.test.index', 'r') as f:
        for line in f.readlines():
            test_idx.append(int(line))
    test_idx_reorder = np.sort(test_idx)
    # train + test数据集特征向量
    features = sp.vstack((allx, tx)).tolil()
    # 调整test数据集在features的顺序，tx是乱序的，简单拼接之后tx的序号和features的行号对应不上
    # tx中第一个instance原始下标应该是2692但拼接后是1708，下面的调整是将第1708条instance放到2692位置上
    features[test_idx, :] = features[test_idx_reorder, :]
    # 邻接矩阵
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    labels = np.vstack((ally, ty))
    labels[test_idx, :] = labels[test_idx_reorder, :]
    # 测试数据，1000条
    idx_test = test_idx_reorder.tolist()
    # 训练数据，140条
    idx_train = range(len(y))
    # 验证数据，500条
    idx_val = range(len(y), len(y) + 500)
    """ 
    bool [True  True  True ... False False False]
    train_mask = [0,140)为True 其余为False
    val_mask = [140,640)为True 其余为False
    test_mask = [1708, 2707]为True 其余为False
    """
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    """
    adj: 链接矩阵
    features: 特征矩阵
    y_train: 训练数据的label
    y_val: 验证数据的label
    y_test: 测试数据的label
    train_mask: 训练数据label的mask参数 
    val_mask: 验证数据的label的mask参数 
    test_mask:  测试数据的label的mask参数 
    """
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


if __name__ == '__main__':
    gcn_load_data()
