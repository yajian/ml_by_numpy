# coding=utf-8
# Reference:**********************************************
# @Time     : 2020/9/27 6:00 PM
# @Author   : huangyajian
# @File     : layer.py
# @Software : PyCharm
# @Comment  :
# Reference:**********************************************

import numpy as np

def forward_hidden(adj, hidden, weight_hidden, activation=lambda x: x):
    """
    这里就是gcn按层传播公式的实现
    :param adj: 邻接矩阵
    :param hidden: 输入数据，第一层是feature，第二层是第一层的输出
    :param weight_hidden: 隐层参数
    :param activation: 激活函数
    :return:
    """
    A_hat = np.dot(adj, hidden)
    A_tidle = np.dot(A_hat, weight_hidden)
    H = activation(A_tidle)
    return H


def backward_hidden(adj, hidden, weight_hidden, pre_layer_grad, mask, backward_act=lambda x: np.ones(x.shape),
                    mask_flag=False):
    """
    隐藏层反向传播计算
    :param adj: 邻接矩阵
    :param hidden: 输入数据，第一层是feature，第二层是第一层的输出
    :param weight_hidden: 隐层参数
    :param pre_layer_grad: 前一层梯度
    :param mask:
    :param backward_act:
    :param mask_flag:
    :return:
    """
    A_hat = np.dot(adj, hidden)
    if mask_flag:
        A_hat = np.multiply(A_hat, mask.reshape(-1, 1))
    A_tilde = np.dot(A_hat, weight_hidden)
    dact = backward_act(A_tilde)
    # 权重参数梯度
    dW = np.dot(A_hat.T, np.multiply(pre_layer_grad, dact))
    dAhat = np.dot(np.multiply(pre_layer_grad, dact), weight_hidden.T)
    # 输入的梯度
    dX = np.dot(adj.T, dAhat)
    return dX, dW
