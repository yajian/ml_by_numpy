# coding=utf-8
# Reference:**********************************************
# @Time     : 2020/9/27 5:57 PM
# @Author   : huangyajian
# @File     : model.py
# @Software : PyCharm
# @Comment  :
# Reference:**********************************************

from weight_processor import preprocess_adj, preprocess_features, init_weight
from layer import forward_hidden, backward_hidden
import numpy as np
from metrics import forward_cross_entropy_loss, l2_loss, masked_accuracy, backward_cross_entropy_loss
from optimizer import Adam


class GCN(object):
    def __init__(self, load_data_function, hidden_unit=16, learning_rate=0.01, weight_decay=5e-4):
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data_function()
        # 邻接矩阵再正则化
        self.adj = preprocess_adj(adj)
        # 特征归一化
        self.features = preprocess_features(features)  # preprocess

        self.y_train, self.train_mask = y_train, train_mask
        self.y_val, self.val_mask = y_val, val_mask
        self.y_test, self.test_mask = y_test, test_mask
        # init
        # 节点个数
        self.n = adj.shape[0]
        # 特征个数，1433维
        self.f = features.shape[1]
        # 类别个数，7维
        self.c = y_train.shape[1]
        # 隐藏单元个数
        self.h = hidden_unit
        # init weight
        # 第一层权重参数
        self.weight_hidden = init_weight((self.f, self.h))
        # 第二层权重参数
        self.weight_outputs = init_weight((self.h, self.c))

        self.adam_weight_hidden = Adam(weights=self.weight_hidden, learning_rate=learning_rate)
        self.adam_weight_outputs = Adam(weights=self.weight_outputs, learning_rate=learning_rate)
        # 第二层输入，反向传播使用
        self.hidden = np.zeros((self.n, self.h))
        # 第二层输出，反向传播使用
        self.outputs = np.zeros((self.n, self.c))

        self.weight_decay = weight_decay

        # test
        self.grad_loss = None
        self.grad_weight_outputs = None
        self.grad_hidden = None
        self.grad_weight_hidden = None

    def train(self):
        # outputs的输出形状(2708，16)，即(batch, hidden)
        self.hidden = forward_hidden(self.adj, self.features, self.weight_hidden, activation=lambda x: np.maximum(x, 0))
        # outputs的输出形状(2708，7)，即(batch, classes)
        self.outputs = forward_hidden(self.adj, self.hidden, self.weight_outputs)
        # 正向传播过程中的loss
        loss = forward_cross_entropy_loss(self.outputs, self.y_train, self.train_mask)
        # 正则项
        weight_decay_loss = self.weight_decay * l2_loss(self.weight_hidden)
        loss += weight_decay_loss
        # 计算准确率
        acc = masked_accuracy(self.outputs, self.y_train, self.train_mask)
        return loss, acc

    def update(self):
        y_train, train_mask = self.y_train, self.train_mask
        # 计算交叉熵损失函数
        grad_loss = backward_cross_entropy_loss(self.outputs, y_train, train_mask)
        # 计算输出层梯度，grad_hidden是第二层输入的梯度，grad_weight_outputs是第二层权重参数梯度
        grad_hidden, grad_weight_outputs = backward_hidden(self.adj, self.hidden, self.weight_outputs, grad_loss,
                                                           mask=train_mask, mask_flag=True)
        # 计算隐藏层梯度，grad_weight_hidden是第一层权重参数梯度
        _, grad_weight_hidden = backward_hidden(self.adj, self.features, self.weight_hidden, grad_hidden,
                                                mask=train_mask, backward_act=lambda x: np.where(x <= 0, 0, 1))
        grad_weight_hidden += self.weight_decay * self.weight_hidden
        self.grad_loss = grad_loss
        self.grad_weight_outputs = grad_weight_outputs
        self.grad_hidden = grad_hidden
        self.adam_weight_hidden.minimize(grad_weight_hidden)
        self.adam_weight_outputs.minimize(grad_weight_outputs)
        self.weight_hidden = self.adam_weight_hidden.theta_t
        self.weight_outputs = self.adam_weight_outputs.theta_t

    def eval(self):
        y_train, train_mask = self.y_val, self.val_mask
        hidden = forward_hidden(self.adj, self.features, self.weight_hidden, activation=lambda x: np.maximum(x, 0))
        outputs = forward_hidden(self.adj, hidden, self.weight_outputs)
        loss = forward_cross_entropy_loss(outputs, y_train, train_mask)
        loss += self.weight_decay * l2_loss(self.weight_hidden)
        acc = masked_accuracy(outputs, y_train, train_mask)
        return loss, acc

    def test(self):
        y_train, train_mask = self.y_test, self.test_mask
        hidden = forward_hidden(self.adj, self.features, self.weight_hidden, activation=lambda x: np.maximum(x, 0))
        outputs = forward_hidden(self.adj, hidden, self.weight_outputs)
        loss = forward_cross_entropy_loss(outputs, y_train, train_mask)
        loss += self.weight_decay * l2_loss(self.weight_hidden)
        acc = masked_accuracy(outputs, y_train, train_mask)
        return loss, acc
