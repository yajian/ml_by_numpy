# coding=utf-8
# Reference:**********************************************
# @Time     : 2020/9/27 6:04 PM
# @Author   : huangyajian
# @File     : metrics.py
# @Software : PyCharm
# @Comment  :
# Reference:**********************************************

import numpy as np

from utils import softmax


def wrapper(x):
    return x.reshape(x.shape[0], )


# cross-entrocpy loss
def forward_cross_entropy_loss(outputs, y_onehot, mask):
    """
    计算交叉熵损失
    :param outputs: 模型输出
    :param y_onehot: 实际label
    :param mask:
    :return:
    """
    softmax_x = softmax(outputs)
    cross_sum = -np.multiply(y_onehot, np.log(softmax_x))
    cross_sum = wrapper(np.sum(cross_sum, axis=1)).astype(np.float32)  # todo, attention shape here!
    # start operation
    mask = np.array(mask, dtype=np.float32)
    mask /= np.mean(mask)
    cross_sum = np.multiply(cross_sum, mask)
    return np.mean(cross_sum)


def backward_cross_entropy_loss(outputs, y_onehot, train_mask):
    """
    softmax反向传播
    :param outputs:
    :param y_onehot:
    :param train_mask:
    :return:
    """
    # softmax的求导公式
    dX = softmax(outputs) - y_onehot
    mask = np.array(train_mask, dtype=np.float32)
    mask /= np.mean(mask)
    dX = np.multiply(dX, mask.reshape(-1, 1))
    return dX / outputs.shape[0]


def l2_loss(x):
    x_square = x ** 2
    x_sum = np.sum(x_square)
    x_l2 = x_sum / 2
    return x_l2


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking"""
    preds_max = np.argmax(preds, axis=1)
    labels_max = np.argmax(labels, axis=1)
    correct_predictions = np.equal(wrapper(preds_max), labels_max)
    accuracy_all = np.array(correct_predictions, dtype=np.float32)
    mask = np.array(mask, dtype=np.float32)
    mask /= np.mean(mask)
    accuracy_all *= mask
    return np.mean(accuracy_all)
