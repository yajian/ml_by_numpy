# coding=utf-8
# Reference:**********************************************
# @Time     : 2020/9/27 5:53 PM
# @Author   : huangyajian
# @File     : utils.py
# @Software : PyCharm
# @Comment  :
# Reference:**********************************************
import numpy as np


def softmax(x):
    """softmax x"""
    exp_x = np.exp(x)
    sum_x = np.sum(exp_x, axis=1).reshape(-1, 1)
    softmax_x = exp_x / sum_x
    return softmax_x
