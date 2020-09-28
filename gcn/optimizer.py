# coding=utf-8
# Reference:**********************************************
# @Time     : 2020/9/28 10:35 AM
# @Author   : huangyajian
# @File     : optimizer.py
# @Software : PyCharm
# @Comment  :
# Reference:**********************************************

import numpy as np


class Adam(object):
    def __init__(self, weights, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.theta_t = weights
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon = 1e-8
        self.m_t = np.zeros(weights.shape)
        self.v_t = np.zeros(weights.shape)
        self.t = 0

    def minimize(self, g_t):
        self.t += 1
        alpha_t = self.learning_rate * ((1 - self.beta_2 ** self.t) ** 0.5) / (1 - self.beta_1 ** self.t)
        self.m_t = self.beta_1 * self.m_t + (1 - self.beta_1) * g_t
        self.v_t = self.beta_2 * self.v_t + (1 - self.beta_2) * np.multiply(g_t, g_t)
        self.theta_t -= alpha_t * self.m_t / (np.power(self.v_t, 0.5) + self.epsilon)
