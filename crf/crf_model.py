# coding=utf-8
# Reference:**********************************************
# @Time     : 2023/2/1 4:16 PM
# @Author   : huangyajian
# @File     : crf_model.py
# @Software : PyCharm
# @Comment  :
# Reference:**********************************************


import numpy as np
import random
from scipy import special, optimize
from data_processor import getData, get_feature_functions

random.seed(10)

START = '|-'
END = '-|'


def log_dot_vm(loga, logM):
    """通过log向量和log矩阵，计算log(向量 点乘 矩阵)"""
    return special.logsumexp(np.expand_dims(loga, axis=1) + logM, axis=0)


def log_dot_mv(logM, logb):
    """通过log向量和log矩阵，计算log(向量 点乘 矩阵)"""
    return special.logsumexp(logM + np.expand_dims(logb, axis=0), axis=1)


class CRF():
    def __init__(self, feature_functions, labels):
        # 特征模版
        self.ft_run = feature_functions
        # 初始化特征权重参数
        self.w = np.random.randn(len(self.ft_run))
        self.labels = labels
        self.label_id = {l: i for i, l in enumerate(labels)}

    def get_features(self, x_vec):
        """
            给定一个输入x_vec，计算这个输入上的所有的(y',y)组合的特征值。
            size: len(x_vec) + 1, len(y), len(y), len(self.ft_run)
            Axes:
            0 - T or time or sequence index
            1 - y' or previous label
            2 - y  or current  label
            3 - f(y', y, x_vec, i) for i s
        """
        result = np.zeros((len(x_vec) + 1, len(self.labels), len(self.labels), len(self.ft_run)))
        for i in range(len(x_vec) + 1):
            for j, yp in enumerate(self.labels):
                for k, y in enumerate(self.labels):
                    for l, f in enumerate(self.ft_run):
                        result[i, j, k, l] = f(yp, y, x_vec, i)
        return result

    def create_vector_list(self, x, y):
        print("create vector list ...")
        print("total training data num:", len(x))
        # 基于特征模版提取文本特征
        observations = [self.get_features(sentence) for sentence in x]  # 形状 [len(x), len(sentence)+1, Y, Y, K]
        labels = len(y) * [None]  # 形状 [len(x), Y]
        for i in range(len(y)):
            # 增加起始、终止状态
            y[i].insert(0, START)
            y[i].append(END)
            # label离散化
            labels[i] = np.array([self.label_id[y] for y in y[i]], copy=False, dtype=np.int_)
        return observations, labels

    def forward(self, log_M_s, start):
        """
        :param log_M_s: [len(vec)+1, Y, Y]
        :param start:
        :return:
        """
        # 文本长度
        T = log_M_s.shape[0]
        # label的空间大小
        Y = log_M_s.shape[1]
        # 这里是T+1是因为增加了起始、终止状态，最终形成了T+1个转移步骤
        alphas = np.NINF * np.ones((T + 1, Y))  # log0 = ninf, [len(vec)+2, Y]
        alpha = alphas[0]  # [Y]
        alpha[start] = 0  # log1 = 0
        for t in range(1, T + 1):
            alphas[t] = log_dot_vm(alpha, log_M_s[t - 1])  # alpha: [Y], log_M_s[t - 1]:[Y,Y]
            alpha = alphas[t]
        return alphas

    def backward(self, log_M_s, end):
        """
        :param log_M_s: [len(vec)+1, Y, Y]
        :param end:
        :return:
        """
        # 文本长度
        T = log_M_s.shape[0]
        # label的空间大小
        Y = log_M_s.shape[1]
        betas = np.NINF * np.ones((T + 1, Y))  # log0 = ninf, [len(vec)+2, Y]
        beta = betas[-1]
        beta[end] = 0  # log1 = 0
        for t in reversed(range(T)):
            betas[t] = log_dot_mv(log_M_s[t], beta)  # log_M_s[t - 1]:[Y,Y], beta: [Y]
            beta = betas[t]
        return betas

    def neg_likelihood_and_deriv(self, x_vecs, y_vecs, w):
        """
            求负对数似然函数和关于w的偏导。
            关键变量的尺寸中，Y是标注空间的个数，K是特征函数的个数。
            Y=21, K=2332
        """
        likelihood = 0
        derivative = np.zeros(len(self.w))  # [K,1]
        for x_vec, y_vec in zip(x_vecs, y_vecs):
            all_features = x_vec  # [len(vec)+1, Y, Y, K]
            length = x_vec.shape[0]
            yp_vec_ids = y_vec[:-1]
            y_vec_ids = y_vec[1:]
            log_M_s = np.dot(all_features, w)  # [len(vec)+1, Y, Y]
            log_alphas = self.forward(log_M_s, self.label_id[START])  # [len(x_vec) + 2, Y]
            log_betas = self.backward(log_M_s, self.label_id[END])  # [len(x_vec) + 2, Y]
            last = log_alphas[-1]
            # 规范化因子
            log_Z = special.logsumexp(last)
            # reshape
            log_alphas1 = np.expand_dims(log_alphas[1:], axis=2)
            log_betas1 = np.expand_dims(log_betas[:-1], axis=1)
            # log_probs: len(x_vec) + 1, Y, Y
            log_probs = log_alphas1 + log_M_s + log_betas1 - log_Z
            log_probs = np.expand_dims(log_probs, axis=3)
            # 计算特征函数关于模型的期望，即f_k关于条件分布P(Y|X)的期望
            exp_features = np.sum(np.exp(log_probs) * all_features, axis=(0, 1, 2))
            # 计算特征函数关于训练数据的期望，即f_k关于联合分布P(Y,X)的期望
            emp_features = np.sum(all_features[range(length), yp_vec_ids, y_vec_ids], axis=0)
            # 计算似然函数
            likelihood += np.sum(log_M_s[range(length), yp_vec_ids, y_vec_ids]) - log_Z
            # 计算似然函数的偏导
            derivative += emp_features - exp_features
        return -likelihood, -derivative

    def train(self, x, y, debug=False):
        # x_vecs的形状 [len(x), 21, 21, K]
        # y_vecs的形状 [len(x), len(x[0])+2]，包括了起始、终止状态，所以是len(x[0])+2
        x_vecs, y_vecs = self.create_vector_list(x, y)
        print("start training ...")
        l = lambda w: self.neg_likelihood_and_deriv(x_vecs, y_vecs, w)
        val = optimize.fmin_l_bfgs_b(l, self.w)
        if debug:
            print(val)
        self.w, _, _ = val
        return self.w

    def predict(self, x_vec):
        """给定x，预测y。使用Viterbi算法"""
        all_features = self.get_features(x_vec)
        log_potential = np.dot(all_features, self.w)  # [len(x_vec), 21, 21]
        T = len(x_vec)
        Y = len(self.labels)
        # Psi保存每个时刻最优情况的下标
        Psi = np.ones((T, Y), dtype=np.int32) * -1  # [len(x_vec), 21]
        # 初始化
        delta = log_potential[0, 0]
        # 递推
        for t in range(1, T):
            next_delta = np.zeros(Y)
            for y in range(Y):
                w = delta + log_potential[t, :, y]
                Psi[t, y] = psi = w.argmax()
                next_delta[y] = w[psi]
            delta = next_delta
        # 回溯找到最优路径
        y = delta.argmax()
        trace = []
        for t in reversed(range(T)):
            trace.append(y)
            y = Psi[t, y]
        trace.reverse()
        return [self.labels[i] for i in trace]


if __name__ == '__main__':
    labels, observes, word_sets, word_data, label_data = getData('./sample.txt')
    feature_functions = get_feature_functions(word_sets, labels, observes)
    crf = CRF(feature_functions, labels)
    crf.train(word_data, label_data)
    for x_vec, y_vec in zip(word_data[-5:], label_data[-5:]):
        print("raw data: ", x_vec)
        print("prediction: ", crf.predict(x_vec))
        print("ground truth: ", y_vec)
