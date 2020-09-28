# coding=utf-8
# Reference:**********************************************
# @Time     : 2020/8/20 10:22 PM
# @Author   : huangyajian
# @File     : lstm.py
# @Software : PyCharm
# @Comment  :
# Reference:**********************************************
import numpy as np


class LSTM(object):

    def __init__(self, data_dim, hidden_dim=100):
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim

        # 输入门第一部分参数矩阵
        self.Wi, self.Ui, self.bi = self._init_variable()
        # 输入门第二部分参数矩阵
        self.Wc, self.Uc, self.bc = self._init_variable()
        # 遗忘门参数矩阵
        self.Wf, self.Uf, self.bf = self._init_variable()
        # 输出门参数矩阵
        self.Wo, self.Uo, self.bo = self._init_variable()

        self.Wy = np.random.uniform(-np.sqrt(1.0 / self.hidden_dim), np.sqrt(1.0 / self.hidden_dim),
                                    (self.data_dim, self.hidden_dim))

        self.by = np.random.uniform(-np.sqrt(1.0 / self.hidden_dim), np.sqrt(1.0 / self.hidden_dim),
                                    (self.data_dim, 1))

    def _init_variable(self):
        W = np.random.uniform(-np.sqrt(1.0 / self.hidden_dim), np.sqrt(1.0 / self.hidden_dim),
                              (self.hidden_dim, self.hidden_dim))
        U = np.random.uniform(-np.sqrt(1.0 / self.data_dim), np.sqrt(1.0 / self.data_dim),
                              (self.hidden_dim, self.data_dim))
        b = np.random.uniform(-np.sqrt(1.0 / self.data_dim), np.sqrt(1.0 / self.data_dim),
                              (self.hidden_dim, 1))
        return W, U, b

    def _init_s(self, T):
        # 记录每个时刻的输入门状态
        iss = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))
        # 记录每个时刻的遗忘门状态
        fss = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))
        # 记录每个时刻的输出门状态
        oss = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))
        # 记录每个时刻c'的状态
        ass = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))
        # 记录每个时刻隐状态的状态
        hss = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))
        # 记录每个时刻的细胞状态
        css = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))
        # 记录每个时刻的输出值
        ys = np.array([np.zeros((self.data_dim, 1))] * T)

        return {'iss': iss, 'fss': fss, 'oss': oss,
                'ass': ass, 'hss': hss, 'css': css,
                'ys': ys}

    def forward(self, x):
        T = len(x)
        stats = self._init_s(T)
        for t in range(T):
            # 前一时刻的隐状态
            ht_pre = np.array(stats['hss'][-1]).reshape(-1, 1)
            # 输入门
            stats['iss'][t] = self._cal_gate(self.Wi, self.Ui, self.bi, x, self.sigmoid)
            # 遗忘门
            stats['fss'][t] = self._cal_gate(self.Wf, self.Uf, self.bf, x, self.sigmoid)
            # 输出门
            stats['oss'][t] = self._cal_gate(self.Wo, self.Uo, self.bo, x, self.sigmoid)
            # 用于计算细胞状态的c'
            stats['ass'][t] = self._cal_gate(self.Wc, self.Uc, self.bc, x, self.sigmoid)

            # 细胞状态，ct=ft*ct_pre + it*c'
            stats['css'][t] = stats['fss'][t] * stats['css'][t - 1] + stats['iss'][t] * stats['ass'][t]

            # 隐状态
            stats['hss'][t] = stats['oss'][t] * self.tanh(stats['css'][t])
            # 最终输出
            stats['ys'][t] = self.softmax(self.Wy.dot(stats['hss'][t]) + self.by)

        return stats

    def _init_grad(self):
        dW = np.zeros(self.Wi.shape)
        dU = np.zeros(self.Ui.shape)
        db = np.zeros(self.bi.shape)
        return dW, dU, db

    def bptt(self, x, y):
        dWi, dUi, dbi = self._init_grad()
        dWf, dUf, dbu = self._init_grad()
        dWo, dUo, dbo = self._init_grad()
        dWc, dUc, dbc = self._init_grad()
        dWy = np.zeros(self.Wy.shape)
        dby = np.zeros(self.by.shape)

        # 前向计算
        stats = self.forward(x)
        # 目标函数对输出y的偏导数，即softmax的求导结果y'-y
        delta_o = stats['ys']  # [data_dim, T]
        delta_o[np.arange(len(y)), y] -= 1
        # 以下两项需要累加
        delta_ht = np.zeros((self.hidden_dim, 1))
        delta_ct = np.zeros((self.hidden_dim, 1))
        for t in reversed(range(len(y))):
            # 输出层Wy，by的偏导数
            dWy += delta_o[t].dot(stats['hss'][t].reshape(1, -1))
            dby += delta_o[t]
            # 目标函数对隐状态的偏导数
            delta_ht += self.Wy.T.dot(delta_o[t])  # [hidden_dim ,1]
            # 各个门及状态的偏导数
            delta_ot = delta_ht * self.tanh(stats['css'][t])  # [hidden_dim ,1]
            delta_ct += delta_ht * stats['oss'][t] * (1 - self.tanh(stats['css'][t]) ** 2)  # [hidden_dim ,1]
            delta_it = delta_ct * stats['ass'][t]  # [hidden_dim ,1]
            delta_ft = delta_ct * stats['css'][t - 1]  # [hidden_dim ,1]
            delta_at = delta_ct * stats['iss'][t]  # [hidden_dim ,1]

            delta_it_net = delta_it * stats['iss'][t] * (1 - stats['iss'][t])  # [hidden_dim ,1]
            delta_ft_net = delta_ft * stats['fss'][t] * (1 - stats['fss'][t])  # [hidden_dim ,1]
            delta_ot_net = delta_ot * stats['oss'][t] * (1 - stats['oss'][t])  # [hidden_dim ,1]
            delta_at_net = delta_at * (1 - stats['ass'][t] ** 2)  # [hidden_dim ,1]
            # 更新各权重矩阵的偏导数，由于所有时刻共享权值，故所有时刻累加
            dWi, dUi, dbi = self._cal_grad_delta(dWi, dUi, dbi, delta_it_net, stats['iss'][t - 1], x[t])
            dWc, dUc, dbc = self._cal_grad_delta(dWc, dUc, dbc, delta_at_net, stats['css'][t - 1], x[t])
            dWo, dUo, dbo = self._cal_grad_delta(dWo, dUo, dbo, delta_ot_net, stats['oss'][t - 1], x[t])
            dWf, dUf, dbf = self._cal_grad_delta(dWf, dUf, dbf, delta_ft_net, stats['fss'][t - 1], x[t])
            delta_ht += self.Wo.T.dot(delta_ot_net) + \
                        self.Wi.T.dot(delta_it_net) + \
                        self.Wf.T.dot(delta_ft_net) + \
                        self.Wc.T.dot(delta_at_net)
            delta_ct = delta_ct * stats['fss'][t]

        return [dWf, dUf, dbf, dWi, dUi, dbi, dWc, dUc, dbc, dWo, dUo, dbo, dWy, dby]

    def _cal_grad_delta(self, dW, dU, db, delta_net, ht_pre, x):
        dW += delta_net * ht_pre
        dU += delta_net * x
        db += delta_net
        return dW, dU, db

    def sgd_step(self, x, y, learning_rate):
        dWf, dUf, dbf, \
        dWi, dUi, dbi, \
        dWc, dUc, dbc, \
        dWo, dUo, dbo, \
        dWy, dby = self.bptt(x, y)
        self.Wf, self.Uf, self.bf = self._update_weights(learning_rate, self.Wf, self.Uf, self.bf, dWf, dUf, dbf)
        self.Wi, self.Ui, self.bi = self._update_weights(learning_rate, self.Wi, self.Ui, self.bi, dWi, dUi, dbi)
        self.Wc, self.Uc, self.bc = self._update_weights(learning_rate, self.Wc, self.Uc, self.bc, dWc, dUc, dbc)
        self.Wo, self.Uo, self.bo = self._update_weights(learning_rate, self.Wo, self.Uo, self.bo, dWo, dUo, dbo)
        self.wy = self.wy - learning_rate * dWy
        self.by = self.by - learning_rate * dby

    def _update_weights(self, learning_rate, W, U, b, dW, dU, db):
        W -= learning_rate * dW
        U -= learning_rate * dU
        b -= learning_rate * db
        return W, U, b

    def _cal_gate(self, w, u, b, ht_pre, x, activation):
        return activation(w.dot(ht_pre) + u[:, x].reshape(-1, 1) + b)

    def softmax(self, x):
        return np.exp(x - np.max(x)) / sum(np.exp(x - np.max(x)))

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
