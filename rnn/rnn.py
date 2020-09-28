# coding=utf-8
# Reference:**********************************************
# @Time     : 2020/8/13 2:59 PM
# @Author   : huangyajian
# @File     : rnn.py
# @Software : PyCharm
# @Comment  :
# Reference:**********************************************
import numpy as np
from datetime import datetime


class RNN:
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate

        self.W = np.random.uniform(-np.sqrt(1. / word_dim), -np.sqrt(1. / word_dim), (hidden_dim, word_dim))
        self.U = np.random.uniform(-np.sqrt(1. / hidden_dim), -np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))
        self.V = np.random.uniform(-np.sqrt(1. / hidden_dim), -np.sqrt(1. / hidden_dim), (word_dim, hidden_dim))

    def forward_propagation(self, x):
        T = len(x)
        # hidden states 初始化为0
        s = np.zeros((T + 1, self.hidden_dim))
        # 这里把第0个状态放在了s数组最后面，这样t就不用从1开始取值了
        s[-1] = np.zeros(self.hidden_dim)
        # output zeros
        o = np.zeros((T, self.word_dim))
        for t in range(T):
            # 若输入x是一个one-hot向量时，W*x操作是把W矩阵中第i列的元素取出（i表示非零元素的下标，本示例中即x[t]）
            s[t] = np.tanh(self.W[:, x[t]] + self.U.dot(s[t - 1]))  # t时刻的状态计算
            o[t] = self.softmax(self.V.dot(s[t]))  # t时刻输出的计算
        return [o, s]

    def softmax(self, x):
        # 这里减去最大值是防止出现上溢和下溢
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def calculate_loss(self, x, y):
        N = np.sum(len(y_i) for y_i in y)
        return self.calculate_total_loss(x, y) / N

    def calculate_total_loss(self, x, y):
        # 总体loss
        L = 0
        for i in range(len(y)):
            # 正向传播
            o, s = self.forward_propagation(x[i])
            # 从输出矩阵中获取每个时刻的输出，o是on
            # range(len(y[i]))代表每个时刻即1-20，在o中是横坐标，y[i]在o中代表纵坐标
            correct_word_predictions = o[range(len(y[i])), y[i]]
            # logloss
            L += -1 * np.sum(np.log(correct_word_predictions))
        return L

    def bptt(self, x, y):
        T = len(y)
        o, s = self.forward_propagation(x)
        # 参数w的梯度
        dLdW = np.zeros(self.W.shape)
        # 参数U的梯度
        dLdU = np.zeros(self.U.shape)
        # 参数V的梯度
        dLdV = np.zeros(self.V.shape)
        delta_o = o
        # 计算残差
        delta_o[range(len(y)), y] -= 1.0
        for t in reversed(range(T)):
            # 计算V的梯度，，参考公式（1）
            dLdV = np.outer(delta_o[t], s[t].T)
            # 计算每个时刻t的delta值，这里是以t时刻为起始向前传播，参考公式（3）
            delta_t = self.V.T.dot(delta_o[t]) * (1 - s[t] ** 2)
            # 计算(t-4, t]之间时刻产生的梯度
            for bptt_step in reversed(range(max(0, t - self.bptt_truncate), t + 1)):
                # print("Backpropagation step t=%d bptt step=%d " % (t, bptt_step))
                # 计算U的梯度，参考公式（4）
                dLdU += np.outer(delta_t, s[bptt_step - 1])
                # 计算W的梯度，参考公式（5）
                dLdW[:, x[bptt_step]] += delta_t
                # 更新delta_t，参考公式（2）
                delta_t = self.U.T.dot(delta_t) * (1 - s[bptt_step - 1] ** 2)
        return [dLdW, dLdV, dLdU]

    def sgd(self, x, y, learning_rate):

        dLdW, dLdV, dLdU = self.bptt(x, y)
        self.W -= learning_rate * dLdW
        self.V -= learning_rate * dLdV
        self.U -= learning_rate * dLdU

    def predict(self, x):
        o, s = self.forward_propagation(x)
        return np.argmax(o, axis=1)

    def train(self, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
        losses = []
        num_examples_seen = 0
        for epoch in range(nepoch):
            if epoch % evaluate_loss_after == 0:
                loss = self.calculate_loss(X_train, y_train)
                losses.append((num_examples_seen, loss))
                time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss))
                # Adjust the learning rate if loss increases
                if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                    learning_rate = learning_rate * 0.5
                    print("Setting learning rate to %f" % learning_rate)
            for i in range(len(y_train)):
                self.sgd(X_train[i], y_train[i], learning_rate)
                num_examples_seen += 1
