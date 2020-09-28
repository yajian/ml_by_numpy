# coding=utf-8
# Reference:**********************************************
# @Time     : 2020/8/17 4:59 PM
# @Author   : huangyajian
# @File     : train.py
# @Software : PyCharm
# @Comment  :
# Reference:**********************************************

import preprocess
from rnn import RNN
import numpy as np

dat = []
data = preprocess.Dataset()
rnn = RNN(data.vocabulary_size)
x = data.X_train[20006]
print(x)
np.random.seed(10)

# Train on a small subset of the data to see what happens
print("正在训练...")
losses = RNN.train(rnn, data.X_train[1:2], data.Y_train[1:2], nepoch=10, evaluate_loss_after=1)
predict = rnn.predict(x)

print("predict shape = " + str(predict.shape))
print(predict)
array_of_words = " ".join([data.index_to_word[x] for x in predict])

print(array_of_words)
