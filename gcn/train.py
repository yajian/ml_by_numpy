# coding=utf-8
# Reference:**********************************************
# @Time     : 2020/9/28 10:49 AM
# @Author   : huangyajian
# @File     : train.py
# @Software : PyCharm
# @Comment  :
# Reference:**********************************************

from data_loader import gcn_load_data
from model import GCN

load_data_function = gcn_load_data
gcn_model = GCN(load_data_function=load_data_function, hidden_unit=16, learning_rate=0.01, weight_decay=5e-4)
cost_val = []
epochs = 200
for i in range(epochs):
    train_loss, train_acc = gcn_model.train()
    # print("model loss: {}, model acc: {}".format(train_loss, train_acc))

    gcn_model.update()
    # val step
    val_loss, val_acc = gcn_model.eval()
    print("iteration: {}, train_loss: {}, train_acc: {}, val_loss: {}, val_acc: {}".
          format(i, train_loss, train_acc, val_loss, val_acc))
    cost_val.append(val_loss)
test_loss, test_acc = gcn_model.test()
print("start test, the loss: {}, the acc: {}".format(test_loss, test_acc))
