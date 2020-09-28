# coding=utf-8
# Reference:**********************************************
# @Time     : 2020/8/10 5:36 PM
# @Author   : huangyajian
# @File     : mlp.py
# @Software : PyCharm
# @Comment  :
# Reference:**********************************************
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def initialize_parameters(layer_dims):
    np.random.seed(3)
    L = len(layer_dims)
    parameters = {}
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.1
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters


def relu_forward(Z):
    A = np.maximum(0, Z)
    return A


def linear_forward(x, w, b):
    z = np.dot(w, x) + b
    return z


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A


def forward_propagation(X, parameters):
    # parameters里包含w和b，所以除2
    L = len(parameters) // 2
    A = X
    caches = []
    for l in range(1, L):
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        z = linear_forward(A, W, b)
        caches.append((A, W, b, z))
        A = relu_forward(z)

    WL = parameters['W' + str(L)]
    bL = parameters['b' + str(L)]
    zL = linear_forward(A, WL, bL)
    caches.append((A, WL, bL, zL))
    AL = sigmoid(zL)
    return AL, caches


def compute_cost(AL, Y):
    """
    :param AL: 最后一层的激活值，即预测值
    :param Y: 真实label
    :return: cost
    """
    m = Y.shape[1]
    # multiply是对应位置元素相乘
    cost = 1.0 / m * np.nansum(np.multiply(-np.log(AL), Y) + np.multiply(-np.log(1 - AL), 1 - Y))
    # 把shape中为1的维度去掉
    cost = np.squeeze(cost)
    return cost


def relu_backward(dA, z):
    dout = np.multiply(dA, np.int64(z > 0))
    return dout


def linear_backward(dZ, cache):
    # dw和db是这一层w和b的梯度
    # da是为了计算下一层w梯度使用的
    A, W, b, z = cache
    dW = np.dot(dZ, A.T)
    db = np.sum(dZ, axis=1, keepdims=True)
    da = np.dot(W.T, dZ)
    return da, dW, db


def backward_propagation(AL, Y, caches):
    m = Y.shape[1]
    L = len(caches) - 1
    # 交叉熵的导数
    dz = 1. / m * (AL - Y)
    da, dWL, dbL = linear_backward(dz, caches[L])
    gradients = {"dW" + str(L + 1): dWL,
                 "db" + str(L + 1): dbL}
    for l in reversed(range(0, L)):
        A, W, b, z = caches[l]
        # 这里计算激活函数的导数
        dout = relu_backward(da, z)
        da, dW, db = linear_backward(dout, caches[l])
        gradients["dW" + str(l + 1)] = dW
        gradients["db" + str(l + 1)] = db
    return gradients


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
    return parameters


def L_layer_model(X, Y, layer_dims, learning_rate, num_iterations):
    costs = []
    parameters = initialize_parameters(layer_dims)
    # for k,v in parameters.items():
    #     print(k, v.shape)
    for i in range(0, num_iterations):
        AL, caches = forward_propagation(X, parameters)
        cost = compute_cost(AL, Y)
        if i % 1000 == 0:
            print("cost after iteration:{}: {}".format(i, cost))
            costs.append(cost)
        grads = backward_propagation(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)

    print("length of cost")
    print(len(costs))
    plt.clf()
    plt.plot(costs)
    plt.xlabel("iterations(thousand)")
    plt.ylabel("cost")
    plt.show()
    return parameters


def predict(X_test, y_test, parameters):
    m = y_test.shape[1]
    Y_prediction = np.zeros((1, m))
    prob, caches = forward_propagation(X_test, parameters)
    for i in range(prob.shape[1]):
        if prob[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0
    accuracy = 1 - np.mean(np.abs(Y_prediction) - y_test)
    return accuracy


def DNN(X_train, y_train, X_test, y_test, layer_dims, learning_rate=0.001, num_iterations=30000):
    parameters = L_layer_model(X_train, y_train, layer_dims, learning_rate, num_iterations)
    accuracy = predict(X_test, y_test, parameters)
    return accuracy


if __name__ == '__main__':
    # x_data: (569,30), y_data:(569,)
    X_data, y_data = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, train_size=0.8, random_state=28)
    # X_train: (30, 455)
    X_train = X_train.T
    # y_train: (1, 455)
    y_train = y_train.reshape(y_train.shape[0], -1).T
    # x_test: (30, 114)
    X_test = X_test.T
    # y_test: (1, 114)
    y_test = y_test.reshape(y_test.shape[0], -1).T

    accuracy = DNN(X_train, y_train, X_test, y_test, [X_train.shape[0], 10, 5, 1])
    print(accuracy)
