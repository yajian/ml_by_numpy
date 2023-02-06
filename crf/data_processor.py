# coding=utf-8
# Reference:**********************************************
# @Time     : 2023/2/1 4:06 PM
# @Author   : huangyajian
# @File     : data_processor.py
# @Software : PyCharm
# @Comment  : 参考https://github.com/applenob/simple_crf
# Reference:**********************************************


from collections import defaultdict
import re

START = '|-'
END = '-|'


def get_feature_functions(word_sets, labels, observes):
    """生成各种特征函数"""
    print("get feature functions ...")
    # 状态转移特征模版：
    # _yp是前一个状态的label
    # _y是当前状态的label
    # 这个模版的意思是：输入yp和y，如果yp和_yp相同、y和_y相同，说明与转移模版匹配，则取值为1，否则为0
    # 因为label有21个取值，所以这个特征模版有20 * 20 = 400个
    transition_functions = [
        lambda yp, y, x_v, i, _yp=_yp, _y=_y: 1 if yp == _yp and y == _y else 0
        for _yp in labels[:-1] for _y in labels[1:]
    ]

    # # 状态特征模版：
    # # _y是当前的label，即词性
    # # _x是当前的观测值，即文本
    # # 这个模版的意思是：输入y、x_v、i，如果x_v[i]和观测值_x一样、y和_y一样，说明与状态特征模版匹配，则取值为1，否则为0
    # # label有21个取值，observes有92个取值，所以共有21 * 92=1932个特征模版
    # tagval_functions = [
    #     lambda yp, y, x_v, i, _y=_y, _x=_x: 1 if i < len(x_v) and y == _y and x_v[i].lower() == _x else 0
    #     for _y in labels
    #     for _x in observes]
    return transition_functions


def getData(path):
    word_data = []
    label_data = []
    all_labels = set()
    word_sets = defaultdict(set)
    observes = set()
    for line in open(path, 'r'):
        words, labels = [], [] # 共4条样本，单词数[37, 27, 29, 36]
        tokens = line.strip().split()
        for token in tokens:
            word, label = token.split('/')
            all_labels.add(label)
            word_sets[label].add(word.lower())
            observes.add(word.lower())
            words.append(word)
            labels.append(label)
        word_data.append(words)
        label_data.append(labels)
    labels = [START, END] + list(all_labels)
    return labels, observes, word_sets, word_data, label_data


def main():
    data = getData('./sample.txt')
    print(data[0])
    print(data[1])
    print(data[2])


if __name__ == '__main__':
    main()
