# coding=utf-8
# Reference:**********************************************
# @Time     : 2020/8/17 3:39 PM
# @Author   : huangyajian
# @File     : preprocess.py
# @Software : PyCharm
# @Comment  :
# Reference:**********************************************

import csv
import numpy as np
import nltk
import itertools
import os
import pickle

unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"


class Dataset(object):

    def preprocess_data(self):
        print("正在读取 reddit-comments-2015-08.csv ....")
        with open('./reddit-comments-2015-08.csv', 'r', encoding='utf-8') as f:
            reader = csv.reader(f, skipinitialspace=True)
            sentences = itertools.chain(
                *[nltk.sent_tokenize(x[0].lower()) for x in reader])
            sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
        print("一共处理了%s句子" % len(sentences))

        # 对句子进行分词
        tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
        # 统计词频
        word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
        print("一共有%s个不同的词" % len(word_freq.items()))

        # 获取出现最多的前vocabulary_size个单词，然后构建它们的单词和索引映射矩阵 index_to_word,word_to_index
        vocab = word_freq.most_common(self.vocabulary_size - 1)
        self.index_to_word = [x[0] for x in vocab]
        self.index_to_word.append(unknown_token)
        self.word_to_index = dict([(word, i) for i, word in enumerate(self.index_to_word)])
        print("正在使用的词汇表大小为%d" % self.vocabulary_size)
        print("词汇表出现的最后一个单词为%s 并且它的出现次数为%d" % (vocab[-1][0], vocab[-1][1]))

        # 将所有没有出现在词汇表中的单词替换成unknown_token
        for i, sent in enumerate(tokenized_sentences):
            tokenized_sentences[i] = [w if w in self.word_to_index else unknown_token for w in sent]

        print("处理前的句子：%s" % sentences[0])
        print("处理后的句子：%s" % tokenized_sentences[0])

        # 构建训练数据
        # x:SENTENCE_START  what are n't you understanding about this ? !
        # [SENTENCE_START, 51, 27, 16, 10, 856, 53, 25, 34, 69]
        # y:what are n't you understanding about this ? ! SENTENCE_END
        # [51, 27, 16, 10, 856, 53, 25, 34, 69, SENTENCE_END]

        self.X_train = np.asarray([[self.word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
        self.Y_train = np.asarray([[self.word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
        print(self.X_train.shape)
        print(self.Y_train.shape)

    def __init__(self):
        self.vocabulary_size = 8000
        print("原先的训练数据：" + str(os.path.exists('train.pkl')))

        if os.path.exists("train.pkl"):
            with open('train.pkl', 'rb') as f:
                print("正在加载数据....")
                self.X_train = pickle.load(f)
                self.Y_train = pickle.load(f)
                self.vocabulary_size = pickle.load(f)
                self.index_to_word = pickle.load(f)
                self.word_to_index = pickle.load(f)

        else:
            print("正在生成新数据...")
            with open("train.pkl", 'wb') as f:
                self.preprocess_data()
                print("正在保存数据...")
                pickle.dump(self.X_train, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.Y_train, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.vocabulary_size, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.index_to_word, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.word_to_index, f, pickle.HIGHEST_PROTOCOL)
                f.flush()


if __name__ == '__main__':
    dataset = Dataset()
    print(dataset.word_to_index)
