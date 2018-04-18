#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import random

import numpy as np


def read_data(filename):
    print('filename =', filename)
    """ 把文本文件读进来，拆分为单个字符 """
    with open(filename, encoding="utf-8") as f:
        data = f.read()
    data = list(data)
    return data


def index_data(sentences, dictionary):
    """ 使用字典 dictionary 把句子转换为索引 """
    shape = sentences.shape
    sentences = sentences.reshape([-1])
    index = np.zeros_like(sentences, dtype=np.int32)
    for i in range(len(sentences)):
        try:
            index[i] = dictionary[sentences[i]]
        except KeyError:
            index[i] = dictionary['UNK']

    return index.reshape(shape)


def get_train_data(vocabulary, batch_size, num_steps):
    ##################
    # Your Code here
    ##################
    """ 生成每个 batch 的训练数据 
    vocabulary 输入数据的原文，例如全宋词。是一个字符列表，每个元素是单个字符。【已根据字典，转换为索引】
    return 一个元组（data，label） 元素是单词列表
     """
    print('vocabulary =', vocabulary[:10])
    #raw_x = np.array(vocabulary, dtype=np.str)
    raw_x = vocabulary
    # 真值是，输入字符的下个字符的索引，也就是说，目的是根据输入字符预测下个字符。
    raw_y = vocabulary[1:]
    #为什么尾部要加一个，因为真值比输入少一个，最后一个字符是个句号，没有下一个了。
    raw_y.append(0) 
    data_size = len(raw_x)
    print('data_size =', data_size)
    
    # 根据 batch_size 拆分为若干段 
    data_partition_size = data_size // batch_size
    data_x = np.zeros([batch_size, data_partition_size], dtype=np.int32)
    data_y = np.zeros([batch_size, data_partition_size], dtype=np.int32)
    
    for i in range(batch_size):
        data_x[i] = raw_x[data_partition_size * i:data_partition_size * (i + 1)]
        data_y[i] = raw_y[data_partition_size * i:data_partition_size * (i + 1)]
    # further divide batch partitions into num_steps for truncated backprop
    #根据时间维度大小，从每一段中分若干次读取数据 
    epoch_size = data_partition_size // num_steps
    for i in range(epoch_size):
        x = data_x[:, i * num_steps:(i + 1) * num_steps]
        y = data_y[:, i * num_steps:(i + 1) * num_steps]
        yield (x, y)

def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary
