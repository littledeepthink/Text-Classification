# coding: utf-8
import numpy as np
import utils
from keras.preprocessing.sequence import pad_sequences
# from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import os

# the dict of labels
def get_label_dict(filename):
    air_dict = dict()
    for text in open(filename, 'r', encoding='utf-8'):
        text = text.split(' ')
        air_name = text[0][:-2]
        if air_name in air_dict:
            air_dict[air_name] += 1
        else:
            air_dict[air_name] = 1
    return air_dict

# the preliminary processed data
def get_preli_data(filename, char2index, label2index):
    X_data, y_data = [], []
    for text in open(filename, 'r', encoding='utf-8'):
        text = text.split(' ')
        # print(text)
        y = label2index[text[0][:-2]]
        X = [char2index.get(s, 0) for s in ''.join(text[1:]).strip() if s != ' ']
        y_data.append(y)
        X_data.append(X)
    return X_data, y_data

# zero padding for X_data
def get_standard_data(X_data, y_data, max_length):
    X_arr = pad_sequences(X_data, maxlen=max_length, dtype='int32', padding='post', truncating='post', value=0)
    # enc = OneHotEncoder(dtype='float32')
    # y_arr = enc._fit_transform(y_data).toarray()
    y_arr = np.array(y_data, dtype='int32')
    return X_arr, y_arr

# the over-sampled standard data
def over_sampled(X_arr, y_arr, air_dict, label2index, thres):
    sample = dict()
    for air, ct in air_dict.items():
        if ct <= thres:
            sample[label2index[air]] = thres
    print('Start Sampling...')
    ros = RandomOverSampler(ratio=sample, random_state=42)
    X_ros, y_ros = ros.fit_sample(X_arr, y_arr)
    return X_ros, y_ros

# 统计文本样本的长度以确定padding时的max_length
def count(data, length):
    ct = []
    num = 0
    for lst in data:
        ct.append(len(lst))
        if len(lst) > length:
            num += 1
    print('max langth is {}'.format(max(ct)))  # 293
    print('min langth is {}'.format(min(ct)))  # 4
    return num / len(ct)

# train/dev/test
def data_split(X_arr, y_arr):
    X_train, X_test, y_train, y_test = train_test_split(X_arr, y_arr, test_size = 0.02734, random_state = 42)
    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.02810, random_state=64)
    return X_train, y_train, X_dev, y_dev, X_test, y_test

if __name__ == '__main__':
    fold_path = os.getcwd() + '\\related_data'
    filename = fold_path + '\data_true.txt'

    # label的预处理
    air_dict = get_label_dict(filename)
    label2index = {o: i for i, o in enumerate(sorted(air_dict.keys()))}
    index2label = dict(enumerate(sorted(air_dict.keys())))

    # print(air_dict, '||', label2index)

    # load pretrained word embedding and save it locally
    wv_path = fold_path + '\wiki_100_utf8.txt'
    vocab, embed = utils.load_pretrained_wordvector(wv_path)
    char2index = {o: i for i, o in enumerate(vocab, 1)}
    index2char = dict(enumerate(vocab, 1))

    n_vocab, n_embed = len(vocab) + 1, len(embed[0])
    char_embed_matrix = np.asarray(embed, dtype='float32')  # 将字符型数据转化为浮点型
    np.save(fold_path + '\char_embed_matrix.npy', char_embed_matrix)

    X_data, y_data = get_preli_data(filename, char2index, label2index) # 列表数据

    # # 统计文本样本的长度以确定padding时的max_length
    # ratio = count(X_data, length=120) # 1005, 1.79%
    # # print(ratio)

    # zero padding for X_data
    X_arr, y_arr = get_standard_data(X_data, y_data, max_length=120)

    # the over-sampled standard data
    X_arr, y_arr = over_sampled(X_arr, y_arr, air_dict, label2index, thres=1000)
    # print(Counter(y_arr), y_arr.shape)

    all_data = data_split(X_arr, y_arr)
    # print(all_data[2].shape)

    lst = ['\X_train.npy', '\y_train.npy', '\X_dev.npy', '\y_dev.npy', '\X_test.npy', '\y_test.npy']
    for name in lst:
        np.save(fold_path + name, all_data[lst.index(name)])
