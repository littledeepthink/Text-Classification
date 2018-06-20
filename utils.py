# coding: utf-8
import numpy as np
import os

def load_pretrained_wordvector(filename):
    vocab, embed = [], []
    file = open(filename, 'r', encoding='utf-8')
    id = 0
    for line in file.readlines():
        temp = line.strip().split()
        if id > 0:
            vocab.append(temp[0])
            embed.append(temp[1:])
        else:
            n_embed = int(temp[1])
            # vocab.append('unk')
            embed.append([0] * n_embed)
        id += 1
    print('Great! Loaded word_embedding successfully !')
    file.close()

    return vocab, embed


if __name__ == '__main__':
    fold_path = os.getcwd() + '\\related_data'

    file_read = fold_path + '\output_final.txt'
    with open(file_read, 'r', encoding='utf-8') as f:
        data = f.readlines()

    # 将预测正确的样本加入新的训练集
    file_write = fold_path + '\data_true.txt'
    f = open(file_write, 'a', encoding='utf-8')
    for s in data:
        air_name = s.split('\t')[-4]
        text = s.split('\t')[-1]
        if air_name[:-2] in text:
            f.write(air_name + ' ' + text.replace('【', '').replace('】', ''))
    f.close()
