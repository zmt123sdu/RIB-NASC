#!/usr/A/anaconda3/envs/py_3.8 python
# -*- coding: UTF-8 -*-
'''
@Project ：RIB-NASC-latest 
@File    ：dataset.py
@Author  ：Mingtong Zhang
@Date    ：2025/11/27 21:50 
'''
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


class EurDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)

    def __getitem__(self, index):
        sents = self.data[index]
        return sents

    def __len__(self):
        return len(self.data)


class collater():
    def __init__(self, fixed_length_padding=False, len_max=32):
        self.fixed_length_padding = fixed_length_padding
        self.len_max = len_max

    def __call__(self, batch):

        batch_size = len(batch)

        if self.fixed_length_padding:
            """使每个句子都padding到最大固定长度"""
            len_max = self.len_max
            sents = np.zeros((batch_size, len_max), dtype=np.int64)
            sort_by_len = sorted(batch, key=lambda x: len(x), reverse=True)

            for i, sent in enumerate(sort_by_len):
                length = len(sent)
                sents[i, :length] = sent  # padding the questions
        else:
            """使每个batch中的每个句子都padding到当前batch最大长度"""
            len_max = max(map(lambda x: len(x), batch))  # get the max length of tgt sentence in current batch
            sents = np.zeros((batch_size, len_max), dtype=np.int64)
            sort_by_len = sorted(batch, key=lambda x: len(x), reverse=True)

            for i, sent in enumerate(sort_by_len):
                length = len(sent)
                sents[i, :length] = sent  # padding the questions

        return torch.from_numpy(sents)


class syn_collater():
    def __init__(self, len_max, num_syn_token=2):
        self.len_max = len_max
        self.num_syn_token = num_syn_token

    def __call__(self, batch):

        batch_size = len(batch)
        sort_by_len = sorted(batch, key=lambda x: len(x[0]), reverse=True)
        sort_by_len_ori = [item[0] for item in sort_by_len]  # 提取每个元素的第一个列表
        sort_by_len_syn = [item[1: self.num_syn_token] for item in sort_by_len]  # 提取每个元素的第二个列表

        """使每个句子都padding到最大固定长度"""
        len_max = self.len_max
        sents_ori = np.zeros((batch_size, len_max), dtype=np.int64)
        sents_syn = np.zeros((batch_size, self.num_syn_token - 1, len_max), dtype=np.int64)

        for i, sent in enumerate(sort_by_len_ori):
            length = len(sent)
            sents_ori[i, :length] = sent  # padding the questions

        for i, sent in enumerate(sort_by_len_syn):
            for j in range(self.num_syn_token - 1):
                length = len(sent[j])
                sents_syn[i, j, :length] = sent[j]

        return torch.from_numpy(sents_ori), torch.from_numpy(sents_syn)
