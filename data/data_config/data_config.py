#!/usr/A/anaconda3/envs/py_3.8 python
# -*- coding: UTF-8 -*-
'''
@Project ：RIB-NASC-latest 
@File    ：data_config.py
@Author  ：Mingtong Zhang
@Date    ：2025/12/3 10:02 
'''
import argparse
from utils.utils_general import load_config


def get_data_parser(data_config_number):
    config_file_path = f'./data_config/data_config_{data_config_number}.txt'
    config_dict = load_config(config_file_path)

    parser = get_mian_parser(syn_vocab_name=config_dict['syn_vocab_name'], num_syn_token=config_dict['num_syn_token'], syn_th=config_dict['syn_th'],
                             seed=config_dict['seed'], stop_words=config_dict.get('stop_words'))

    return parser


def get_mian_parser(syn_vocab_name, num_syn_token, syn_th, seed, stop_words):
    parser = argparse.ArgumentParser()
    # File path parameters
    parser.add_argument('--train-data-path', default='../data/europarl/train_data.pkl', type=str)
    parser.add_argument('--test-data-path', default='../data/europarl/test_data.pkl', type=str)
    parser.add_argument('--vocab-file', default='../data/europarl/vocab.pkl', type=str)
    parser.add_argument('--syn_vocab_file', default=f'../data/europarl/{syn_vocab_name}.pkl', type=str)
    parser.add_argument('--num_syn_token', default=num_syn_token, type=int)
    parser.add_argument('--syn_th', default=syn_th, type=float)
    parser.add_argument('--seed', default=seed, type=int)
    parser.add_argument('--stop_words', default=stop_words, type=str)

    return parser