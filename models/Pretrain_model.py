#!/usr/A/anaconda3/envs/py_3.8 python
# -*- coding: UTF-8 -*-
'''
@Project ：RIB-NASC-v2 
@File    ：Pretrain_model.py
@Author  ：Mingtong Zhang
@Date    ：2025/5/29 16:51 
'''
from transformers import BertConfig, BertModel
from transformers import RobertaModel, RobertaConfig


def get_bert(bert_name, path, get_hidden_states=False, embedding_only=True):
    if 'roberta-base' in bert_name:
        print('load roberta-base')
        model_config = RobertaConfig.from_pretrained(path)
        model_config.output_hidden_states = get_hidden_states
        bert = RobertaModel.from_pretrained(path, config=model_config)
        bert = bert.embeddings if embedding_only else bert
    elif 'bert-base-uncased' in bert_name:
        print('load bert-base-uncased')
        model_config = BertConfig.from_pretrained(path)
        model_config.output_hidden_states = get_hidden_states
        bert = BertModel.from_pretrained(path, config=model_config)
        bert = bert.embeddings if embedding_only else bert
    else:
        bert = None
    return bert
