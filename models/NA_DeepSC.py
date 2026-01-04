#!/usr/A/anaconda3/envs/py_3.8 python
# -*- coding: UTF-8 -*-
'''
@Project ：RIB-NASC-latest 
@File    ：NA_DeepSC.py
@Author  ：Mingtong Zhang
@Date    ：2025/11/27 19:56 
'''
import torch.nn as nn
from models.DeepSC import Encoder, JSC_Encoder, JSC_Decoder, Decoder


class NA_DeepSC(nn.Module):
    def __init__(self, num_layers, src_vocab_size, tgt_vocab_size, model_dim, channel_dim, num_heads, dim_feedforward, dropout=0.1):
        super(NA_DeepSC, self).__init__()

        self.semantic_encoder = Encoder(num_layers, src_vocab_size, model_dim, num_heads, dim_feedforward, dropout)

        self.jsc_encoder = JSC_Encoder(in_features_dim=model_dim, intermediate_dim=2 * model_dim, out_features_dim=channel_dim)

        self.jsc_decoder = JSC_Decoder(in_features_dim=channel_dim, intermediate_dim=2 * model_dim, out_features_dim=model_dim)

        self.semantic_decoder = Decoder(num_layers, tgt_vocab_size, model_dim, num_heads, dim_feedforward, dropout)

        self.dense = nn.Linear(model_dim, tgt_vocab_size)