#!/usr/A/anaconda3/envs/py_3.8 python
# -*- coding: UTF-8 -*-
'''
@Project ：RIB-NASC-latest 
@File    ：NASC.py
@Author  ：Mingtong Zhang
@Date    ：2025/11/27 19:51 
'''
import torch.nn as nn
from models.DeepSC import Encoder, JSC_Encoder, JSC_Decoder, EncoderLayer


class DeEncoder(nn.Module):
    def __init__(self, num_layers, model_dim, num_heads, dim_feedforward, dropout=0.1):
        super(DeEncoder, self).__init__()

        self.d_model = model_dim
        self.num_heads = num_heads
        self.enc_layers = nn.ModuleList([EncoderLayer(model_dim, num_heads, dim_feedforward, dropout) for _ in range(num_layers)])

    def forward(self, x, src_key_mask=None):
        for enc_layer in self.enc_layers:
            x = enc_layer(x, src_key_padding_mask=src_key_mask)
        return x


class NASC(nn.Module):
    def __init__(self, num_layers, src_vocab_size, tgt_vocab_size, model_dim, channel_dim, num_heads, dim_feedforward, dropout=0.1):
        super(NASC, self).__init__()

        self.semantic_encoder = Encoder(num_layers, src_vocab_size, model_dim, num_heads, dim_feedforward, dropout)

        self.jsc_encoder = JSC_Encoder(in_features_dim=model_dim, intermediate_dim=2 * model_dim, out_features_dim=channel_dim)

        self.jsc_decoder = JSC_Decoder(in_features_dim=channel_dim, intermediate_dim=2 * model_dim, out_features_dim=model_dim)

        self.semantic_decoder = DeEncoder(num_layers, model_dim, num_heads, dim_feedforward, dropout)

        self.dense = nn.Linear(model_dim, tgt_vocab_size)
