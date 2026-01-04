#!/usr/A/anaconda3/envs/py_3.8 python
# -*- coding: UTF-8 -*-
'''
@Project ：RIB-NASC-latest 
@File    ：RIB-NASC.py
@Author  ：Mingtong Zhang
@Date    ：2025/12/1 20:03 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.DeepSC import Encoder, JSC_Encoder, JSC_Decoder
from models.NASC import DeEncoder


def shuffling(x):
    idxs = torch.randperm(x.size(0))
    return x[idxs]


class Mine_Net_v0(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(Mine_Net_v0, self).__init__()

        self.dense1 = nn.Linear(x_dim + y_dim, hidden_size)
        self.dense2 = nn.Linear(hidden_size, hidden_size)
        self.dense3 = nn.Linear(hidden_size, x_dim)

    def sample_shuffle_batch(self, z, z_hat):
        z_hat_shuffle = shuffling(z_hat)
        joint = torch.cat((z, z_hat), -1)
        marg = torch.cat((z, z_hat_shuffle), -1)

        return joint, marg

    def t_func(self, inputs):
        x = self.dense1(inputs)
        x = F.relu(x)

        x = self.dense2(x)
        x = F.relu(x)

        output = self.dense3(x).sum(dim=-1)

        return output

    def forward(self, x_samples, y_samples):
        joint, marginal = self.sample_shuffle_batch(x_samples, y_samples)

        joint_scores = self.t_func(joint)
        marg_scores = self.t_func(marginal)

        # 避免指数运算溢出
        marg_max = marg_scores.max().detach()  # 分离最大值以稳定计算
        safe_marg_exp = torch.exp(marg_scores - marg_max)
        log_mean_exp = marg_max + torch.log(torch.mean(safe_marg_exp))
        mi_dq = torch.mean(joint_scores) - log_mean_exp
        return mi_dq


class CLUB_Net_v0(nn.Module):
    def __init__(self, channel_dim, hidden_size, club_num_layers, club_num_heads, tgt_vocab_size):
        super(CLUB_Net_v0, self).__init__()

        self.jsc_decoder = JSC_Decoder(in_features_dim=channel_dim, intermediate_dim=2 * hidden_size, out_features_dim=hidden_size)

        self.semantic_decoder = DeEncoder(club_num_layers, hidden_size, num_heads=club_num_heads, dim_feedforward=4 * hidden_size, dropout=0.1)

        self.dense = nn.Linear(hidden_size, tgt_vocab_size)

    def get_pred(self, x_samples):
        x = self.jsc_decoder(x_samples)
        x = self.semantic_decoder(x)
        pred = self.dense(x)

        return pred

    def loglikeli(self, x_samples, y_samples, pad_mask):
        preds = self.get_pred(x_samples)
        preds = preds.reshape(-1, preds.size(-1))[pad_mask]
        log_probs = F.log_softmax(preds, dim=-1)

        y_samples_p = y_samples.reshape(-1)[pad_mask]
        positive = log_probs.gather(dim=-1, index=y_samples_p.unsqueeze(-1)).squeeze(-1)
        positive = positive.mean()
        return positive

    def forward(self, x_samples, y_samples, pad_mask):
        preds = self.get_pred(x_samples)
        preds = preds.reshape(-1, preds.size(-1))[pad_mask]
        log_probs = F.log_softmax(preds, dim=-1)

        sample_size = x_samples.shape[0]
        random_index = torch.randperm(sample_size).long()

        y_samples_p = y_samples.reshape(-1)[pad_mask]
        positive = log_probs.gather(dim=-1, index=y_samples_p.unsqueeze(-1)).squeeze(-1)
        positive = positive.mean()

        y_samples_n = y_samples[random_index]
        pad_mask_n = pad_mask.reshape(y_samples_n.size(0), -1)[random_index]
        pad_mask_n = pad_mask_n.reshape(-1)
        y_samples_n = y_samples_n.reshape(-1)[pad_mask_n]
        negative = log_probs.gather(dim=-1, index=y_samples_n.unsqueeze(-1)).squeeze(-1)
        negative = negative.mean()

        upper_bound = positive - negative
        return upper_bound

    def learning_loss(self, x_samples, y_samples, pad_mask):
        return - self.loglikeli(x_samples, y_samples, pad_mask)


class RIB_NASC_v0(nn.Module):
    def __init__(self, num_layers, src_vocab_size, tgt_vocab_size, model_dim, channel_dim, num_heads, dim_feedforward, mine_hidden_dim, club_hidden_dim, club_num_layers,
                 club_num_heads, dropout=0.1):
        super(RIB_NASC_v0, self).__init__()

        self.mine_net = Mine_Net_v0(channel_dim, channel_dim, mine_hidden_dim)

        self.club_net = CLUB_Net_v0(channel_dim, club_hidden_dim, club_num_layers, club_num_heads, tgt_vocab_size)

        self.semantic_encoder = Encoder(num_layers, src_vocab_size, model_dim, num_heads, dim_feedforward, dropout)

        self.jsc_encoder = JSC_Encoder(in_features_dim=model_dim, intermediate_dim=2 * model_dim, out_features_dim=channel_dim)

        self.jsc_decoder = JSC_Decoder(in_features_dim=channel_dim, intermediate_dim=2 * model_dim, out_features_dim=model_dim)

        self.semantic_decoder = DeEncoder(num_layers, model_dim, num_heads, dim_feedforward, dropout)

        self.dense = nn.Linear(model_dim, tgt_vocab_size)
