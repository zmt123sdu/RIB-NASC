#!/usr/A/anaconda3/envs/py_3.8 python
# -*- coding: UTF-8 -*-
'''
@Project ：RIB-NASC-latest 
@File    ：utils_for_DeepSC.py
@Author  ：Mingtong Zhang
@Date    ：2025/11/28 17:10 
'''
import torch
import torch.nn as nn
from utils.utils_general import Channels


def create_key_masks(src, tgt, padding_idx, device):
    src_key_mask = (src == padding_idx).float().unsqueeze(1)  # [batch, 1, seq_len]
    tgt_key_mask = (tgt == padding_idx).float().unsqueeze(1)  # [batch, 1, seq_len]
    look_ahead_mask = torch.triu(torch.full((tgt.size(1), tgt.size(1)), float('1')), diagonal=1)
    return src_key_mask.to(device), tgt_key_mask.to(device), look_ahead_mask.to(device)


def real_to_complex_conversion(real_input):
    batch_size = real_input.size(0)

    Z = real_input.reshape(batch_size, -1)
    Z_temp = Z.reshape(batch_size, Z.size(-1) // 2, 2)
    complex_output = torch.complex(Z_temp[:, :, 0], Z_temp[:, :, 1])

    return complex_output


def complex_to_real_conversion(complex_input, ori_shape):
    rx_sig_complex = complex_input.reshape(ori_shape[0], ori_shape[1], ori_shape[2] // 2)
    rx_sig_real = rx_sig_complex.real
    rx_sig_imag = rx_sig_complex.imag
    rx_sig = torch.stack((rx_sig_real, rx_sig_imag), dim=-1)
    real_output = rx_sig.reshape(ori_shape[0], ori_shape[1], ori_shape[2])

    return real_output


def PowerNormalize(z, num_symbol, power_tx=1):
    z_norm_squared = torch.sum(z.real ** 2 + z.imag ** 2, dim=1, keepdim=True)
    # 避免除以零
    z_norm_squared = torch.clamp(z_norm_squared, min=1e-12)
    # 计算归一化常数
    normalization_factor = torch.sqrt(num_symbol * power_tx / z_norm_squared)  # 等价于 sqrt(NP) / sqrt(z^H z)
    # 归一化 z
    normalized_z = z * normalization_factor  # 广播机制应用到每个样本

    return normalized_z


def train_step(model, sent, pad, end, opt, channel, device, args):
    # 训练步骤开始
    model.train()
    opt.zero_grad()

    condition_tensor = (sent == end).clone().detach()
    sent_no_end = torch.where(condition_tensor, pad, sent)
    src = sent_no_end[:, 1: -1]
    tgt_inp = sent_no_end[:, :-1]
    tgt_real = sent[:, 1:]

    # 产生掩码
    src_key_mask, tgt_key_mask, look_ahead_mask = create_key_masks(src, tgt_inp, pad, device=device)
    key_mask = 1 - src_key_mask.transpose(-2, -1)

    # 语义编码
    enc_output = model.semantic_encoder(src, src_key_mask=src_key_mask)

    # 联合信源信道编码
    jsc_enc_output = model.jsc_encoder(enc_output)
    jsc_enc_output = jsc_enc_output * key_mask
    input_shape = jsc_enc_output.shape

    # 获得复值语义特征向量
    z_complex = real_to_complex_conversion(jsc_enc_output)

    # 功率归一化
    len_sent = torch.sum(key_mask, dim=1)
    num_symbol = len_sent * (jsc_enc_output.size(-1) // 2)
    tx_sig = PowerNormalize(z_complex, num_symbol, power_tx=args.power_tx)

    # 通过信道
    snr = torch.randint(args.snr_min, args.snr_max + 1, (tx_sig.size(0), 1)).to(device)
    channels = Channels(channel_type=channel, power_tx=args.power_tx, SNR=snr, device=device)
    rx_sig, noise_var = channels.passing_channel(tx_sig=tx_sig)

    # 复数符号 -> 实数符号
    rx_sig = complex_to_real_conversion(rx_sig, input_shape)

    # 屏蔽padding位置信道的影响
    rx_sig = rx_sig * key_mask

    # 联合信源信道译码
    jsc_dec_output = model.jsc_decoder(rx_sig)

    # 语义译码
    dec_output = model.semantic_decoder(tgt_inp, jsc_dec_output, look_ahead_mask=look_ahead_mask, tgt_key_mask=tgt_key_mask, memory_key_mask=src_key_mask)
    pred = model.dense(dec_output)

    # 计算Loss
    pad_mask = (tgt_real != pad).reshape(-1)
    pred_logits = pred.reshape(-1, pred.size(-1))[pad_mask]
    target = tgt_real.reshape(-1)[pad_mask]

    loss_fn = nn.CrossEntropyLoss(ignore_index=pad, reduction='mean')
    loss = loss_fn(pred_logits, target)

    loss.backward()
    opt.step()

    return loss.item()


def eval_step(model, sent, pad, end, channel, device, args):
    # 训练步骤开始
    model.eval()

    condition_tensor = (sent == end).clone().detach()
    sent_no_end = torch.where(condition_tensor, pad, sent)
    src = sent_no_end[:, 1: -1]
    tgt_inp = sent_no_end[:, :-1]
    tgt_real = sent[:, 1:]

    # 产生掩码
    src_key_mask, tgt_key_mask, look_ahead_mask = create_key_masks(src, tgt_inp, pad, device=device)
    key_mask = 1 - src_key_mask.transpose(-2, -1)

    # 语义编码
    enc_output = model.semantic_encoder(src, src_key_mask=src_key_mask)

    # 联合信源信道编码
    jsc_enc_output = model.jsc_encoder(enc_output)
    jsc_enc_output = jsc_enc_output * key_mask
    input_shape = jsc_enc_output.shape

    # 获得复值语义特征向量
    z_complex = real_to_complex_conversion(jsc_enc_output)

    # 功率归一化
    len_sent = torch.sum(key_mask, dim=1)
    num_symbol = len_sent * (jsc_enc_output.size(-1) // 2)
    tx_sig = PowerNormalize(z_complex, num_symbol, power_tx=args.power_tx)

    # 通过信道
    snr = torch.randint(args.snr_min, args.snr_max + 1, (tx_sig.size(0), 1)).to(device)
    channels = Channels(channel_type=channel, power_tx=args.power_tx, SNR=snr, device=device)
    rx_sig, noise_var = channels.passing_channel(tx_sig=tx_sig)

    # 复数符号 -> 实数符号
    rx_sig = complex_to_real_conversion(rx_sig, input_shape)

    # 屏蔽padding位置信道的影响
    rx_sig = rx_sig * key_mask

    # 联合信源信道译码
    jsc_dec_output = model.jsc_decoder(rx_sig)

    # 语义译码
    dec_output = model.semantic_decoder(tgt_inp, jsc_dec_output, look_ahead_mask=look_ahead_mask, tgt_key_mask=tgt_key_mask, memory_key_mask=src_key_mask)
    pred = model.dense(dec_output)

    # 计算Loss
    pad_mask = (tgt_real != pad).reshape(-1)
    pred_logits = pred.reshape(-1, pred.size(-1))[pad_mask]
    target = tgt_real.reshape(-1)[pad_mask]

    loss_fn = nn.CrossEntropyLoss(ignore_index=pad, reduction='mean')
    loss = loss_fn(pred_logits, target)

    return loss.item()


def greedy_decode(model, sent, pad, end, start_symbol, snr, channel, device, args):
    # 训练步骤开始
    model.eval()

    condition_tensor = (sent == end).clone().detach()
    sent_no_end = torch.where(condition_tensor, pad, sent)
    src = sent_no_end[:, 1: -1]
    tgt_inp = sent_no_end[:, :-1]

    # 产生掩码
    src_key_mask, _, _ = create_key_masks(src, tgt_inp, pad, device=device)
    key_mask = 1 - src_key_mask.transpose(-2, -1)

    # 语义编码
    enc_output = model.semantic_encoder(src, src_key_mask=src_key_mask)

    # 联合信源信道编码
    jsc_enc_output = model.jsc_encoder(enc_output)
    jsc_enc_output = jsc_enc_output * key_mask
    input_shape = jsc_enc_output.shape

    # 获得复值语义特征向量
    z_complex = real_to_complex_conversion(jsc_enc_output)

    # 功率归一化
    len_sent = torch.sum(key_mask, dim=1)
    num_symbol = len_sent * (jsc_enc_output.size(-1) // 2)
    tx_sig = PowerNormalize(z_complex, num_symbol, power_tx=args.power_tx)

    # 通过信道
    snr = snr * torch.ones(tx_sig.size(0), 1).to(device)
    channels = Channels(channel_type=channel, power_tx=args.power_tx, SNR=snr, device=device)
    rx_sig, noise_var = channels.passing_channel(tx_sig=tx_sig)

    # 复数符号 -> 实数符号
    rx_sig = complex_to_real_conversion(rx_sig, input_shape)

    # 屏蔽padding位置信道的影响
    rx_sig = rx_sig * key_mask

    # 联合信源信道译码
    jsc_dec_output = model.jsc_decoder(rx_sig)

    # 自回归贪婪译码
    outputs = torch.ones(src.size(0), 1).fill_(start_symbol).type_as(src.data)
    for i in range(args.len_max + 1):
        # create the decode mask
        look_ahead_mask = torch.triu(torch.full((outputs.size(-1), outputs.size(-1)), float('1')), diagonal=1).to(device)
        tgt_key_mask = (outputs == pad).float().unsqueeze(1).to(device)  # [batch, 1, seq_len]
        dec_output = model.semantic_decoder(outputs, jsc_dec_output, look_ahead_mask=look_ahead_mask, tgt_key_mask=tgt_key_mask, memory_key_mask=src_key_mask)
        pred = model.dense(dec_output)
        prob = pred[:, -1:, :]  # (batch_size, 1, vocab_size)
        _, next_word = torch.max(prob, dim=-1)
        outputs = torch.cat([outputs, next_word], dim=1)

    return outputs
