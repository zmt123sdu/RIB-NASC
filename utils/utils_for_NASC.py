#!/usr/A/anaconda3/envs/py_3.8 python
# -*- coding: UTF-8 -*-
'''
@Project ：RIB-NASC-latest 
@File    ：utils_for_NASC.py
@Author  ：Mingtong Zhang
@Date    ：2025/11/29 11:07 
'''
import torch
import torch.nn as nn
from utils.utils_general import Channels


def create_key_masks(src, padding_idx, device):
    src_key_mask = (src == padding_idx).float().unsqueeze(1)  # [batch, 1, seq_len]
    return src_key_mask.to(device)


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
    tgt_real = sent[:, 1:]

    # 产生掩码
    src_key_mask = create_key_masks(src, pad, device=device)
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

    # 填充一个空token, 以保证接收端恢复出的句子的总长度为L+1, +1是为了恢复出[EOS] token
    # [B, L, D] -> [B, L+1, D]
    pad_token = torch.zeros_like(jsc_dec_output[:, 0, :]).unsqueeze(1)
    jsc_dec_output = torch.cat((jsc_dec_output, pad_token), dim=1)
    # 构建接收端key_mask,规避填充空token的影响
    # [B, L, D] -> [B, L+1, D]
    one_mask = torch.ones_like(src_key_mask[:, :, 0]).unsqueeze(-1)
    tgt_key_mask = torch.cat((src_key_mask, one_mask), dim=-1)

    # 语义译码
    dec_output = model.semantic_decoder(jsc_dec_output, src_key_mask=tgt_key_mask)
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
    tgt_real = sent[:, 1:]

    # 产生掩码
    src_key_mask = create_key_masks(src, pad, device=device)
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

    # 填充一个空token, 以保证接收端恢复出的句子的总长度为L+1, +1是为了恢复出[EOS] token
    # [B, L, D] -> [B, L+1, D]
    pad_token = torch.zeros_like(jsc_dec_output[:, 0, :]).unsqueeze(1)
    jsc_dec_output = torch.cat((jsc_dec_output, pad_token), dim=1)
    # 构建接收端key_mask,规避填充空token的影响
    # [B, L, D] -> [B, L+1, D]
    one_mask = torch.ones_like(src_key_mask[:, :, 0]).unsqueeze(-1)
    tgt_key_mask = torch.cat((src_key_mask, one_mask), dim=-1)

    # 语义译码
    dec_output = model.semantic_decoder(jsc_dec_output, src_key_mask=tgt_key_mask)
    pred = model.dense(dec_output)

    # 计算Loss
    pad_mask = (tgt_real != pad).reshape(-1)
    pred_logits = pred.reshape(-1, pred.size(-1))[pad_mask]
    target = tgt_real.reshape(-1)[pad_mask]

    loss_fn = nn.CrossEntropyLoss(ignore_index=pad, reduction='mean')
    loss = loss_fn(pred_logits, target)

    return loss.item()


def na_decode(model, sent, pad, end, snr, channel, device, args):
    # 训练步骤开始
    model.eval()

    condition_tensor = (sent == end).clone().detach()
    sent_no_end = torch.where(condition_tensor, pad, sent)
    src = sent_no_end[:, 1: -1]

    # 产生掩码
    src_key_mask = create_key_masks(src, pad, device=device)
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

    # 填充一个空token, 以保证接收端恢复出的句子的总长度为L+1, +1是为了恢复出[EOS] token
    # [B, L, D] -> [B, L+1, D]
    pad_token = torch.zeros_like(jsc_dec_output[:, 0, :]).unsqueeze(1)
    jsc_dec_output = torch.cat((jsc_dec_output, pad_token), dim=1)
    # 构建接收端key_mask,规避填充空token的影响
    # [B, L, D] -> [B, L+1, D]
    one_mask = torch.ones_like(src_key_mask[:, :, 0]).unsqueeze(-1)
    tgt_key_mask = torch.cat((src_key_mask, one_mask), dim=-1)

    # 语义译码
    dec_output = model.semantic_decoder(jsc_dec_output, src_key_mask=tgt_key_mask)
    pred = model.dense(dec_output)
    _, outputs = torch.max(pred, dim=-1)

    return outputs
