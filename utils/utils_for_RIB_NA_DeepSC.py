#!/usr/A/anaconda3/envs/py_3.8 python
# -*- coding: UTF-8 -*-
'''
@Project ：RIB-NASC-latest 
@File    ：utils_for_RIB_NA_DeepSC.py
@Author  ：Mingtong Zhang
@Date    ：2025/12/10 21:46 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class RandSynReplace(nn.Module):
    def __init__(self, p_sent, p_token):
        super().__init__()
        self.p_sent = p_sent
        self.p_token = p_token

    def forward(self, logits, targets_real, targets_syn, tgt_pad_mask, pad_idx):
        batch_size, num_candidates, seq_len = targets_syn.shape
        device = targets_real.device

        # 生成句子级别的替换掩码
        sent_replace_mask = torch.rand(batch_size, device=device) < self.p_sent  # (batch_size,)
        sent_replace_mask_expanded = sent_replace_mask.unsqueeze(1).expand(batch_size, seq_len)  # (batch, seq)

        # 生成词元级别的替换掩码
        token_replace_prob = torch.rand((batch_size, seq_len), device=device) < self.p_token

        # 1. 随机选择候选索引
        replace_idx = torch.randint(0, num_candidates, (batch_size, seq_len), device=device)

        # 2. 收集选中的候选
        selected_candidates = torch.gather(targets_syn, 1, replace_idx.unsqueeze(1).expand(-1, -1, seq_len)).squeeze(1)

        # 3. 处理选中pad的情况
        first_candidate = targets_syn[:, 0, :]  # 获取第一个候选
        is_pad = (selected_candidates == pad_idx)
        corrected_candidates = torch.where(is_pad, first_candidate, selected_candidates)

        # 4. 最终有效性检查
        valid_replace_mask = (corrected_candidates != pad_idx)

        # 综合所有替换条件
        total_replace_mask = (sent_replace_mask_expanded & token_replace_prob & valid_replace_mask)

        # 生成最终目标：在掩码位置使用合成目标，其余使用真实目标
        targets = torch.where(total_replace_mask, corrected_candidates, targets_real)

        # 提取有效目标（非填充位置）
        valid_targets = targets[tgt_pad_mask.bool()]

        # 计算对数概率（高效向量化实现）
        pad_mask = tgt_pad_mask.reshape(-1).bool()
        logits = logits.reshape(-1, logits.size(-1))[pad_mask]
        log_probs = F.log_softmax(logits, dim=-1)

        # 收集对应目标的log概率
        log_probs_t = log_probs.gather(dim=-1, index=valid_targets.unsqueeze(-1)).squeeze(-1)

        # 计算平均负对数似然损失
        return (-log_probs_t).mean()


def train_mi_step(model, sent, pad, end, opt_mine, opt_club, channel, device, args):
    # 训练步骤开始
    model.eval()
    with torch.no_grad():
        # 预处理输入
        condition_tensor = (sent == end).clone().detach()
        sent_no_end = torch.where(condition_tensor, pad, sent)
        src = sent_no_end[:, 1: -1]

        # 生成掩码
        src_key_mask = create_key_masks(src, pad, device=device)
        key_mask = 1 - src_key_mask.transpose(-2, -1)

        # 语义编码（同时获取CLUB需要的src_embedding）
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

        # 屏蔽padding的影响
        rx_sig = rx_sig * key_mask

    pad_mask = key_mask.reshape(-1).bool()
    # ========== MINE训练部分 ==========
    model.mine_net.train()
    opt_mine.zero_grad()
    # 采样并计算互信息下界
    x_sample = jsc_enc_output.detach()
    y_sample = rx_sig.detach()
    mi_z_z_lb = model.mine_net(x_sample, y_sample)
    loss_mine = -mi_z_z_lb
    loss_mine.backward()
    torch.nn.utils.clip_grad_norm_(model.mine_net.parameters(), 10.0)
    opt_mine.step()

    # ========== CLUB训练部分 ==========
    model.club_net.train()
    opt_club.zero_grad()

    # 计算互信息上界
    x_sample = rx_sig.detach()
    y_sample = src.detach()

    loss_club = model.club_net.learning_loss(x_sample, y_sample, pad_mask)
    loss_club.backward()
    torch.nn.utils.clip_grad_norm_(model.club_net.parameters(), 10.0)
    opt_club.step()

    return loss_mine.item(), loss_club.item()


def train_step(model, sent, sent_syn, pad, end, unk, opt, channel, device, args):
    # 训练步骤开始
    model.train()
    model.club_net.eval()
    model.mine_net.eval()
    opt.zero_grad()

    condition_tensor = (sent == end).clone().detach()
    sent_no_end = torch.where(condition_tensor, pad, sent)
    src = sent_no_end[:, 1: -1]
    tgt_real = sent[:, 1:]
    syn_real = sent_syn[:, :, 1:]

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

    # 产生语义译码器的输入序列和key_padding_mask
    tgt_inp = torch.zeros_like(tgt_real) * pad
    indices = torch.arange(tgt_inp.size(1)).expand(tgt_inp.size(0), -1).to(device)
    tgt_inp[indices < len_sent] = unk
    tgt_key_mask = (tgt_inp == pad).float().unsqueeze(1)

    # 语义译码
    dec_output = model.semantic_decoder(tgt_inp, jsc_dec_output, look_ahead_mask=None, tgt_key_mask=tgt_key_mask, memory_key_mask=src_key_mask)
    pred = model.dense(dec_output)

    # 计算Loss
    # 首先计算mi_s_z
    tgt_pad_mask = 1 - (tgt_real == pad).float()
    loss_fn_syn = RandSynReplace(p_sent=args.p_sent, p_token=args.p_token)
    mi_s_z_lb_n = loss_fn_syn(pred, tgt_real, syn_real, tgt_pad_mask, pad)

    pad_mask = key_mask.reshape(-1).bool()
    # 再计算mi_x_z
    x_sample = rx_sig
    y_sample = src
    mi_x_z_ub = model.club_net(x_sample, y_sample, pad_mask)

    # 最后计算mi_z_z
    x_sample = jsc_enc_output.detach()
    y_sample = rx_sig.detach()
    mi_z_z_lb = model.mine_net(x_sample, y_sample)

    loss = mi_s_z_lb_n + args.beta * (mi_x_z_ub - mi_z_z_lb)

    loss.backward()
    opt.step()

    return loss.item(), mi_s_z_lb_n.item(), mi_x_z_ub.item(), mi_z_z_lb.item()


def eval_step(model, sent, sent_syn, pad, end, unk, channel, device, args):
    # 训练步骤开始
    model.eval()

    condition_tensor = (sent == end).clone().detach()
    sent_no_end = torch.where(condition_tensor, pad, sent)
    src = sent_no_end[:, 1: -1]
    tgt_real = sent[:, 1:]
    syn_real = sent_syn[:, :, 1:]

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

    # 产生语义译码器的输入序列和key_padding_mask
    tgt_inp = torch.zeros_like(tgt_real) * pad
    indices = torch.arange(tgt_inp.size(1)).expand(tgt_inp.size(0), -1).to(device)
    tgt_inp[indices < len_sent] = unk
    tgt_key_mask = (tgt_inp == pad).float().unsqueeze(1)

    # 语义译码
    dec_output = model.semantic_decoder(tgt_inp, jsc_dec_output, look_ahead_mask=None, tgt_key_mask=tgt_key_mask, memory_key_mask=src_key_mask)
    pred = model.dense(dec_output)

    # 计算Loss
    # 首先计算mi_s_z
    tgt_pad_mask = 1 - (tgt_real == pad).float()
    loss_fn_syn = RandSynReplace(p_sent=args.p_sent, p_token=args.p_token)
    mi_s_z_lb_n = loss_fn_syn(pred, tgt_real, syn_real, tgt_pad_mask, pad)

    pad_mask = key_mask.reshape(-1).bool()
    # 再计算mi_x_z
    x_sample = rx_sig
    y_sample = src
    mi_x_z_ub = model.club_net(x_sample, y_sample, pad_mask)

    # 最后计算mi_z_z
    x_sample = jsc_enc_output.detach()
    y_sample = rx_sig.detach()
    mi_z_z_lb = model.mine_net(x_sample, y_sample)

    loss = mi_s_z_lb_n + args.beta * (mi_x_z_ub - mi_z_z_lb)

    return loss.item(), mi_s_z_lb_n.item(), mi_x_z_ub.item(), mi_z_z_lb.item()


def na_decode(model, sent, pad, end, unk, snr, channel, device, args):
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
    snr = snr * torch.ones(tx_sig.size(0), 1).to(device)
    channels = Channels(channel_type=channel, power_tx=args.power_tx, SNR=snr, device=device)
    rx_sig, noise_var = channels.passing_channel(tx_sig=tx_sig)

    # 复数符号 -> 实数符号
    rx_sig = complex_to_real_conversion(rx_sig, input_shape)

    # 屏蔽padding位置信道的影响
    rx_sig = rx_sig * key_mask

    # 联合信源信道译码
    jsc_dec_output = model.jsc_decoder(rx_sig)

    # 产生语义译码器的输入序列和key_padding_mask
    tgt_inp = torch.zeros_like(tgt_real) * pad
    indices = torch.arange(tgt_inp.size(1)).expand(tgt_inp.size(0), -1).to(device)
    tgt_inp[indices < len_sent] = unk
    tgt_key_mask = (tgt_inp == pad).float().unsqueeze(1)

    # 语义译码
    dec_output = model.semantic_decoder(tgt_inp, jsc_dec_output, look_ahead_mask=None, tgt_key_mask=tgt_key_mask, memory_key_mask=src_key_mask)
    pred = model.dense(dec_output)
    _, outputs = torch.max(pred, dim=-1)

    return outputs
