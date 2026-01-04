#!/usr/A/anaconda3/envs/py_3.8 python
# -*- coding: UTF-8 -*-
'''
@Project ：RIB-NASC-latest 
@File    ：utils_general.py
@Author  ：Mingtong Zhang
@Date    ：2025/11/27 19:58 
'''
import collections
import math
import time
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from matplotlib import pyplot as plt
from datetime import datetime
import json
from scipy import io
from w3lib.html import remove_tags


def initNetParams(model):
    '''Init net parameters.'''
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_normal_(p)
    return model


def SNR_to_noise(snr, power_tx=1):
    snr = 10 ** (snr / 10)
    noise_var = power_tx / snr

    return noise_var


def optimizer_choice(optim, param, lr, adam_beats_1=0.9, adam_beats_2=0.999, weight_decay=0.0):
    if optim == 'sgd':
        optimizer = torch.optim.SGD(param, lr=lr)
    elif optim == 'adam':
        optimizer = torch.optim.Adam(param, lr=lr, betas=(adam_beats_1, adam_beats_2), weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(param, lr=lr)

    return optimizer


def scheduler_choice(scheduler_type, optimizer, decay_step_list, lr_decay_rate, step_size=10):
    if scheduler_type == 'MultiStep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decay_step_list, gamma=lr_decay_rate)
    elif scheduler_type == 'Exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay_rate)
    elif scheduler_type == 'Step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=lr_decay_rate)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=1.0)

    return scheduler


def setup_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def count_corpus(tokens):
    """Count token frequencies."""
    # Here `tokens` is a 1D list or 2D list
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # Flatten a list of token lists into a list of tokens
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


def bleu(pred_seq, label_seq, k=4, weights=(1, 0, 0, 0)):  # @save
    """计算BLEU"""
    pred_tokens, label_tokens = pred_seq, label_seq
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / max(1, len_pred)))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[''.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[''.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[''.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / max(1, (len_pred - n + 1)), weights[n - 1])
    return score


def load_config(filename='config.txt'):
    config = {}
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            # 跳过注释行
            if line.startswith("#") or not line:
                continue
            key, value = line.split(' = ')
            # 检查是否是数值列表
            if ',' in value:
                items = value.split(',')
                try:
                    # 尝试将每个元素转换为浮点数
                    value = [float(v.strip()) for v in items]
                    if all(val.is_integer() for val in value):  # 如果全是整数，转换为整数列表
                        value = list(map(int, value))
                except ValueError:
                    # 如果转换失败，则视为字符串列表
                    value = [v.strip() for v in items]
            elif value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                value = int(value)
            elif value.replace('.', '', 1).isdigit() or (value.startswith('-') and value[1:].replace('.', '', 1).isdigit()):
                value = float(value)
            elif value == 'True':
                value = True
            elif value == 'False':
                value = False
            elif value == 'None':
                value = None  # 将字符串 'None' 转换为 None
            config[key] = value
    return config


def get_required_epoch(path, order=-1):
    idx_list = []
    for fn in os.listdir(path):
        if not fn.endswith('.pth'): continue
        idx = int(os.path.splitext(fn)[0].split('_')[-1])  # read the idx of image
        idx_list.append((os.path.join(path, fn), idx))

    idx_list.sort(key=lambda x: x[1])  # sort the image by the idx
    _, epoch = idx_list[order]

    return epoch


class Vocab:
    """Vocabulary for text."""

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # Sort according to frequencies
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # 将预定义的特殊字符加入字典
        self.token_to_idx = reserved_tokens
        self.idx_to_token = [token for idx, token in enumerate(self.token_to_idx)]
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # Index for the unknown token
        return 3

    @property
    def token_freqs(self):  # Index for the unknown token
        return self._token_freqs


class Timer:
    """Record multiple running times."""

    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()

    def time_show(self, time_sec, mode=1):
        if mode == 1:
            if time_sec > 3600:
                hour = int(time_sec / 3600)
                min = int((time_sec - 3600 * hour) / 60)
                sec = time_sec - 3600 * hour - 60 * min
                time_description = f'{hour} hour {min} min {sec:.2f} sec'
            elif time_sec > 60:
                min = int(time_sec / 60)
                sec = time_sec - 60 * min
                time_description = f'{min} min {sec:.2f} sec'
            else:
                sec = time_sec
                time_description = f'{sec:.2f} sec'
        else:
            time_description = f'{time_sec,:.2f} sec'
        return time_description


class TrainingLogger:
    """训练数据记录器，训练完成后绘制图表"""

    def __init__(self, log_dir="training_logs", legend=None):
        if legend is None:
            legend = ['train_loss', 'eval_loss']

        self.legend = legend
        self.log_dir = log_dir

        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)

        # 生成唯一的时间戳，避免覆盖之前的日志
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(log_dir, f"training_log_{self.timestamp}.csv")
        self.config_path = os.path.join(log_dir, f"training_config_{self.timestamp}.json")

        # 初始化数据存储
        self.data = {label: [] for label in legend}
        self.epochs = []

        # 记录配置
        self.config = {
            "legend": legend,
            "timestamp": self.timestamp,
            "start_time": datetime.now().isoformat()
        }

        # 创建CSV文件并写入标题
        with open(self.csv_path, 'w') as f:
            f.write('epoch,' + ','.join(legend) + '\n')

        print(f"训练日志将保存到: {self.csv_path}")

    def add(self, epoch, values):
        """添加一个epoch的数据点"""
        if not hasattr(values, "__len__"):
            values = [values]

        self.epochs.append(epoch)

        # 记录数据
        for i, label in enumerate(self.legend):
            if i < len(values):
                self.data[label].append(values[i])
            else:
                self.data[label].append(None)  # 如果没有对应值，记录为None

        # 将数据追加到CSV文件
        with open(self.csv_path, 'a') as f:
            f.write(f"{epoch}," + ','.join(str(values[i]) if i < len(values) else '' for i in range(len(self.legend))) + '\n')

        # 打印当前进度
        value_str = ", ".join(f"{self.legend[i]}={values[i]:.6f}" for i in range(len(values)))
        print(f"Epoch {epoch}: {value_str}")

    def plot(self, plot_save_path, x_label='Epoch', y_label='Loss', title='Training Curve', x_scale='linear', y_scale='log', fig_size=(7, 5), save_plot=True, legend_idx=None):
        """训练完成后绘制图表"""
        # 创建图表
        plt.figure(figsize=fig_size)

        # 绘制每条曲线
        colors = ['blue', 'red', 'green', 'cyan', 'magenta', 'orange', 'purple', 'brown']
        linestyles = ['-', '--', '-.', ':']

        # 选择画图的数据
        legend = self.legend if legend_idx is None else [self.legend[i] for i in legend_idx]

        for i, label in enumerate(legend):
            if self.data[label] and any(v is not None for v in self.data[label]):
                color = colors[i % len(colors)]
                linestyle = linestyles[i % len(linestyles)]

                # 过滤掉None值
                x_vals = [self.epochs[j] for j in range(len(self.epochs))
                          if j < len(self.data[label]) and self.data[label][j] is not None]
                y_vals = [v for v in self.data[label] if v is not None]

                if x_vals and y_vals and len(x_vals) == len(y_vals):
                    if y_scale == 'log':
                        plt.semilogy(x_vals, y_vals, color=color, linestyle=linestyle,
                                     label=label, linewidth=2)
                    else:
                        plt.plot(x_vals, y_vals, color=color, linestyle=linestyle,
                                 label=label, linewidth=2)

        # 设置图表属性
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 设置坐标轴尺度
        if x_scale == 'log':
            plt.xscale('log')
        if y_scale == 'log':
            plt.yscale('log')

        plt.tight_layout()

        # 保存图表
        if save_plot:
            plot_path = os.path.join(self.log_dir, f"{plot_save_path}_{self.timestamp}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"训练曲线已保存为: {plot_path}")

        # 显示图表
        plt.show()

        return plt.gcf()

    def save_config(self, additional_config=None):
        """保存训练配置"""
        if additional_config:
            self.config.update(additional_config)

        self.config['end_time'] = datetime.now().isoformat()

        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

        print(f"训练配置已保存到: {self.config_path}")


class Data_savemat:
    """将数据保存为mat格式"""

    def __init__(self):
        self.x_data = []
        self.y_data = []
        self.legend = []

    def append(self, x=1, y=1, legend=None):
        self.x_data.append(x)
        self.y_data.append(y)
        if legend:
            self.legend.append(legend)
            self.x_data.pop()
            self.y_data.pop()

    def savemat(self, filename):
        io.savemat(filename, {'x_data': self.x_data, 'y_data': self.y_data, 'legend': self.legend})


class Channels:
    def __init__(self, channel_type, power_tx, SNR, device):
        self.channel_type = channel_type
        self.power_tx = power_tx
        self.device = device
        self.SNR = SNR

    def _generate_noise(self, signal_shape, device):
        noise_var = SNR_to_noise(self.SNR, power_tx=self.power_tx)
        std_per_dim = torch.sqrt(noise_var / 2)
        std_per_dim = std_per_dim.unsqueeze(-1) if len(signal_shape) == 3 else std_per_dim
        noise_real = torch.randn(signal_shape, device=device) * std_per_dim
        noise_imag = torch.randn(signal_shape, device=device) * std_per_dim
        return torch.complex(noise_real, noise_imag), noise_var

    def _estimate_channel(self, h_true, csi_err_type, slow=False):
        if csi_err_type == 'LS_CE':
            noise_var = SNR_to_noise(self.SNR, power_tx=self.power_tx)
            csi_err_var = noise_var / self.power_tx
            csi_err_std_per_dim = torch.sqrt(csi_err_var / 2)

            if slow:
                batch_size = h_true.shape[0]
                err_shape = (batch_size, 1) if h_true.ndim == 2 else (batch_size, 1, 1)
                csi_err_real = torch.randn(err_shape, device=self.device) * csi_err_std_per_dim
                csi_err_imag = torch.randn(err_shape, device=self.device) * csi_err_std_per_dim
            else:
                csi_err_real = torch.randn_like(h_true.real) * csi_err_std_per_dim
                csi_err_imag = torch.randn_like(h_true.imag) * csi_err_std_per_dim

            csi_err = torch.complex(csi_err_real, csi_err_imag)
            h_est = h_true + csi_err
            return h_est, csi_err

        return h_true, torch.zeros_like(h_true)

    def _apply_channel_effect(self, tx_sig, h, h_est=None):
        noise, noise_var = self._generate_noise(tx_sig.shape, self.device)
        rx_sig = tx_sig * h + noise
        rx_sig = rx_sig / h_est
        return rx_sig, noise_var

    def AWGN(self, tx_sig):
        noise, noise_var = self._generate_noise(tx_sig.shape, self.device)
        rx_sig = tx_sig + noise

        return rx_sig, noise_var

    def Rayleigh(self, tx_sig, slow=False, csi_err_type=None):
        # 信道系数生成
        if slow:
            batch_size = tx_sig.shape[0]
            h_real = torch.randn(batch_size, 1, device=self.device) * math.sqrt(0.5)
            h_imag = torch.randn(batch_size, 1, device=self.device) * math.sqrt(0.5)
            h = torch.complex(h_real, h_imag).repeat(1, *tx_sig.shape[1:])
        else:
            shape = tx_sig.shape
            h_real = torch.randn(shape, device=self.device) * math.sqrt(0.5)
            h_imag = torch.randn(shape, device=self.device) * math.sqrt(0.5)
            h = torch.complex(h_real, h_imag)

        # 信道估计
        h_est, _ = self._estimate_channel(h, csi_err_type, slow)

        # 应用信道
        return self._apply_channel_effect(tx_sig, h, h_est)

    def Rician(self, tx_sig, K, slow=False, theta=None, csi_err_type=None):
        # 生成直射路径分量（LOS component）
        los_magnitude = math.sqrt(K / (K + 1))  # 直射路径幅度
        if theta is None:
            theta = torch.full((tx_sig.shape[0], 1), math.pi / 4, device=self.device)
        else:
            theta = torch.rand(tx_sig.real.size(0), 1, device=self.device) * 2 * math.pi  # 随机相位

        los_real = los_magnitude * torch.cos(theta)  # 直射路径实部
        los_imag = los_magnitude * torch.sin(theta)  # 直射路径虚部

        # 散射路径分量
        scatter_std = math.sqrt(1 / (2 * (K + 1)))

        if slow:
            batch_size = tx_sig.shape[0]
            scatter_real = torch.randn(batch_size, 1, device=self.device) * scatter_std
            scatter_imag = torch.randn(batch_size, 1, device=self.device) * scatter_std
            h = torch.complex(los_real + scatter_real, los_imag + scatter_imag)
            h = h.repeat(1, *tx_sig.shape[1:])
        else:
            scatter_real = torch.randn_like(tx_sig.real) * scatter_std
            scatter_imag = torch.randn_like(tx_sig.imag) * scatter_std
            h = torch.complex(los_real + scatter_real, los_imag + scatter_imag)

        # 信道估计
        h_est, _ = self._estimate_channel(h, csi_err_type, slow)

        # 应用信道
        return self._apply_channel_effect(tx_sig, h, h_est)

    def passing_channel(self, tx_sig):
        if self.channel_type == 'AWGN':
            rx, noise_var = self.AWGN(tx_sig)
        elif self.channel_type == 'Rayleigh_fast':
            rx, noise_var = self.Rayleigh(tx_sig)
        elif self.channel_type == 'Rayleigh_fast_ICSI(LS)':
            rx, noise_var = self.Rayleigh(tx_sig, csi_err_type='LS_CE')
        elif self.channel_type == 'Rayleigh_slow':
            rx, noise_var = self.Rayleigh(tx_sig, slow=True)
        elif self.channel_type == 'Rayleigh_slow_ICSI(LS)':
            rx, noise_var = self.Rayleigh(tx_sig, slow=True, csi_err_type='LS_CE')
        else:
            rx, noise_var = None, None

        return rx, noise_var


class BleuScore():
    def __init__(self, w1, w2, w3, w4):
        self.w1 = w1  # 1-gram weights
        self.w2 = w2  # 2-grams weights
        self.w3 = w3  # 3-grams weights
        self.w4 = w4  # 4-grams weights

    def compute_blue_score(self, predicted, real):
        score = []
        for (sent1, sent2) in zip(predicted, real):
            sent1 = remove_tags(sent1).split()
            sent2 = remove_tags(sent2).split()
            score.append(bleu(sent1, sent2, weights=(self.w1, self.w2, self.w3, self.w4)))
        return score


class SeqtoText:
    def __init__(self, vocb_dictionary, end_idx):
        self.reverse_word_map = dict(zip(vocb_dictionary.values(), vocb_dictionary.keys()))
        self.end_idx = end_idx

    def sequence_to_text(self, list_of_indices):
        # Looking up words in dictionary
        words = []
        for idx in list_of_indices:
            if idx == self.end_idx:
                break
            else:
                words.append(self.reverse_word_map.get(idx))
        words = ' '.join(words)
        return (words)


def cosine_similarity(tensor1, tensor2, dim=1):
    """计算cosine_similarity"""
    if not isinstance(tensor1, torch.Tensor):
        tensor1 = torch.tensor(tensor1)

    if not isinstance(tensor2, torch.Tensor):
        tensor2 = torch.tensor(tensor2)

    if len(tensor1.shape) == 1:
        tensor1 = tensor1.unsqueeze(0)

    if len(tensor2.shape) == 1:
        tensor2 = tensor2.unsqueeze(0)

    cos_similarities = F.cosine_similarity(tensor1, tensor2, dim=dim)

    return cos_similarities


class SimScore():
    def __init__(self, model_cal_sim):
        self.model_cal_sim = model_cal_sim

    def compute_sim_score(self, predicted, real):
        sent1_list, sent2_list = [], []
        for (sent1, sent2) in zip(predicted, real):
            sent1 = remove_tags(sent1)
            sent2 = remove_tags(sent2)

            sent1_list.append(sent1)
            sent2_list.append(sent2)

        sent1_embeddings = self.model_cal_sim.encode(sent1_list)
        sent2_embeddings = self.model_cal_sim.encode(sent2_list)
        score = cosine_similarity(sent1_embeddings, sent2_embeddings)
        return score.tolist()
