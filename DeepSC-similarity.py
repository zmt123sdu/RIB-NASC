#!/usr/A/anaconda3/envs/py_3.8 python
# -*- coding: UTF-8 -*-
'''
@Project ：RIB-NASC-latest 
@File    ：DeepSC-similarity.py
@Author  ：Mingtong Zhang
@Date    ：2025/12/1 11:33 
'''
import pickle
import time
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from data.dataset import EurDataset, collater
from models.DeepSC import DeepSC
from utils.utils_general import setup_seed, Timer, Data_savemat, SimScore, SeqtoText
from utils.utils_for_DeepSC import greedy_decode
from config.test_config import get_test_parser

# Some global simulation parameters for debugging
channel = 'Rayleigh_fast_ICSI(LS)'
best_mode = True
train_config_number = '0'
test_config_number = '0'

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

parser = get_test_parser(case='DeepSC', channel=channel, version=None, train_config_number=train_config_number, test_config_number=test_config_number,
                         trained_epoch=None, best_mode=best_mode)

np.set_printoptions(precision=4)
torch.set_printoptions(precision=4)


def performance(args, model, test_dataloader, SNR):
    print('testing on:', device)
    sim_score_cal = SimScore(model_cal_sim)
    S2T = SeqtoText(token_to_idx, end_idx)
    score = []

    model.eval()
    with torch.no_grad():
        for loop in range(args.test_loop):
            Tx_word = []
            Rx_word = []
            timer.start()

            for snr in SNR:
                print(f'-----------SNR = {snr}: 测试开始------------')
                timer_SNR = Timer()
                timer_SNR.start()
                predicted_word = []
                target_word = []

                test_bar = tqdm(test_dataloader, ncols=100)

                for sents_test in test_bar:
                    sents_test: torch.Tensor = sents_test.to(device)  # 将一个batch的测试数据送入CPU or GPU

                    out = greedy_decode(model=model, sent=sents_test, pad=pad_idx, end=end_idx, snr=snr, channel=args.channel, device=device, start_symbol=start_idx,
                                        args=args)
                    sentences = out.tolist()
                    result_string = list(map(S2T.sequence_to_text, sentences))
                    predicted_word = predicted_word + result_string

                    target_sent = sents_test.tolist()
                    result_string = list(map(S2T.sequence_to_text, target_sent))
                    target_word = target_word + result_string

                    test_bar.set_description(f"SNR: {snr} dB")

                Rx_word.append(predicted_word)
                Tx_word.append(target_word)
                print(f"the time for the SNR: {snr} dB is {timer_SNR.time_show(timer_SNR.stop())}")
            print(f"the test for decode has taken {timer.time_show(timer.stop())}")

            sim_score = []
            idx2 = 0
            for sent_Rx, sent_Tx in zip(Rx_word, Tx_word):
                sim_score_snr = []
                for i in tqdm(range(0, len(sent_Rx), args.sim_cal_bs), desc=f"SimCal SNR: {SNR[idx2]} dB"):
                    # 提取当前批次的句子
                    batch_rx = sent_Rx[i: i + args.sim_cal_bs]
                    batch_tx = sent_Tx[i: i + args.sim_cal_bs]
                    sim_score_snr = sim_score_snr + sim_score_cal.compute_sim_score(batch_rx, batch_tx)
                sim_score.append(sim_score_snr)
                print(np.mean(np.array(sim_score_snr)))
                idx2 = idx2 + 1

            score.append(np.mean(np.array(sim_score), axis=1))

            print(f"Test Loop: {loop + 1} is end")
            print(f"the test has taken {timer.time_show(timer.stop())}")

        score = np.mean(np.array(score), axis=0)

        return score


if __name__ == '__main__':
    args = parser.parse_args()
    setup_seed(args.seed)
    SNR = [-3, 0, 3, 6, 9, 12, 15, 18, 21, 24]

    vocab = pickle.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab.token_to_idx
    idx_to_token = dict(zip(token_to_idx.values(), token_to_idx.keys()))
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]

    test_dataset = EurDataset(data_path=args.test_data_path)
    collate_fn = collater(fixed_length_padding=args.fixed_padding, len_max=args.len_max + 2)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_bs, num_workers=0, pin_memory=True, collate_fn=collate_fn, shuffle=True)

    # 在特定时间后开始运行程序
    time_waiting = 0 * 3600  # 等待的秒数
    time.sleep(time_waiting)

    model = DeepSC(num_layers=args.num_layers, src_vocab_size=num_vocab, tgt_vocab_size=num_vocab, model_dim=args.d_model, channel_dim=args.d_channel,
                   num_heads=args.num_heads, dim_feedforward=args.dff, dropout=args.dropout).to(device)

    model_path = args.trained_path
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"loading from {model_path}")

    # 加载计算语义相似度的预训练模型
    pretrained_model_folder = 'pretrained_model/sentence_transformer'
    pretrained_model_path = os.path.join(pretrained_model_folder, args.pretrained_model_name) if args.pretrained_model_name else None
    model_cal_sim = SentenceTransformer(model_name_or_path=pretrained_model_path).to(device)

    timer = Timer()
    sim_score = performance(args, model, test_dataloader, SNR)
    print(sim_score)

    data = Data_savemat()
    for i in range(len(sim_score)):
        data.append(x=SNR[i], y=(sim_score[i]))
    data.savemat(filename=f'results/{args.test_info}-sim-score.mat')
