#!/usr/A/anaconda3/envs/py_3.8 python
# -*- coding: UTF-8 -*-
'''
@Project ：RIB-NASC-latest 
@File    ：RIB-DeepSC-train.py
@Author  ：Mingtong Zhang
@Date    ：2025/12/11 14:15 
'''
import os
import pickle
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import EurDataset, syn_collater
from models.RIB_DeepSC import RIB_DeepSC_v0
from utils.utils_for_RIB_DeepSC import train_step, eval_step, train_mi_step
from utils.utils_general import setup_seed, Timer, TrainingLogger, initNetParams, Data_savemat, optimizer_choice, scheduler_choice
from config.train_config import get_train_parser

# Some global simulation parameters for debugging
version = 'v0'
channel = 'Rayleigh_fast_ICSI(LS)'
config_number = '0'
beta = 0.001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = get_train_parser(case='RIB-DeepSC', channel=channel, version=version, config_number=config_number, beta=beta)
torch.set_printoptions(precision=4)


def train_eval(model, num_epoch, lr, lr_mine, lr_club, optim, init, scheduler_type, eval_interval):
    # 初始化网络参数
    if init:
        initNetParams(model)

    # 将模型发送到对应的设备
    print('training on:', device)
    model.to(device)

    # 优化器选择
    param_backbone = [{'params': model.semantic_encoder.parameters()}, {'params': model.jsc_encoder.parameters()}, {'params': model.jsc_decoder.parameters()},
                      {'params': model.semantic_decoder.parameters()}, {'params': model.dense.parameters()}]
    param_mine = [{'params': model.mine_net.parameters()}]
    param_club = [{'params': model.club_net.parameters()}]
    optimizer = optimizer_choice(optim=optim, param=param_backbone, lr=lr, adam_beats_1=0.9, adam_beats_2=0.999, weight_decay=0.0)
    optimizer_mine = optimizer_choice(optim=optim, param=param_mine, lr=lr_mine, adam_beats_1=0.9, adam_beats_2=0.999, weight_decay=0.0)
    optimizer_club = optimizer_choice(optim=optim, param=param_club, lr=lr_club, adam_beats_1=0.9, adam_beats_2=0.999, weight_decay=0.0)
    # 学习率管理
    scheduler = scheduler_choice(scheduler_type=scheduler_type, optimizer=optimizer, decay_step_list=args.decay_step_list, lr_decay_rate=args.lr_decay_rate)

    start_epoch = 1

    logger = TrainingLogger(log_dir=f'./training_logs/{args.train_info}',
                            legend=['train_loss', 'eval_loss', 'train SZ', 'eval SZ', 'train XZ-J', 'eval XZ-J', 'train ZZ-J', 'eval ZZ-J'])
    data = Data_savemat()
    timer.start()
    best_eval_loss = 10.0

    for epoch_index in range(start_epoch, num_epoch + 1):
        print("-----------第 {} 轮训练开始------------".format(epoch_index))
        train_loss, train_s_z, train_loss_mine, train_x_z_j, train_loss_club, train_z_z_j = [], [], [], [], [], []
        timer_epoch = Timer()
        timer_epoch.start()
        train_bar = tqdm(train_dataloader, ncols=200)

        for sents, sents_syn in train_bar:
            sents, sents_syn = sents.to(device), sents_syn.to(device)

            batch_train_loss_mine, batch_train_loss_club = train_mi_step(model=model, sent=sents, pad=pad_idx, end=end_idx, opt_club=optimizer_club, channel=args.channel,
                                                                         opt_mine=optimizer_mine, device=device, args=args)

            batch_train_loss, batch_train_s_z, batch_train_x_z_j, batch_train_z_z_j = train_step(model=model, sent=sents, sent_syn=sents_syn, pad=pad_idx, end=end_idx,
                                                                                                 opt=optimizer, channel=args.channel, device=device, args=args)

            train_bar.set_description(f"Epoch: {epoch_index}; Type: Train; Loss: {batch_train_loss:.5f}, MI_SZ: {batch_train_s_z:.5f}, "
                                      f"Loss_club: {batch_train_loss_club:.5f}, MI_XZ_J: {batch_train_x_z_j:5f},"
                                      f"Loss_mine: {batch_train_loss_mine:.5f}, MI_ZZ_J: {batch_train_z_z_j:5f}")

            train_loss.append(batch_train_loss)
            train_s_z.append(batch_train_s_z)
            train_x_z_j.append(batch_train_x_z_j)
            train_loss_mine.append(batch_train_loss_mine)
            train_z_z_j.append(batch_train_z_z_j)
            train_loss_club.append(batch_train_loss_club)

        if epoch_index % eval_interval == 0:
            eval_loss, eval_s_z, eval_x_z_j, eval_z_z_j = [], [], [], []
            eval_bar = tqdm(eval_dataloader, ncols=200)
            with torch.no_grad():
                for sents_eval, sents_syn in eval_bar:
                    sents_eval, sents_syn = sents_eval.to(device), sents_syn.to(device)  # 将一个batch的训练数据送入CPU or GPU

                    batch_eval_loss, batch_eval_s_z, batch_eval_x_z_j, batch_eval_z_z_j = eval_step(model=model, sent=sents_eval, sent_syn=sents_syn, pad=pad_idx, end=end_idx,
                                                                                                    channel=args.channel, device=device, args=args)

                    eval_bar.set_description(f"Epoch: {epoch_index}; Type: Eval; Loss: {batch_eval_loss:.5f}, MI_SZ: {batch_eval_s_z:.5f},"
                                             f"MI_XZ_J: {batch_eval_x_z_j:.5f}, MI_ZZ_J: {batch_eval_z_z_j:5f}")

                    eval_loss.append(batch_eval_loss)
                    eval_s_z.append(batch_eval_s_z)
                    eval_x_z_j.append(batch_eval_x_z_j)
                    eval_z_z_j.append(batch_eval_z_z_j)

            train_loss_avg, eval_loss_avg = sum(train_loss) / len(train_loss), sum(eval_loss) / len(eval_loss)
            train_s_z_avg, eval_s_z_avg = sum(train_s_z) / len(train_s_z), sum(eval_s_z) / len(eval_s_z)
            train_x_z_j_avg, eval_x_z_j_avg = sum(train_x_z_j) / len(train_x_z_j), sum(eval_x_z_j) / len(eval_x_z_j)
            train_z_z_j_avg, eval_z_z_j_avg = sum(train_z_z_j) / len(train_z_z_j), sum(eval_z_z_j) / len(eval_z_z_j)

            logger.add(epoch_index, [train_loss_avg, eval_loss_avg, train_s_z_avg, eval_s_z_avg, train_x_z_j_avg, eval_x_z_j_avg, train_z_z_j_avg, eval_z_z_j_avg])
            data.append(x=epoch_index, y=(train_loss_avg, eval_loss_avg, train_s_z_avg, eval_s_z_avg, train_x_z_j_avg, eval_x_z_j_avg, train_z_z_j_avg, eval_z_z_j_avg))

            if eval_loss_avg < best_eval_loss:
                os.makedirs(args.best_save_path, exist_ok=True)
                save_file = os.path.join(args.best_save_path, f"epoch_{epoch_index}.pth")
                torch.save({'model_state_dict': model.state_dict(), }, save_file)
                print(f'The best Model is saved at epoch-{epoch_index}')
                best_eval_loss = eval_loss_avg

        scheduler.step()
        print(f"the training has taken {timer.time_show(timer.stop())}, and the learning rate is {scheduler.get_last_lr()[0]}")
        print(f"the time for the Epoch: {epoch_index} is {timer_epoch.time_show(timer_epoch.stop())}")

    logger.plot(plot_save_path=f'{args.train_info}-Loss', x_label='Epoch', y_label='Loss', title='Training Curve', y_scale='linear', legend_idx=[0, 1])
    logger.plot(plot_save_path=f'{args.train_info}-SZ', x_label='Epoch', y_label='MI SZ-LB', title='Training Curve', y_scale='linear', legend_idx=[2, 3])
    logger.plot(plot_save_path=f'{args.train_info}-XZ_J', x_label='Epoch', y_label='MI XZ-UB', title='Training Curve', y_scale='linear', legend_idx=[4, 5])
    logger.plot(plot_save_path=f'{args.train_info}-ZZ_J', x_label='Epoch', y_label='MI ZZ-LB', title='Training Curve', y_scale='linear', legend_idx=[6, 7])
    data.savemat(f'{args.train_info}.mat')


if __name__ == '__main__':
    args = parser.parse_args()
    setup_seed(args.seed)

    """ preparing the dataset """
    vocab = pickle.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab.token_to_idx
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    end_idx = token_to_idx["<END>"]
    # 训练与验证数据
    train_dataset = EurDataset(data_path=args.train_syn_data_path)
    eval_dataset = EurDataset(data_path=args.test_syn_data_path)
    print('训练数据集长度: {}'.format(len(train_dataset)))
    print('验证数据集长度: {}'.format(len(eval_dataset)))
    print(f'语义编码速率为{args.d_channel / 2}')

    # DataLoader分割数据集为一个个batch
    collate_fn = syn_collater(len_max=args.len_max + 2, num_syn_token=args.num_syn_token)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True, collate_fn=collate_fn, shuffle=True, drop_last=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True, collate_fn=collate_fn, shuffle=True, drop_last=False)

    # 在特定时间后开始运行程序
    time_waiting = 0 * 3600  # 等待的秒数
    time.sleep(time_waiting)

    # 创建网络模型
    model = RIB_DeepSC_v0(num_layers=args.num_layers, src_vocab_size=num_vocab, tgt_vocab_size=num_vocab, model_dim=args.d_model, channel_dim=args.d_channel,
                          num_heads=args.num_heads, dim_feedforward=args.dff, dropout=args.dropout, mine_hidden_dim=args.mine_hidden_dim,
                          club_hidden_dim=args.club_hidden_dim, club_num_layers=args.club_num_layers, club_num_heads=args.club_num_heads).to(device)

    print("模型总参数:", sum(p.numel() for p in model.parameters()))

    timer = Timer()

    train_eval(model, num_epoch=args.epochs, lr=args.lr, lr_mine=args.lr_mine, lr_club=args.lr_club, optim=args.optimizer, init=args.init, scheduler_type=args.scheduler_type,
               eval_interval=1)
