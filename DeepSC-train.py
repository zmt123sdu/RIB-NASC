#!/usr/A/anaconda3/envs/py_3.8 python
# -*- coding: UTF-8 -*-
'''
@Project ：RIB-NASC-latest 
@File    ：DeepSC-train.py
@Author  ：Mingtong Zhang
@Date    ：2025/11/28 11:35 
'''
import os
import pickle
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import EurDataset, collater
from models.DeepSC import DeepSC
from utils.utils_for_DeepSC import train_step, eval_step
from utils.utils_general import setup_seed, Timer, TrainingLogger, initNetParams, Data_savemat, optimizer_choice, scheduler_choice
from config.train_config import get_train_parser

# Some global simulation parameters for debugging
channel = 'Rayleigh_fast_ICSI(LS)'
config_number = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = get_train_parser(case='DeepSC', channel=channel, version=None, config_number=config_number)
torch.set_printoptions(precision=4)


def train_eval(model, num_epoch, lr, optim, init, scheduler_type, eval_interval):
    # 初始化网络参数
    if init:
        initNetParams(model)

    # 将模型发送到对应的设备
    print('training on:', device)
    model.to(device)

    # 优化器选择
    param = (param for param in model.parameters() if param.requires_grad)
    optimizer = optimizer_choice(optim=optim, param=param, lr=lr, adam_beats_1=args.adam_beats_1, adam_beats_2=args.adam_beats_2, weight_decay=args.weight_decay)
    # 学习率管理
    scheduler = scheduler_choice(scheduler_type=scheduler_type, optimizer=optimizer, decay_step_list=args.decay_step_list, lr_decay_rate=args.lr_decay_rate)

    start_epoch = 1

    logger = TrainingLogger(log_dir=f'./training_logs/{args.train_info}', legend=['train_loss', 'eval_loss'])
    data = Data_savemat()
    timer.start()
    best_eval_loss = 10.0

    for epoch_index in range(start_epoch, num_epoch + 1):
        print("-----------第 {} 轮训练开始------------".format(epoch_index))
        train_loss = []
        timer_epoch = Timer()
        timer_epoch.start()
        train_bar = tqdm(train_dataloader, ncols=100)

        for sents in train_bar:
            sents = sents.to(device)

            batch_loss_train = train_step(model=model, sent=sents, pad=pad_idx, end=end_idx, opt=optimizer, channel=args.channel, device=device, args=args)
            train_bar.set_description(f"Epoch: {epoch_index}; Type: Train; Loss: {batch_loss_train:.5f}")

            train_loss.append(batch_loss_train)

        if epoch_index % eval_interval == 0:
            eval_loss = []
            eval_bar = tqdm(eval_dataloader, ncols=100)
            with torch.no_grad():
                for sents_eval in eval_bar:
                    sents_eval = sents_eval.to(device)  # 将一个batch的训练数据送入CPU or GPU

                    batch_loss_eval = eval_step(model=model, sent=sents_eval, pad=pad_idx, end=end_idx, channel=args.channel, device=device, args=args)
                    eval_loss.append(batch_loss_eval)

                    eval_bar.set_description(f"Epoch: {epoch_index}; Type: Eval; Loss: {batch_loss_eval:.5f}")

            train_loss_avg = sum(train_loss) / len(train_loss)
            eval_loss_avg = sum(eval_loss) / len(eval_loss)
            logger.add(epoch_index, [train_loss_avg, eval_loss_avg])
            data.append(x=epoch_index, y=(train_loss_avg, eval_loss_avg))

            if eval_loss_avg < best_eval_loss:
                os.makedirs(args.best_save_path, exist_ok=True)
                save_file = os.path.join(args.best_save_path, f"epoch_{epoch_index}.pth")
                torch.save({'model_state_dict': model.state_dict(), }, save_file)
                print(f'The best Model is saved at epoch-{epoch_index}')
                best_eval_loss = eval_loss_avg

        scheduler.step()
        print(f"the training has taken {timer.time_show(timer.stop())}, and the learning rate is {scheduler.get_last_lr()[0]}")
        print(f"the time for the Epoch: {epoch_index} is {timer_epoch.time_show(timer_epoch.stop())}")

    logger.plot(plot_save_path=f'{args.train_info}', x_label='Epoch', y_label='Loss', title='Training Curve', y_scale='linear')
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
    train_dataset = EurDataset(data_path=args.train_data_path)
    eval_dataset = EurDataset(data_path=args.test_data_path)
    print('训练数据集长度: {}'.format(len(train_dataset)))
    print('验证数据集长度: {}'.format(len(eval_dataset)))
    print(f'语义编码速率为{args.d_channel / 2}')

    # DataLoader分割数据集为一个个batch
    collate_fn = collater(fixed_length_padding=args.fixed_padding, len_max=args.len_max + 2)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True, collate_fn=collate_fn, shuffle=True, drop_last=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True, collate_fn=collate_fn, shuffle=True, drop_last=False)

    # 在特定时间后开始运行程序
    time_waiting = 0 * 3600  # 等待的秒数
    time.sleep(time_waiting)

    # 创建网络模型
    model = DeepSC(num_layers=args.num_layers, src_vocab_size=num_vocab, tgt_vocab_size=num_vocab, model_dim=args.d_model, channel_dim=args.d_channel,
                   num_heads=args.num_heads, dim_feedforward=args.dff, dropout=args.dropout).to(device)

    print("模型总参数:", sum(p.numel() for p in model.parameters()))

    timer = Timer()

    train_eval(model, num_epoch=args.epochs, lr=args.lr, optim=args.optimizer, init=args.init, scheduler_type=args.scheduler_type, eval_interval=1)
