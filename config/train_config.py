#!/usr/A/anaconda3/envs/py_3.8 python
# -*- coding: UTF-8 -*-
'''
@Project ：RIB-NASC-latest 
@File    ：train_config.py
@Author  ：Mingtong Zhang
@Date    ：2025/11/27 19:52 
'''
import argparse
from utils.utils_general import load_config


def get_train_parser(case, channel, version, config_number, beta=0.001):
    config_file_path = f'./config/train_config_{config_number}.txt'
    config_dict = load_config(config_file_path)
    data_config_num = str(config_dict['data_config_num'])

    parser = get_mian_parser(channel=channel, snr_min=config_dict['snr_min'], snr_max=config_dict['snr_max'], fixed_padding=config_dict['fixed_padding'], seed=config_dict['seed'],
                             lr_initial=config_dict['lr_initial'], d_channel=config_dict['d_channel'], epochs=config_dict['epochs'], batch_size=config_dict['batch_size'],
                             scheduler_type=config_dict['lr_scheduler_type'], decay_step_list=config_dict['decay_step_list'], lr_decay_rate=config_dict['lr_decay_rate'],
                             optimizer=config_dict['optimizer'], len_max=config_dict['len_max'], d_model=config_dict['d_model'], dff=config_dict['dff'],
                             num_layers=config_dict['num_layers'], num_heads=config_dict['num_heads'], init=config_dict['init'], dropout=config_dict['dropout'],
                             power_tx=config_dict['power_tx'], weight_decay=config_dict.get('weight_decay'), adam_beats_1=config_dict.get('adam_beats_1'),
                             adam_beats_2=config_dict.get('adam_beats_2'))

    if case == 'NASC':
        train_info = f'{case}_{channel}_train-config({config_number})'
        parser = get_parser_set1(parser, version=version, train_info=train_info)

    elif case == 'DeepSC':
        train_info = f'{case}_{channel}_train-config({config_number})'
        parser = get_parser_set1(parser, version=version, train_info=train_info)

    elif case == 'NA-DeepSC':
        train_info = f'{case}_{channel}_train-config({config_number})'
        parser = get_parser_set1(parser, version=version, train_info=train_info)

    elif case == 'RIB-NASC':
        train_info = f'{case}_{version}_{channel}_{beta}_train-config({config_number})'
        parser = get_parser_set2(parser, version=version, train_info=train_info, data_config_num=data_config_num, num_syn_token=config_dict.get('num_syn_token'),
                                 p_token=config_dict['p_token'], lr_mine=config_dict['lr_mine'], lr_club=config_dict['lr_club'], mine_hidden_dim=config_dict['mine_hidden_dim'],
                                 club_hidden_dim=config_dict['club_hidden_dim'], p_sent=config_dict.get('p_sent'), club_num_layers=config_dict.get('club_num_layers'),
                                 club_num_heads=config_dict.get('club_num_heads'), beta=beta, num_sampling=config_dict.get('num_sampling'), tau=config_dict.get('tau'),
                                 sampling_method=config_dict.get('sampling_method'))

    elif case == 'RIB-NA-DeepSC':
        train_info = f'{case}_{version}_{channel}_{beta}_train-config({config_number})'
        parser = get_parser_set2(parser, version=version, train_info=train_info, data_config_num=data_config_num, num_syn_token=config_dict.get('num_syn_token'),
                                 p_token=config_dict['p_token'], lr_mine=config_dict['lr_mine'], lr_club=config_dict['lr_club'], mine_hidden_dim=config_dict['mine_hidden_dim'],
                                 club_hidden_dim=config_dict['club_hidden_dim'], p_sent=config_dict.get('p_sent'), club_num_layers=config_dict.get('club_num_layers'),
                                 club_num_heads=config_dict.get('club_num_heads'), beta=beta, num_sampling=config_dict.get('num_sampling'), tau=config_dict.get('tau'),
                                 sampling_method=config_dict.get('sampling_method'))

    elif case == 'RIB-DeepSC':
        train_info = f'{case}_{version}_{channel}_{beta}_train-config({config_number})'
        parser = get_parser_set2(parser, version=version, train_info=train_info, data_config_num=data_config_num, num_syn_token=config_dict.get('num_syn_token'),
                                 p_token=config_dict['p_token'], lr_mine=config_dict['lr_mine'], lr_club=config_dict['lr_club'], mine_hidden_dim=config_dict['mine_hidden_dim'],
                                 club_hidden_dim=config_dict['club_hidden_dim'], p_sent=config_dict.get('p_sent'), club_num_layers=config_dict.get('club_num_layers'),
                                 club_num_heads=config_dict.get('club_num_heads'), beta=beta, num_sampling=config_dict.get('num_sampling'), tau=config_dict.get('tau'),
                                 sampling_method=config_dict.get('sampling_method'))

    return parser


def get_parser_set1(parser, version, train_info):
    parser.add_argument('--checkpoint-path', default=f'./checkpoints/{train_info}', type=str)
    parser.add_argument('--best-save-path', default=f'./checkpoints/{train_info}_best', type=str)
    parser.add_argument('--version', default=version, type=str)
    parser.add_argument('--train-info', default=train_info, type=str)

    return parser


def get_parser_set2(parser, version, train_info, data_config_num, num_syn_token, p_token, p_sent, lr_mine, lr_club, mine_hidden_dim, club_hidden_dim, club_num_layers,
                    club_num_heads, beta, num_sampling, sampling_method, tau):
    parser.add_argument('--train-syn-data-path', default=f'./data/europarl/train_syn_data_config({data_config_num}).pkl', type=str)
    parser.add_argument('--test-syn-data-path', default=f'./data/europarl/test_syn_data_config({data_config_num}).pkl', type=str)
    parser.add_argument('--checkpoint-path', default=f'./checkpoints/{train_info}', type=str)
    parser.add_argument('--best-save-path', default=f'./checkpoints/{train_info}_best', type=str)
    parser.add_argument('--version', default=version, type=str)
    parser.add_argument('--train-info', default=train_info, type=str)
    parser.add_argument('--num_syn_token', default=num_syn_token, type=int)
    parser.add_argument('--beta', default=beta, type=float)
    parser.add_argument('--p_sent', default=p_sent, type=float)
    parser.add_argument('--p_token', default=p_token, type=float)
    parser.add_argument('--lr_mine', default=lr_mine, type=float)
    parser.add_argument('--lr_club', default=lr_club, type=float)
    parser.add_argument('--mine_hidden_dim', default=mine_hidden_dim, type=int)
    parser.add_argument('--club_hidden_dim', default=club_hidden_dim, type=int)
    parser.add_argument('--club_num_layers', default=club_num_layers, type=int)
    parser.add_argument('--club_num_heads', default=club_num_heads, type=int)
    parser.add_argument('--num_sampling', default=num_sampling, type=int)
    parser.add_argument('--sampling_method', default=sampling_method, type=str)
    parser.add_argument('--tau', default=tau, type=float)

    return parser


def get_mian_parser(channel, d_channel, snr_min, snr_max, fixed_padding, lr_initial, epochs, seed, init, batch_size, scheduler_type,
                    decay_step_list, lr_decay_rate, optimizer, len_max, d_model, dff, num_layers, num_heads, dropout, power_tx, weight_decay, adam_beats_1, adam_beats_2):
    parser = argparse.ArgumentParser()
    # File path parameters
    parser.add_argument('--train-data-path', default='./data/europarl/train_data.pkl', type=str)
    parser.add_argument('--test-data-path', default='./data/europarl/test_data.pkl', type=str)
    parser.add_argument('--vocab-file', default='./data/europarl/vocab.pkl', type=str)
    # Model basic parameters
    parser.add_argument('--d-model', default=d_model, type=int)
    parser.add_argument('--d-channel', default=d_channel, type=int)
    parser.add_argument('--dff', default=dff, type=int)
    parser.add_argument('--num-layers', default=num_layers, type=int)
    parser.add_argument('--num-heads', default=num_heads, type=int)
    parser.add_argument('--dropout', default=dropout, type=float)
    # Channel simulation parameters
    parser.add_argument('--channel', default=channel, type=str)
    parser.add_argument('--snr-min', default=snr_min, type=float)
    parser.add_argument('--snr-max', default=snr_max, type=float)
    parser.add_argument('--power_tx', default=power_tx, type=float)
    # Dataset parameters
    parser.add_argument('--len-max', default=len_max, type=int)
    parser.add_argument('--fixed-padding', default=fixed_padding, type=bool)
    # Training set parameters
    parser.add_argument('--seed', default=seed, type=int)
    parser.add_argument('--init', default=init, type=bool)
    parser.add_argument('--batch-size', default=batch_size, type=int)
    parser.add_argument('--lr', default=lr_initial, type=float)
    parser.add_argument('--scheduler_type', default=scheduler_type, type=str)
    parser.add_argument('--decay_step_list', default=decay_step_list, nargs='+')
    parser.add_argument('--lr_decay_rate', default=lr_decay_rate, type=float)
    parser.add_argument('--optimizer', default=optimizer, type=str)
    parser.add_argument('--weight_decay', default=weight_decay, type=float)
    parser.add_argument('--adam_beats_1', default=adam_beats_1, type=float)
    parser.add_argument('--adam_beats_2', default=adam_beats_2, type=float)
    parser.add_argument('--epochs', default=epochs, type=int)

    return parser
