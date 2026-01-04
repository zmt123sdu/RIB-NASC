#!/usr/A/anaconda3/envs/py_3.8 python
# -*- coding: UTF-8 -*-
'''
@Project ：RIB-NASC-latest 
@File    ：test_config.py
@Author  ：Mingtong Zhang
@Date    ：2025/11/28 16:32 
'''
import argparse
from utils.utils_general import get_required_epoch, load_config


def get_test_parser(case, channel, version, train_config_number, test_config_number, trained_epoch=200, best_mode=False, beta=0.001):
    config_file_path = f'./config/test_config_{test_config_number}.txt'
    config_dict = load_config(config_file_path)

    parser = get_mian_parser(channel=channel, test_loop=config_dict['test_loop'], test_bs=config_dict['test_batch_size'], seed=config_dict['seed'],
                             fixed_padding=config_dict['fixed_padding'], dropout=config_dict['dropout'], d_channel=config_dict['d_channel'],
                             d_model=config_dict['d_model'], dff=config_dict['dff'], num_heads=config_dict['num_heads'], num_layers=config_dict['num_layers'],
                             len_max=config_dict['len_max'], pretrained_model_name=config_dict['pretrained_model_name'], sim_cal_bs=config_dict['sim_cal_bs'],
                             power_tx=config_dict['power_tx'])

    if case == 'NASC':
        train_info = f'{case}_{channel}_train-config({train_config_number})'
        trained_fold = f'./checkpoints/{train_info}'
        if best_mode:
            trained_fold = f'./checkpoints/{train_info}_best'
            trained_epoch = get_required_epoch(trained_fold)
        trained_path = f'{trained_fold}/epoch_{trained_epoch}.pth'
        test_info = train_info + f'_test-config({test_config_number})_trained-epoch({trained_epoch})'
        parser = get_parser_set1(parser, train_info=train_info, test_info=test_info, trained_path=trained_path, true_condition=config_dict.get('true_condition'))

    elif case == 'DeepSC':
        train_info = f'{case}_{channel}_train-config({train_config_number})'
        trained_fold = f'./checkpoints/{train_info}'
        if best_mode:
            trained_fold = f'./checkpoints/{train_info}_best'
            trained_epoch = get_required_epoch(trained_fold)
        trained_path = f'{trained_fold}/epoch_{trained_epoch}.pth'
        test_info = train_info + f'_test-config({test_config_number})_trained-epoch({trained_epoch})'
        parser = get_parser_set1(parser, train_info=train_info, test_info=test_info, trained_path=trained_path, true_condition=config_dict.get('true_condition'))

    elif case == 'NA-DeepSC':
        train_info = f'{case}_{channel}_train-config({train_config_number})'
        trained_fold = f'./checkpoints/{train_info}'
        if best_mode:
            trained_fold = f'./checkpoints/{train_info}_best'
            trained_epoch = get_required_epoch(trained_fold)
        trained_path = f'{trained_fold}/epoch_{trained_epoch}.pth'
        test_info = train_info + f'_test-config({test_config_number})_trained-epoch({trained_epoch})'
        parser = get_parser_set1(parser, train_info=train_info, test_info=test_info, trained_path=trained_path, true_condition=config_dict.get('true_condition'))

    elif case == 'RIB-NASC':
        train_info = f'{case}_{version}_{channel}_{beta}_train-config({train_config_number})'
        trained_fold = f'./checkpoints/{train_info}'
        if best_mode:
            trained_fold = f'./checkpoints/{train_info}_best'
            trained_epoch = get_required_epoch(trained_fold)
        trained_path = f'{trained_fold}/epoch_{trained_epoch}.pth'
        test_info = train_info + f'_test-config({test_config_number})_trained-epoch({trained_epoch})'
        parser = get_parser_set2(parser, version=version, train_info=train_info, test_info=test_info, trained_path=trained_path, mine_hidden_dim=config_dict['mine_hidden_dim'],
                                 club_hidden_dim=config_dict['club_hidden_dim'], club_num_layers=config_dict.get('club_num_layers'),
                                 club_num_heads=config_dict.get('club_num_heads'))

    elif case == 'RIB-DeepSC':
        train_info = f'{case}_{version}_{channel}_{beta}_train-config({train_config_number})'
        trained_fold = f'./checkpoints/{train_info}'
        if best_mode:
            trained_fold = f'./checkpoints/{train_info}_best'
            trained_epoch = get_required_epoch(trained_fold)
        trained_path = f'{trained_fold}/epoch_{trained_epoch}.pth'
        test_info = train_info + f'_test-config({test_config_number})_trained-epoch({trained_epoch})'
        parser = get_parser_set2(parser, version=version, train_info=train_info, test_info=test_info, trained_path=trained_path, mine_hidden_dim=config_dict['mine_hidden_dim'],
                                 club_hidden_dim=config_dict['club_hidden_dim'], club_num_layers=config_dict.get('club_num_layers'),
                                 club_num_heads=config_dict.get('club_num_heads'))

    elif case == 'RIB-NA-DeepSC':
        train_info = f'{case}_{version}_{channel}_{beta}_train-config({train_config_number})'
        trained_fold = f'./checkpoints/{train_info}'
        if best_mode:
            trained_fold = f'./checkpoints/{train_info}_best'
            trained_epoch = get_required_epoch(trained_fold)
        trained_path = f'{trained_fold}/epoch_{trained_epoch}.pth'
        test_info = train_info + f'_test-config({test_config_number})_trained-epoch({trained_epoch})'
        parser = get_parser_set2(parser, version=version, train_info=train_info, test_info=test_info, trained_path=trained_path, mine_hidden_dim=config_dict['mine_hidden_dim'],
                                 club_hidden_dim=config_dict['club_hidden_dim'], club_num_layers=config_dict.get('club_num_layers'),
                                 club_num_heads=config_dict.get('club_num_heads'))

    return parser


def get_parser_set1(parser, train_info, test_info, trained_path, true_condition):
    parser.add_argument('--trained-path', default=trained_path, type=str)
    parser.add_argument('--train-info', default=train_info, type=str)
    parser.add_argument('--test-info', default=test_info, type=str)
    parser.add_argument('--true_condition', default=true_condition, type=bool)

    return parser


def get_parser_set2(parser, version, train_info, test_info, trained_path, mine_hidden_dim, club_hidden_dim, club_num_layers, club_num_heads):
    parser.add_argument('--trained-path', default=trained_path, type=str)
    parser.add_argument('--version', default=version, type=str)
    parser.add_argument('--train-info', default=train_info, type=str)
    parser.add_argument('--test-info', default=test_info, type=str)
    parser.add_argument('--mine_hidden_dim', default=mine_hidden_dim, type=int)
    parser.add_argument('--club_hidden_dim', default=club_hidden_dim, type=int)
    parser.add_argument('--club_num_layers', default=club_num_layers, type=int)
    parser.add_argument('--club_num_heads', default=club_num_heads, type=int)

    return parser


def get_mian_parser(channel, d_channel, fixed_padding, d_model, dff, num_layers, num_heads, dropout, len_max, seed, test_loop, test_bs, pretrained_model_name,
                    sim_cal_bs, power_tx):
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
    parser.add_argument('--power_tx', default=power_tx, type=float)
    # Dataset parameters
    parser.add_argument('--len-max', default=len_max, type=int)
    parser.add_argument('--fixed-padding', default=fixed_padding, type=bool)
    # Training set parameters
    parser.add_argument('--seed', default=seed, type=int)
    parser.add_argument('--test-loop', default=test_loop, type=int)
    parser.add_argument('--test-bs', default=test_bs, type=int)
    parser.add_argument('--sim_cal_bs', default=sim_cal_bs, type=int)
    parser.add_argument('--pretrained_model_name', default=pretrained_model_name, type=str)

    return parser
