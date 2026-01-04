#!/usr/A/anaconda3/envs/py_3.8 python
# -*- coding: UTF-8 -*-
'''
@Project ：RIB-NASC-v2 
@File    ：syn_data_generate.py
@Author  ：Mingtong Zhang
@Date    ：2025/5/30 14:39 
'''
import pickle
import numpy as np
from tqdm import tqdm
from nltk.corpus import stopwords
from utils.utils_general import setup_seed
from data_config.data_config import get_data_parser

data_config_number = '0'


class SynonymProcessor:
    def __init__(self, syn_vocab, num_syn_token, syn_th, pad_idx, stop_idxs):
        self.syn_vocab = syn_vocab
        self.num_syn_token = num_syn_token
        self.syn_th = syn_th
        self.pad_idx = pad_idx
        self.stop_idxs = stop_idxs

    def process_sentence(self, sent_ori):
        syn_sents = [sent_ori]
        for i in range(1, self.num_syn_token):
            syn_sent = np.full_like(sent_ori, self.pad_idx)
            for j, token in enumerate(sent_ori):
                if token not in self.stop_idxs:
                    synonym_info = self.syn_vocab.get(token)
                    if synonym_info and len(synonym_info) > i:
                        synonym_index, synonym_score = synonym_info[i]
                        if synonym_score >= self.syn_th:
                            syn_sent[j] = synonym_index
            syn_sents.append(syn_sent.tolist())
        return syn_sents


def load_dataset(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def main():
    parser = get_data_parser(data_config_number)
    args = parser.parse_args()
    setup_seed(args.seed)

    # Load vocabularies
    with open(args.syn_vocab_file, 'rb') as f:
        syn_vocab = pickle.load(f)
    with open(args.vocab_file, 'rb') as f:
        vocab = pickle.load(f)

    token_to_idx = vocab.token_to_idx
    pad_idx = token_to_idx["<PAD>"]

    # Load datasets
    train_dataset = load_dataset(args.train_data_path)
    test_dataset = load_dataset(args.test_data_path)

    # Process stop words
    stop_idxs = {token_to_idx.get(word) for word in stopwords.words('english')} if args.stop_words == 'nltk_en' else set()

    # Initialize processor
    processor = SynonymProcessor(syn_vocab, args.num_syn_token, args.syn_th, pad_idx, stop_idxs)

    # Process datasets
    train_syn_datas, test_syn_datas = [], []
    for sent in tqdm(train_dataset, desc="Processing train sentences", total=len(train_dataset)):
        syn_sents = processor.process_sentence(sent)
        train_syn_datas.append(syn_sents)

    for sent in tqdm(test_dataset, desc="Processing test sentences", total=len(test_dataset)):
        syn_sents = processor.process_sentence(sent)
        test_syn_datas.append(syn_sents)

    # Save results
    output_path = f'../data/europarl/{{}}({data_config_number}).pkl'
    with open(output_path.format('train_syn_data_config'), 'wb') as f:
        pickle.dump(train_syn_datas, f)
    with open(output_path.format('test_syn_data_config'), 'wb') as f:
        pickle.dump(test_syn_datas, f)


if __name__ == '__main__':
    main()
