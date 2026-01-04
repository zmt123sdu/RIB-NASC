#!/usr/A/anaconda3/envs/py_3.8 python
# -*- coding: UTF-8 -*-
'''
@Project ：RIB-NASC-v2 
@File    ：syn_vocab_generate.py
@Author  ：Mingtong Zhang
@Date    ：2025/5/29 16:47 
'''
import torch
import pickle
from tqdm import tqdm
import numpy as np
from transformers import BertTokenizer, RobertaTokenizer
from models.Pretrain_model import get_bert
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from nltk.corpus import wordnet as wn


@dataclass
class SynonymResult:
    token_vocab: Dict[int, List[Tuple[str, float]]]
    idx_vocab: Dict[int, List[Tuple[int, float]]]


class SynonymGenerator:
    def __init__(self, device: str = "cuda:1" if torch.cuda.is_available() else "cpu"):
        self.device = device
        # # 下载 WordNet 数据（如果未下载）
        # try:
        #     nltk.data.find('corpora/wordnet')
        # except LookupError:
        #     print("Downloading WordNet data...")
        #     nltk.download('wordnet', quiet=True)
        #     nltk.download('omw-eng', quiet=True)

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms > 0, norms, 1)

    def _process_common_tokens(self, glove_embeddings: Dict[str, np.ndarray],
                               idx_to_token_raw: Dict[int, str]) -> Optional[Tuple[List[str], List[np.ndarray], List[int]]]:
        common_tokens = []
        common_vectors = []
        common_indices = []

        for idx, token in idx_to_token_raw.items():
            if token in glove_embeddings:
                common_tokens.append(token)
                common_vectors.append(glove_embeddings[token])
                common_indices.append(idx)

        return common_tokens, common_vectors, common_indices

    def _process_similarities(self, sim_matrix: np.ndarray, common_tokens: List[str],
                              common_indices: List[int], top_k: int) -> SynonymResult:
        token_vocab = {}
        idx_vocab = {}

        for i, (token, idx) in tqdm(enumerate(zip(common_tokens, common_indices)), desc="Processing tokens", total=len(common_tokens)):
            sim_scores = sim_matrix[i]
            top_k_indices = np.argsort(sim_scores)[-top_k - 1:][::-1]

            token_synonyms = [(common_tokens[j], float(sim_scores[j])) for j in top_k_indices]
            idx_synonyms = [(common_indices[j], float(sim_scores[j])) for j in top_k_indices]

            token_vocab[idx] = token_synonyms
            idx_vocab[idx] = idx_synonyms

        return SynonymResult(token_vocab, idx_vocab)

    def _get_valid_tokens(self, idx_to_token_raw: Dict[int, str],
                          vocab_bert: Dict[str, int]) -> List[Tuple[int, str, int]]:
        valid_tokens = []
        for idx, token in idx_to_token_raw.items():
            if token in vocab_bert:
                valid_tokens.append((idx, token, vocab_bert[token]))
        return valid_tokens

    def _load_glove_embeddings(self, glove_path: str) -> Dict[str, np.ndarray]:
        embeddings = {}
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading GloVe"):
                values = line.split()
                embeddings[values[0]] = np.asarray(values[1:], dtype='float32')
        return embeddings

    def _compute_bert_similarities(self, embedding_layer: torch.nn.Module,
                                   valid_tokens_data: List[Tuple[int, str, int]]):
        embeddings = embedding_layer.weight.detach()
        valid_indices = torch.tensor([x[2] for x in valid_tokens_data]).to(self.device)
        valid_embeddings = embeddings[valid_indices]

        normalized_embeddings = F.normalize(valid_embeddings, p=2, dim=1)
        similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())

        return similarity_matrix.cpu().numpy()

    def _process_bert_similarities(self, similarity_matrix: np.ndarray, valid_tokens_data: List[Tuple[int, str, int]], top_k: int) -> SynonymResult:
        token_vocab = {}
        idx_vocab = {}

        for i, (idx, token, _) in tqdm(enumerate(valid_tokens_data), desc="Processing tokens", total=len(valid_tokens_data)):
            sim_scores = similarity_matrix[i]
            top_k_indices = np.argsort(sim_scores)[-top_k - 1:][::-1]

            token_synonyms = [(valid_tokens_data[j][1], float(sim_scores[j])) for j in top_k_indices]
            idx_synonyms = [(valid_tokens_data[j][0], float(sim_scores[j])) for j in top_k_indices]

            token_vocab[idx] = token_synonyms
            idx_vocab[idx] = idx_synonyms

        return SynonymResult(token_vocab, idx_vocab)

    def get_glove_synonyms(self, glove_path: str, idx_to_token_raw: Dict[int, str], top_k: int) -> SynonymResult:
        # Load GloVe embeddings
        glove_embeddings = self._load_glove_embeddings(glove_path)

        # Process common tokens
        common_data = self._process_common_tokens(glove_embeddings, idx_to_token_raw)

        common_tokens, common_vectors, common_indices = common_data

        # Calculate similarities
        norm_vectors = self._normalize_vectors(np.array(common_vectors))
        sim_matrix = np.dot(norm_vectors, norm_vectors.T)

        return self._process_similarities(sim_matrix, common_tokens, common_indices, top_k)

    def get_bert_synonyms(self, bert_name: str, bert_path: str, idx_to_token_raw: Dict[int, str], top_k: int) -> SynonymResult:
        # Initialize BERT/RoBERTa
        tokenizer = self._get_tokenizer(bert_name, bert_path)
        bert_model = get_bert(bert_name, bert_path).to(self.device)
        embedding_layer = bert_model.word_embeddings.eval()

        # Get embeddings
        vocab_bert = tokenizer.get_vocab()
        valid_tokens_data = self._get_valid_tokens(idx_to_token_raw, vocab_bert)
        similarity_matrix = self._compute_bert_similarities(embedding_layer, valid_tokens_data)

        return self._process_bert_similarities(similarity_matrix, valid_tokens_data, top_k)

    def _get_tokenizer(self, bert_name: str, bert_path: str):
        if 'roberta-base' in bert_name:
            return RobertaTokenizer.from_pretrained(bert_path, do_lower_case=True)
        return BertTokenizer.from_pretrained(bert_path, do_lower_case=True)

    def get_wordnet_synonyms(self, idx_to_token_raw: Dict[int, str], top_k: int = 10, similarity_threshold: float = 0.0, strategy: str = 'similarity',
                             pos_filter: Optional[List[str]] = None, use_cache: bool = True) -> SynonymResult:

        print(f"使用WordNet生成同义词表 (策略: {strategy}, 阈值: {similarity_threshold})")

        if strategy == 'similarity':
            return self._get_wordnet_similarity_synonyms(idx_to_token_raw, top_k, similarity_threshold, pos_filter, use_cache)
        else:
            raise ValueError(f"未知策略: {strategy}")

    def _get_wordnet_similarity_synonyms(self, idx_to_token_raw: Dict[int, str], top_k: int, similarity_threshold: float, pos_filter: Optional[List[str]],
                                         use_cache: bool) -> SynonymResult:

        token_vocab = {}
        idx_vocab = {}

        # 创建单词到索引的映射
        # idx_to_token_raw = dict(list(idx_to_token_raw.items())[:50])
        token_to_idx = {token: idx for idx, token in idx_to_token_raw.items()}
        all_tokens = list(idx_to_token_raw.values())

        # 计算所有单词的最佳同义词集
        word_synsets = {}
        for token in tqdm(all_tokens, desc="获取同义词集"):
            synsets = wn.synsets(token, pos=pos_filter) if pos_filter else wn.synsets(token)
            if synsets:
                # 选择最佳同义词集（使用频率最高）
                best_syn = self._get_best_synset(synsets)
                word_synsets[token] = best_syn

        # 计算相似度矩阵
        print("计算WordNet相似度矩阵...")
        similarity_matrix = self._compute_wordnet_similarity_matrix(all_tokens, word_synsets, use_cache)

        # 处理每个单词
        for i, (idx, token) in tqdm(enumerate(idx_to_token_raw.items()), desc="生成同义词表", total=len(idx_to_token_raw)):
            if i >= len(similarity_matrix):
                continue

            sim_scores = similarity_matrix[i]

            # 获取相似度最高的同义词
            valid_indices = []
            for j, score in enumerate(sim_scores):
                if j != i and score >= similarity_threshold and all_tokens[j] in token_to_idx:
                    valid_indices.append((j, score))

            # 按相似度排序
            valid_indices.sort(key=lambda x: x[1], reverse=True)
            valid_indices = valid_indices[:top_k]

            # 构建同义词列表
            token_synonyms = []
            idx_synonyms = []

            for j, score in valid_indices:
                synonym_token = all_tokens[j]
                synonym_idx = token_to_idx[synonym_token]
                token_synonyms.append((synonym_token, float(score)))
                idx_synonyms.append((synonym_idx, float(score)))

            if token_synonyms:
                token_vocab[idx] = token_synonyms
                idx_vocab[idx] = idx_synonyms

        return SynonymResult(token_vocab, idx_vocab)

    def _get_best_synset(self, synsets: List) -> 'wn.synset':
        """选择最佳同义词集（基于使用频率）"""
        # 统计同义词集中所有词元的总频率
        best_syn = None
        max_freq = -1

        for syn in synsets:
            total_freq = sum(lemma.count() for lemma in syn.lemmas())
            if total_freq > max_freq:
                max_freq = total_freq
                best_syn = syn

        return best_syn

    def _compute_wordnet_similarity_matrix(self, tokens: List[str],
                                           word_synsets: Dict[str, 'wn.synset'],
                                           use_cache: bool) -> np.ndarray:
        """
        计算WordNet相似度矩阵
        """
        n = len(tokens)
        similarity_matrix = np.zeros((n, n))

        # 缓存相似度计算结果
        cache = {}

        for i in tqdm(range(n), desc="计算相似度"):
            syn_i = word_synsets.get(tokens[i])
            if not syn_i:
                continue

            for j in range(i, n):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                    continue

                syn_j = word_synsets.get(tokens[j])
                if not syn_j:
                    continue

                # 检查缓存
                cache_key = (tokens[i], tokens[j])
                if use_cache and cache_key in cache:
                    sim = cache[cache_key]
                else:
                    # 计算Wu-Palmer相似度
                    sim = self._calculate_synset_similarity(syn_i, syn_j)
                    if use_cache:
                        cache[cache_key] = sim

                similarity_matrix[i][j] = sim
                similarity_matrix[j][i] = sim

        return similarity_matrix

    def _calculate_synset_similarity(self, syn1: 'wn.synset', syn2: 'wn.synset') -> float:
        """计算两个同义词集之间的相似度"""
        if syn1 is None or syn2 is None:
            return 0.0

        # 尝试不同的相似度计算方法
        similarities = []

        # 1. Wu-Palmer相似度
        wup_sim = syn1.wup_similarity(syn2)
        if wup_sim:
            similarities.append(wup_sim)

        # 2. 路径相似度
        path_sim = syn1.path_similarity(syn2)
        if path_sim:
            similarities.append(path_sim)

        # 3. Leacock-Chodorow相似度（需要相同词性）
        if syn1.pos() == syn2.pos():
            try:
                lch_sim = syn1.lch_similarity(syn2)
                if lch_sim:
                    # 将LCH相似度归一化到[0,1]范围
                    lch_sim_normalized = min(1.0, lch_sim / 3.5)  # 经验值
                    similarities.append(lch_sim_normalized)
            except:
                pass

        if not similarities:
            return 0.0

        # 返回平均相似度
        return sum(similarities) / len(similarities)


def main():
    generator = SynonymGenerator()

    # Load vocabulary
    with open('../data/europarl/vocab.pkl', 'rb') as f:
        raw_vocab = pickle.load(f)

    idx_to_token_raw = {idx: token for token, idx in raw_vocab.token_to_idx.items()}
    print(f'Raw vocab size: {len(idx_to_token_raw)}')

    # Configuration
    config = {
        'bert_name': 'bert-base-uncased',
        'bert_path': '../pretrained_model/bert-base-uncased',
        'glove_path': '../pretrained_model/glove.6B/glove.6B.300d.txt',
        'top_k': 10
    }

    # Add synonyms_mode parameter
    synonyms_mode = 'wordnet'  # 'bert' or 'glove' or 'wordnet'

    # Generate synonyms
    if synonyms_mode == 'glove':
        result = generator.get_glove_synonyms(config['glove_path'], idx_to_token_raw, config['top_k'])
        output_prefix = '../data/europarl/glove_6B_300d'

    elif synonyms_mode == 'bert':
        result = generator.get_bert_synonyms(config['bert_name'], config['bert_path'], idx_to_token_raw, config['top_k'])
        output_prefix = f'../data/europarl/{config["bert_name"]}(word_embedding_only)'

    elif synonyms_mode == 'wordnet':
        # 使用WordNet生成同义词
        result = generator.get_wordnet_synonyms(idx_to_token_raw=idx_to_token_raw, top_k=config['top_k'], similarity_threshold=0.3)
        output_prefix = f'../data/europarl/wordnet'
    else:
        raise ValueError(f"未知的模式: {synonyms_mode}")

    # Save results
    for suffix, data in [('syn_token_vocab.pkl', result.token_vocab),
                         ('syn_idx_vocab.pkl', result.idx_vocab)]:
        with open(f'{output_prefix}_{suffix}', 'wb') as f:
            pickle.dump(data, f)


if __name__ == '__main__':
    main()
