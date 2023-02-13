import torch
from torch.nn.modules.dropout import _DropoutNd
import collections
import copy

import math
import numpy as np
from tqdm import tqdm


class TFIDFPreProcess:
    def __init__(self, dataset, vocab_size=4000, p=0.1, dropout_constant=0, device='cpu'):
        self.device = device
        self.dropout_constant = dropout_constant
        self.vocab_size = vocab_size
        self.p = p
        self.idf_dict = TFIDFPreProcess.get_idf_tfidf(dataset['input_ids'])['idf']
        self.idf = torch.zeros(vocab_size, device=self.device)
        for k, v in self.idf_dict.items():
            self.idf[int(k)] += v

    @staticmethod
    def batch_preprocess(batch, tfidf_preprocess, **kwargs):
        dropout_prob = []
        for text in tqdm(batch['input_ids'], desc="Calculating dataset dropout prob"):
            dropout_prob += [tfidf_preprocess.get_text_dropout_prob(text)]
        batch['dropout_prob'] = dropout_prob
        return batch

    @staticmethod
    def get_idf_tfidf(input_ids_corpus):
        """Compute the IDF score for each word. Then compute the TF-IDF score."""
        word_doc_freq = collections.defaultdict(int)
        # Compute IDF
        for i in tqdm(range(len(input_ids_corpus)), desc="Compute the IDF score for each word in dataset"):
            cur_word_dict = {}
            cur_sent = copy.deepcopy(input_ids_corpus[i])
            for word in cur_sent:
                cur_word_dict[word] = 1
            for word in cur_word_dict:
                word_doc_freq[word] += 1
        idf = {}
        for word in word_doc_freq:
            idf[word] = math.log(len(input_ids_corpus) * 1. / word_doc_freq[word])
        # Compute TF-IDF
        tf_idf = {}
        for i in tqdm(range(len(input_ids_corpus)), desc="Compute the TF-IDF score for each word in dataset"):
            cur_sent = copy.deepcopy(input_ids_corpus[i])
            for word in cur_sent:
                if word not in tf_idf:
                    tf_idf[word] = 0
                tf_idf[word] += 1. / len(cur_sent) * idf[word]
        return {
            "idf": idf,
            "tf_idf": tf_idf,
        }

    def get_batch_prob_tensor(self, batch):
        dropout_prob = torch.empty_like(batch, dtype=torch.float, device=self.device)
        for i in range(len(batch)):
            dropout_prob[i] = self.get_text_dropout_prob(batch[i])
        return dropout_prob

    def get_batch_prob(self, texts):
        dropout_prob = []
        for text in tqdm(texts, desc="Calculating dataset dropout prob"):
            dropout_prob += [self.get_text_dropout_prob(text)]
        return dropout_prob

    def get_text_dropout_prob_tensor(self, text):
        """Compute the probability of replacing tokens in a sentence."""
        cur_tf_idf = torch.zeros(self.vocab_size, device=self.device)
        for word in text:
            cur_tf_idf[word] += 1. / len(text) * self.idf[word]
        dropout_prob = torch.empty_like(text, dtype=torch.float, device=self.device)
        for i in range(len(text)):
            dropout_prob[i] = cur_tf_idf[text[i]]
        dropout_prob = torch.max(dropout_prob) - dropout_prob
        dropout_prob = dropout_prob * self.p * len(text) / dropout_prob.sum()
        return dropout_prob

    def get_text_dropout_prob(self, text):
        """Compute the probability of replacing tokens in a sentence."""
        cur_tf_idf = collections.defaultdict(int)
        for word in text:
            cur_tf_idf[word] += 1. / len(text) * self.idf[word]
        replace_prob = []
        for word in text:
            replace_prob += [cur_tf_idf[word]]
        replace_prob = np.array(replace_prob)
        replace_prob = np.max(replace_prob) - replace_prob
        replace_prob = replace_prob * self.p * len(text) / replace_prob.sum()
        return replace_prob.tolist()