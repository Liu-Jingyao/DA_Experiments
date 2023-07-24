import torch
from torch.nn.modules.dropout import _DropoutNd
import collections
import copy

import math
import numpy as np
from tqdm import tqdm


class TFIDFPreProcess:
    def __init__(self, dataset, vocab_size=4000, p=0.1, all_special_ids=[], dropout_constant=0, device='cpu'):
        self.device = device
        self.dropout_constant = dropout_constant
        self.vocab_size = vocab_size
        self.p = p
        self.idf_dict = TFIDFPreProcess.get_idf_tfidf(dataset['input_ids'])['idf']
        self.idf = torch.zeros(vocab_size, device=self.device)
        self.all_special_ids = all_special_ids
        for k, v in self.idf_dict.items():
            self.idf[int(k)] += v

    @staticmethod
    def dropout_prob_batch_preprocess(batch, tfidf_preprocess, **kwargs):
        dropout_prob = []
        while len(tqdm._instances) > 0:
            tqdm._instances.pop().close()
        for text in tqdm(batch['input_ids'], desc="Calculating dataset dropout prob"):
            dropout_prob += [tfidf_preprocess.get_text_dropout_prob(text)]
        batch['dropout_prob'] = dropout_prob
        return batch

    @staticmethod
    def replacement_prob_batch_preprocess(batch, tfidf_preprocess, **kwargs):
        replacement_prob = []
        while len(tqdm._instances) > 0:
            tqdm._instances.pop().close()
        for text in tqdm(batch['input_ids'], desc="Calculating dataset dropout prob"):
            replacement_prob += [tfidf_preprocess.get_text_replacement_prob(text)]
        batch['replacement_prob'] = replacement_prob
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

    def get_text_dropout_prob(self, text):
        """Compute the probability of replacing tokens in a sentence."""
        cur_tf_idf = collections.defaultdict(int)
        for word in text:
            cur_tf_idf[word] += 1. / len(text) * self.idf[word]
        dropout_prob = []
        for word in text:
            dropout_prob += [cur_tf_idf[word]]

        dropout_prob = np.array(dropout_prob)
        dropout_prob = np.max(dropout_prob) - dropout_prob

        for i, word in enumerate(text):
            if word in self.all_special_ids:
                dropout_prob[i] = 0

        if dropout_prob.sum() != 0:
            dropout_prob = dropout_prob * self.p * len(text) / dropout_prob.sum()
            np.clip(dropout_prob, 0, 1, out=dropout_prob)

        return dropout_prob.tolist()

    def get_text_replacement_prob(self, text):
        """Compute the probability of replacing tokens in a sentence."""
        cur_tf_idf = collections.defaultdict(int)
        for word in text:
            cur_tf_idf[word] += 1. / len(text) * self.idf[word]
        replacement_prob = []
        for word in text:
            replacement_prob += [cur_tf_idf[word]]

        replacement_prob = np.array(replacement_prob)

        for i, word in enumerate(text):
            if word in self.all_special_ids:
                replacement_prob[i] = 0

        if replacement_prob.sum() != 0:
            replacement_prob = replacement_prob * self.p * len(text) / replacement_prob.sum()
            np.clip(replacement_prob, 0, 1, out=replacement_prob)

        return replacement_prob.tolist()

def remove_elements_by_keep(a, keep):
    # create boolean mask
    keep[:, -1] = 0
    mask = (keep != 0)
    max_len = len(keep[0])

    # select non-zero elements from a
    a_list = []
    for i in range(a.size(0)):
        new_a_i = a[i][mask[i]]
        new_a_i = torch.nn.functional.pad(new_a_i, (0, max_len - len(new_a_i) - 1), mode='constant', value=0)
        new_a_i = torch.cat((new_a_i, a[i, -1].unsqueeze(0)))
        a_list.append(new_a_i)
    a = torch.stack(a_list)

    return a

def random_word_dropout(input_ids, attention_mask=None, token_type_ids=None, aug_prob=None, **kwargs):
    keep = torch.empty_like(input_ids).bernoulli(1 - aug_prob).bool()
    input_ids = remove_elements_by_keep(input_ids, keep)
    attention_mask = remove_elements_by_keep(attention_mask, keep)
    if token_type_ids is not None:
        token_type_ids = remove_elements_by_keep(token_type_ids, keep)

    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}

def tfidf_word_dropout(input_ids, attention_mask=None, token_type_ids=None, dropout_prob=None, **kwargs):
    keep = torch.bernoulli(1 - dropout_prob).bool()
    input_ids = remove_elements_by_keep(input_ids, keep)
    attention_mask = remove_elements_by_keep(attention_mask, keep)
    if token_type_ids is not None:
        token_type_ids = remove_elements_by_keep(token_type_ids, keep)

    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids, 'keep': keep}