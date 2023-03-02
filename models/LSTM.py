#!/usr/bin/env Python
# coding=utf-8
import copy
import logging
import threading
from abc import ABC

import numpy
import os
from tqdm import tqdm

from multiprocessing import Pool
import utils.names as names
import torch.nn
from transformers import PreTrainedModel, PretrainedConfig, DistilBertTokenizerFast
from transformers.utils import ModelOutput

from data_augmentations.loss_based_replacement import loss_based_replacement
from models.Base import BaseConfig


class LSTMConfig(BaseConfig):
    model_type = names.LSTM

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = 300
        self.hidden_dim = 300
        self.n_layers = 2
        self.bidirectional = True
        self.dropout_rate = 0.5


class LSTM(PreTrainedModel, ABC):
    config_class = LSTMConfig

    def __init__(self, config, tokenizer=None):
        super().__init__(config)

        self.tokenizer = tokenizer

        self.tfidf_word_dropout_flag = self.config.aug_ops[names.TFIDF_WORD_DROPOUT]
        self.random_word_dropout_flag = self.config.aug_ops[names.RANDOM_WORD_DROPOUT]
        self.loss_based_replacement_flag = self.config.aug_ops[names.LOSS_BASED_REPLACEMENT]

        self.embedding = torch.nn.Embedding(config.vocab_size, config.embedding_dim)
        self.lstm = torch.nn.LSTM(config.embedding_dim, config.hidden_dim, config.n_layers,
                                  bidirectional=config.bidirectional,
                                  dropout=config.dropout_rate, batch_first=True)
        self.fc = torch.nn.Linear(config.hidden_dim * 2 if config.bidirectional else config.hidden_dim,
                                  config.output_dim)
        self.dropout = torch.nn.Dropout(config.dropout_rate)

    def forward(self, input_ids, labels=None, dropout_prob=None, **kwargs):
        if self.training and self.tfidf_word_dropout_flag:
            keep = torch.bernoulli(1 - dropout_prob).bool()
            input_ids = torch.where(keep, input_ids, torch.empty_like(input_ids).fill_(0))
        if self.training and self.random_word_dropout_flag:
            keep = torch.empty_like(input_ids).bernoulli(1 - self.random_word_dropout_flag).bool()
            input_ids = torch.where(keep, input_ids, torch.empty_like(input_ids).fill_(0))

        if self.training and self.loss_based_replacement_flag and self.loss_based_replacement_flag > numpy.random.random():
            def current_model_loss(text, label):
                with torch.no_grad():
                    tokenizer = self.tokenizer
                    inner_embedding = copy.deepcopy(self.embedding).cuda()
                    inner_dropout = copy.deepcopy(self.dropout).cuda()
                    inner_lstm = copy.deepcopy(self.lstm).cuda()
                    inner_fc = copy.deepcopy(self.fc).cuda()

                    inner_input_ids = torch.tensor([tokenizer(text, padding='max_length', truncation=True, max_length=tokenizer.max_length)['input_ids']], device=self.device)
                    inner_embedded = inner_embedding(inner_input_ids)
                    inner_embedded = inner_dropout(inner_embedded)
                    inner_labels = torch.tensor([label], device=self.device)

                    output, (hidden, cell) = inner_lstm(inner_embedded)
                    if self.lstm.bidirectional:
                        hidden = inner_dropout(torch.cat([hidden[-1], hidden[-2]], dim=-1))
                    else:
                        hidden = inner_dropout(hidden[-1])
                    logits = inner_dropout(inner_fc(hidden))
                    criterion = torch.nn.CrossEntropyLoss()
                    loss = criterion(logits, inner_labels)
                return loss

            tokenizer = self.tokenizer
            texts = []
            for ids in input_ids:
                texts.append(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(ids)))

            def aug_transform(text, label):
                split_text = text.split()
                split_text = split_text[1: split_text.index('[SEP]')]
                text = " ".join(split_text)
                new_text = loss_based_replacement(text, label, current_model_loss)
                return new_text

            os.environ["TOKENIZERS_PARALLELISM"] = "false"

            import gevent
            jobs = [gevent.spawn(aug_transform, text, labels[i]) for i, text in enumerate(texts)]
            gevent.joinall(jobs)
            new_texts = [job.value for job in jobs]

            input_ids = torch.tensor(tokenizer(new_texts, padding='max_length', truncation=True, max_length=tokenizer.max_length)['input_ids'], device=self.device)

        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)
        output, (hidden, cell) = self.lstm(embedded)
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat([hidden[-1], hidden[-2]], dim=-1))
        else:
            hidden = self.dropout(hidden[-1])
        logits = self.dropout(self.fc(hidden))
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        return ModelOutput(logits=logits, loss=loss)
