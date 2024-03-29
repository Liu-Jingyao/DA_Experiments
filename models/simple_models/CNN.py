from abc import ABC

import utils.names as names
import torch.nn
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from models.simple_models.Base import BaseConfig


class CNNConfig(BaseConfig):
    model_type = names.CNN
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = 100
        self.n_filters = 100
        self.filter_sizes = [3, 4, 5]
        self.dropout = 0.5

class CNN(PreTrainedModel, ABC):
    config_class = CNNConfig
    def __init__(self, config, *args):
        super().__init__(config)

        self.tfidf_word_dropout_flag = self.config.aug_ops[names.TFIDF_WORD_DROPOUT]
        self.random_word_dropout_flag = self.config.aug_ops[names.RANDOM_WORD_DROPOUT]

        self.embedding = torch.nn.Embedding(self.config.vocab_size, self.config.embedding_dim)
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv2d(in_channels=1,
                      out_channels=self.config.n_filters,
                      kernel_size=(fs, self.config.embedding_dim))
            for fs in self.config.filter_sizes
        ])
        self.fc = torch.nn.Linear(len(self.config.filter_sizes) * self.config.n_filters, self.config.output_dim)
        self.dropout = torch.nn.Dropout(self.config.dropout)

    def forward(self, input_ids, labels, dropout_prob=None, **kwargs):
        if self.training and self.tfidf_word_dropout_flag:
            keep = torch.bernoulli(1 - dropout_prob).bool()
            input_ids = torch.where(keep, input_ids, torch.empty_like(input_ids).fill_(0))
        if self.training and self.random_word_dropout_flag:
            keep = torch.empty_like(input_ids).bernoulli(1 - self.random_word_dropout_flag).bool()
            input_ids = torch.where(keep, input_ids, torch.empty_like(input_ids).fill_(0))

        embedded = self.embedding(input_ids)
        embedded = embedded.unsqueeze(1)

        conved = [torch.nn.functional.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [torch.nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        cat = self.dropout(torch.cat(pooled, dim=1))
        logits = self.fc(cat)
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(logits, labels)

        return ModelOutput(logits=logits, loss=loss)