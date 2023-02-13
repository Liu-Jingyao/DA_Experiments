from abc import ABC

import utils.names as names
import torch.nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.utils import ModelOutput

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
    def __init__(self, config):
        super().__init__(config)

        self.tfidf_word_dropout_flag = self.config.tfidf_word_dropout_flag

        self.embedding = torch.nn.Embedding(config.vocab_size, config.embedding_dim)
        self.lstm = torch.nn.LSTM(config.embedding_dim, config.hidden_dim, config.n_layers, bidirectional=config.bidirectional,
                            dropout=config.dropout_rate, batch_first=True)
        self.fc = torch.nn.Linear(config.hidden_dim * 2 if config.bidirectional else config.hidden_dim, config.output_dim)
        self.dropout = torch.nn.Dropout(config.dropout_rate)

    def forward(self, input_ids, labels=None, **kwargs):
        if self.training and self.tfidf_word_dropout_flag:
            dropout_prob = kwargs.get(names.DROPOUT_PROB, None)
            keep = torch.bernoulli(1 - dropout_prob).bool()
            input_ids = torch.where(keep, input_ids, torch.empty_like(input_ids).fill_(0))

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