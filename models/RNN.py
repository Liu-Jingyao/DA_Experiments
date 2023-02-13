from abc import ABC

from torch.autograd import Variable

import utils.names as names
import torch.nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.utils import ModelOutput

class RNNConfig(PretrainedConfig):
    model_type = names.RNN
    def __init__(self, vocab_size=40000, num_labels=3, aug_ops=[], **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = 300
        self.vocab_size = vocab_size
        self.output_dim = num_labels
        self.hidden_dim = 100
        self.n_layers = 5
        self.dropout_rate = 0.5
        self.bidirectional = True
        self.aug_ops = aug_ops

class RNN(PreTrainedModel, ABC):
    config_class = RNNConfig
    def __init__(self, config):
        super().__init__(config)

        self.tfidf_word_dropout_flag = names.TFIDF_WORD_DROPOUT in config.aug_ops
        self.random_word_dropout_flag = names.RANDOM_WORD_DROPOUT in config.aug_ops

        self.hidden_dim = config.hidden_dim
        self.n_layers = config.n_layers
        self.embedding = torch.nn.Embedding(config.vocab_size, config.embedding_dim)
        self.rnn = torch.nn.RNN(config.embedding_dim, config.hidden_dim, config.n_layers, batch_first=True,
                                bidirectional=config.bidirectional)
        self.fc = torch.nn.Linear(config.hidden_dim * 2 if config.bidirectional else config.hidden_dim, config.output_dim)
        self.dropout = torch.nn.Dropout(config.dropout_rate)

    def forward(self, input_ids, labels, dropout_prob=None, **kwargs):
        if self.training and self.tfidf_word_dropout_flag:
            keep = torch.bernoulli(1 - dropout_prob).bool()
            input_ids = torch.where(keep, input_ids, torch.empty_like(input_ids).fill_(0))
        if self.training and self.random_word_dropout_flag:
            keep = torch.empty_like(input_ids).bernoulli(1 - 0.025).bool()
            input_ids = torch.where(keep, input_ids, torch.empty_like(input_ids).fill_(0))
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)
        output, hidden = self.rnn(embedded)
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat([hidden[-1], hidden[-2]], dim=-1))
        else:
            hidden = self.dropout(hidden[-1])
        logits = self.fc(hidden)
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        return ModelOutput(logits=logits, loss=loss)