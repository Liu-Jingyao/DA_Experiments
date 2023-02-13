from abc import ABC

import utils.names as names
import torch.nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.utils import ModelOutput

class LSTMConfig(PretrainedConfig):
    model_type = names.LSTM
    def __init__(self, vocab_size=4000, num_labels=3, aug_ops=[], **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = 300
        self.hidden_dim = 300
        self.output_dim = num_labels
        self.n_layers = 2
        self.bidirectional = True
        self.dropout_rate = 0.5
        self.aug_ops = aug_ops

class LSTM(PreTrainedModel, ABC):
    config_class = LSTMConfig
    def __init__(self, config):
        super().__init__(config)

        self.keyword_augmentation_flag = names.KEYWORD_ENHANCE in config.aug_ops
        self.tf_idf_augmentation_flag = False
        self.tfidf_word_dropout_flag = names.TFIDF_WORD_DROPOUT in config.aug_ops

        self.embedding = torch.nn.Embedding(config.vocab_size, config.embedding_dim)
        self.lstm = torch.nn.LSTM(config.embedding_dim, config.hidden_dim, config.n_layers, bidirectional=config.bidirectional,
                            dropout=config.dropout_rate, batch_first=True)
        self.fc = torch.nn.Linear(config.hidden_dim * 2 if config.bidirectional else config.hidden_dim, config.output_dim)
        self.dropout = torch.nn.Dropout(config.dropout_rate)

    def forward(self, input_ids, dropout_prob=None, labels=None, **kwargs):
        if self.training and self.tfidf_word_dropout_flag:
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