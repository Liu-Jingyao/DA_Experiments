from abc import ABC

import utils.names as names
import torch.nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.utils import ModelOutput

class CNNConfig(PretrainedConfig):
    model_type = names.CNN
    def __init__(self, vocab_size=40000, num_labels=3, aug_ops=[], **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = 100
        self.n_filters = 100
        self.filter_sizes = [3, 4, 5]
        self.output_dim = num_labels
        self.dropout_rate = 0.5
        self.aug_ops = aug_ops

class CNN(PreTrainedModel, ABC):
    config_class = CNNConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.keyword_augmentation_flag = names.KEYWORD_ENHANCE in config.aug_ops
        self.tf_idf_augmentation_flag = False
        self.tfidf_word_dropout_flag = names.TFIDF_WORD_DROPOUT in config.aug_ops

        self.embedding = torch.nn.Embedding(self.config.vocab_size, self.config.embedding_dim)
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv2d(in_channels=1,
                      out_channels=self.config.n_filters,
                      kernel_size=(fs, self.config.embedding_dim))
            for fs in self.config.filter_sizes
        ])
        self.fc = torch.nn.Linear(len(self.config.filter_sizes) * self.config.n_filters, self.config.output_dim)
        self.dropout = torch.nn.Dropout(self.config.dropout)
        if self.tf_idf_augmentation_flag:
            self.tfidf_embeddings = torch.nn.Embedding(config.max_position_embeddings, config.dim)

    def forward(self, input_ids, labels, tfidfs, dropout_prob=None, **kwargs):
        if self.training and self.tfidf_word_dropout_flag:
            keep = torch.bernoulli(1 - dropout_prob).bool()
            input_ids = torch.where(keep, input_ids, torch.empty_like(input_ids).fill_(0))

        embedded = self.embedding(input_ids)
        if self.tf_idf_augmentation_flag:
            tfidf_embeddings = self.tfidf_embeddings(tfidfs)  # (bs, max_seq_length, dim)
            embedded += tfidf_embeddings
        embedded = embedded.unsqueeze(1)

        conved = [torch.nn.functional.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [torch.nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        cat = self.dropout(torch.cat(pooled, dim=1))
        logits = self.fc(cat)
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(logits, labels)

        return ModelOutput(logits=logits, loss=loss)

