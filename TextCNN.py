import numpy as np
import torch.nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers import AutoModel, AutoConfig

class CNNConfig(PretrainedConfig):
    model_type = 'cnn'
    def __init__(self, vocab_size=40000, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = 100
        self.n_filters = 100
        self.filter_sizes = [3, 4, 5]
        self.output_dim = 1
        self.dropout = 0.5

class CNN(PreTrainedModel):
    config_class = CNNConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embedding = torch.nn.Embedding(self.config.vocab_size, self.config.embedding_dim)

        self.convs = torch.nn.ModuleList([
            torch.nn.Conv2d(in_channels=1,
                      out_channels=self.config.n_filters,
                      kernel_size=(fs, self.config.embedding_dim))
            for fs in self.config.filter_sizes
        ])

        self.fc = torch.nn.Linear(len(self.config.filter_sizes) * self.config.n_filters, self.config.output_dim)

        self.dropout = torch.nn.Dropout(self.config.dropout)

    def forward(self, input_ids, labels, **kwargs):
        # text = [batch size, sent len]

        embedded = self.embedding(input_ids)

        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.unsqueeze(1)

        # embedded = [batch size, 1, sent len, emb dim]

        conved = [torch.nn.functional.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled = [torch.nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))

        logits = self.fc(cat).squeeze(1)

        # cat = [batch size, n_filters * len(filter_sizes)]
        criterion = torch.nn.BCEWithLogitsLoss()
        loss = criterion(logits, labels.float())


        return {'logits': logits, 'loss': loss}
