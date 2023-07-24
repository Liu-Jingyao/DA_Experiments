import copy
import os
from abc import ABC
from typing import Optional, Tuple, Union, List

import torch
from transformers import PretrainedConfig, DistilBertModel, BatchEncoding, DistilBertConfig
from transformers.modeling_outputs import BaseModelOutput, SequenceClassifierOutput
from transformers.models.distilbert.modeling_distilbert import Embeddings, TransformerBlock, \
    Transformer, create_sinusoidal_embeddings, DistilBertForSequenceClassification, MultiHeadSelfAttention

from data_augmentations.data_augmentation_consts import ONLINE_DATA_AUGMENTATION_DICT


class DistilBERTConfig(DistilBertConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.aug_ops = None

class DistilBERTForSequenceClassification(DistilBertForSequenceClassification, ABC):
    def forward(self, input_ids: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None,
                dropout_prob: Optional[torch.Tensor] = None,
                replacement_prob = None,
                head_mask: Optional[torch.Tensor] = None, inputs_embeds: Optional[torch.Tensor] = None,
                labels: Optional[torch.LongTensor] = None, output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None,
                **kwargs) -> Union[SequenceClassifierOutput, Tuple[torch.Tensor, ...]]:
        if self.training:
            for aug_name, aug_prob in self.config.aug_ops.items():
                aug_res = ONLINE_DATA_AUGMENTATION_DICT[aug_name](input_ids, labels=labels, aug_prob=aug_prob,
                                                                  dropout_prob=dropout_prob,
                                                                  replacement_prob=replacement_prob,
                                                                  attention_mask=attention_mask,
                                                                  model=self, tokenizer=self.tokenizer,
                                                                  aug_name=aug_name)
                input_ids = aug_res['input_ids']
                attention_mask = aug_res.get('attention_mask', attention_mask)
        return super(DistilBERTForSequenceClassification, self).forward(input_ids, attention_mask, head_mask, inputs_embeds,
                                                                        labels, output_attentions, output_hidden_states, return_dict)