from abc import ABC
from typing import Optional, Union, Tuple

import torch
from transformers import AlbertForSequenceClassification, AlbertConfig
from transformers.modeling_outputs import SequenceClassifierOutput

from data_augmentations.data_augmentation_consts import ONLINE_DATA_AUGMENTATION_DICT

class ALBERTConfig(AlbertConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.aug_ops = None

class ALBERTForSequenceClassification(AlbertForSequenceClassification, ABC):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        dropout_prob = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[SequenceClassifierOutput, Tuple]:
        if self.training:
            for aug_name, aug_prob in self.config.aug_ops.items():
                aug_res = ONLINE_DATA_AUGMENTATION_DICT[aug_name](input_ids, labels=labels, aug_prob=aug_prob, dropout_prob=dropout_prob,
                                                                    model=self, tokenizer=self.tokenizer, aug_name=aug_name,
                                                                  attention_mask=attention_mask,
                                                                  token_type_ids=token_type_ids,
                                                                  )
                input_ids = aug_res['input_ids']
                attention_mask = aug_res.get('attention_mask', attention_mask)
                token_type_ids = aug_res.get('token_type_ids', token_type_ids)
        return super(ALBERTForSequenceClassification, self).forward(input_ids, attention_mask, token_type_ids, position_ids, head_mask,
                                                                    inputs_embeds, labels, output_attentions, output_hidden_states, return_dict)