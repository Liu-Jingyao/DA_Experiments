from abc import ABC
from typing import Optional, Tuple, Union

import torch
from transformers import ElectraForSequenceClassification, ElectraConfig
from transformers.modeling_outputs import SequenceClassifierOutput

from data_augmentations.data_augmentation_consts import ONLINE_DATA_AUGMENTATION_DICT


class ELECTRAConfig(ElectraConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.aug_ops = None

class ELECTRAForSequenceClassification(ElectraForSequenceClassification, ABC):
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        dropout_prob = None, replacement_prob = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        if self.training:
            for aug_name, aug_prob in self.config.aug_ops.items():
                aug_res = ONLINE_DATA_AUGMENTATION_DICT[aug_name](input_ids, labels=labels, aug_prob=aug_prob,
                                                                  attention_mask=attention_mask,
                                                                  token_type_ids=token_type_ids,
                                                                  dropout_prob=dropout_prob,
                                                                  replacement_prob=replacement_prob,
                                                                  model=self, tokenizer=self.tokenizer,
                                                                  aug_name=aug_name)
                input_ids = aug_res['input_ids']
                attention_mask = aug_res.get('attention_mask', attention_mask)
                token_type_ids = aug_res.get('token_type_ids', token_type_ids)
        return super(ELECTRAForSequenceClassification, self).forward(input_ids, attention_mask, token_type_ids, position_ids,
                                                                     head_mask, inputs_embeds, labels, output_attentions,
                                                                     output_hidden_states, return_dict)