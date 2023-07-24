from abc import ABC
from typing import Optional, Tuple, Union

import torch
from transformers import XLNetForSequenceClassification, XLNetConfig
from transformers.models.xlnet.modeling_xlnet import XLNetForSequenceClassificationOutput

from data_augmentations.data_augmentation_consts import ONLINE_DATA_AUGMENTATION_DICT


class XLNETConfig(XLNetConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.aug_ops = None


class XLNETForSequenceClassification(XLNetForSequenceClassification, ABC):
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        dropout_prob = None,
            replacement_prob=None,
        mems: Optional[torch.Tensor] = None,
        perm_mask: Optional[torch.Tensor] = None,
        target_mapping: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        input_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_mems: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,  # delete when `use_cache` is removed in XLNetModel
    ) -> Union[Tuple, XLNetForSequenceClassificationOutput]:
        if self.training:
            for aug_name, aug_prob in self.config.aug_ops.items():
                aug_res = ONLINE_DATA_AUGMENTATION_DICT[aug_name](input_ids, labels=labels, aug_prob=aug_prob,
                                                                  dropout_prob=dropout_prob,
                                                                  replacement_prob=replacement_prob,
                                                                  attention_mask=attention_mask,
                                                                  token_type_ids=token_type_ids,
                                                                  model=self, tokenizer=self.tokenizer,
                                                                  aug_name=aug_name)
                input_ids = aug_res['input_ids']
                attention_mask = aug_res.get('attention_mask', attention_mask)
                token_type_ids = aug_res.get('token_type_ids', token_type_ids)
        return super(XLNETForSequenceClassification, self).forward(input_ids, attention_mask, mems, perm_mask, target_mapping,
                                                                   token_type_ids, input_mask, head_mask, inputs_embeds, labels,
                                                                   use_mems, output_attentions, output_hidden_states, return_dict,
                                                                   **kwargs)