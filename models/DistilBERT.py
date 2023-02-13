import copy
import os
from abc import ABC
from typing import Optional, Tuple, Union, List

import torch
from torch import nn
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from transformers import PretrainedConfig, DistilBertModel, BatchEncoding
from transformers.modeling_outputs import BaseModelOutput, SequenceClassifierOutput
from transformers.models.distilbert.modeling_distilbert import Embeddings, TransformerBlock, \
    Transformer, create_sinusoidal_embeddings, DistilBertForSequenceClassification
import transformers

from models.Base import BaseConfig
from utils import names


class DistilBERTConfig(BaseConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class DistilBERTForSequenceClassification(DistilBertForSequenceClassification, ABC):
    def __init__(self, config: PretrainedConfig, aug_ops=None):
        super().__init__(config)
        self.distilbert = DistilBERT(config)
        self.keyword_enhance_flag = config.keyword_enhance_flag
        self.tfidf_word_dropout_flag = config.tfidf_word_dropout_flag

    def forward(self, input_ids: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None, inputs_embeds: Optional[torch.Tensor] = None,
                labels: Optional[torch.LongTensor] = None, output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None, **kwargs) -> Union[
        SequenceClassifierOutput, Tuple[torch.Tensor, ...]]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_labels)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + distilbert_output[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        ) # Encoder

class DistilBERT(DistilBertModel, ABC):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.embeddings = KeywordEnhancedEmbeddings(config)
        self.transformer = KeywordEnhancedTransformer(config)
        if self.keyword_enhance_flag:
            self.keyword_bert_model = None

    def forward(self, input_ids: Optional[torch.Tensor] = None,  attention_mask: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None, inputs_embeds: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None, **kwargs) -> Union[BaseModelOutput, Tuple[torch.Tensor, ...]]:
        # Note: when adding parameter, add it to KeywordEnhancedBertForSequenceClassification.forward, too!

        # lazy load
        if self.keyword_enhance_flag and not self.keyword_bert_model:
            self.keyword_bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased',
                                                                      num_labels=self.config.num_labels,
                                                                      mirror='tuna', ).cuda()
        if self.keyword_enhance_flag:
            keyword_ids = kwargs.get(names.KEYWORD_IDS, None)
            keyword_hidden_state = self.keyword_bert_model(input_ids=keyword_ids[:, :1]).last_hidden_state
        if self.training and self.tfidf_word_dropout_flag:
            dropout_prob = kwargs.get(names.DROPOUT_PROB, None)
            keep = torch.bernoulli(1 - dropout_prob).bool()
            input_ids = torch.where(keep, input_ids, torch.empty_like(input_ids).fill_(0))

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)  # (bs, seq_length)
        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)  # (bs, seq_length, dim)

        hidden_states = self.transformer.forward(
            x=inputs_embeds,
            attn_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **{'ex_hidden_state': keyword_hidden_state} if self.keyword_enhance_flag else {}
        )

        return hidden_states

class KeywordEnhancedEmbeddings(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.dim, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.dim)
        if config.sinusoidal_pos_embds:
            create_sinusoidal_embeddings(
                n_pos=config.max_position_embeddings, dim=config.dim, out=self.position_embeddings.weight
            )

        self.LayerNorm = nn.LayerNorm(config.dim, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_length = input_ids.size(1)
        if hasattr(self, "position_ids"):
            position_ids = self.position_ids[:, :seq_length]
        else:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)  # (max_seq_length)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # (bs, max_seq_length)

        word_embeddings = self.word_embeddings(input_ids)  # (bs, max_seq_length, dim)
        position_embeddings = self.position_embeddings(position_ids)  # (bs, max_seq_length, dim)

        embeddings = word_embeddings + position_embeddings# (bs, max_seq_length, dim)

        embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        embeddings = self.dropout(embeddings)  # (bs, max_seq_length, dim)

        return embeddings

class KeywordEnhancedTransformer(Transformer):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        module_list = [TransformerBlock(config) for _ in range(config.n_layers)]
        self.layer = nn.ModuleList(module_list)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False,
                output_hidden_states: bool = False, return_dict: Optional[bool] = None,
                modifier_token_lists: List[List[BatchEncoding]] = None,
                ex_hidden_state: Optional[torch.Tensor] = None) -> Union[BaseModelOutput, Tuple[torch.Tensor, ...]]:

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_state = x
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)
            layer_outputs = layer_module(
                x=hidden_state, attn_mask=attn_mask, head_mask=head_mask[i], output_attentions=output_attentions
            )
            hidden_state = layer_outputs[-1]

            if self.config.keyword_enhance_flag and i == 0:
                hidden_state = hidden_state + ex_hidden_state * 0.1

            if output_attentions:
                assert len(layer_outputs) == 2
                attentions = layer_outputs[0]
                all_attentions = all_attentions + (attentions,)
            else:
                assert len(layer_outputs) == 1

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)

        if not return_dict:
            return tuple(v for v in [hidden_state, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_state, hidden_states=all_hidden_states, attentions=all_attentions
        )