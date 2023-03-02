import copy
import os
from abc import ABC
from typing import Optional, Tuple, Union, List

import numpy
import torch
from torch import nn
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from transformers import PretrainedConfig, DistilBertModel, BatchEncoding, DistilBertConfig
from transformers.modeling_outputs import BaseModelOutput, SequenceClassifierOutput
from transformers.models.distilbert.modeling_distilbert import Embeddings, TransformerBlock, \
    Transformer, create_sinusoidal_embeddings, DistilBertForSequenceClassification, MultiHeadSelfAttention
import transformers

from data_augmentations.loss_based_replacement import loss_based_replacement
from models.Base import BaseConfig
from utils import names


class DistilBERTConfig(DistilBertConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.aug_ops = None
        self.seq_len = None

class DistilBERTForSequenceClassification(DistilBertForSequenceClassification, ABC):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.distilbert = DistilBERT(config)
        self.hidden_state_pooling_flag = config.aug_ops[names.HIDDEN_STATE_POOLING]
        self.loss_based_replacement_flag = self.config.aug_ops[names.LOSS_BASED_REPLACEMENT]

        if self.hidden_state_pooling_flag:
            self.config.output_hidden_states = True
            self.attention_pool = MultiHeadSelfAttention(config)
            Wh = torch.FloatTensor(1, config.n_layers, device=self.device)
            self.Wh = torch.nn.Parameter(nn.init.xavier_uniform_(Wh, gain=nn.init.calculate_gain('relu')))
            q = torch.FloatTensor(1, device=self.device)
            self.q = torch.nn.Parameter(nn.init.uniform(q))
        self.hidden_state_cnn_flag = config.aug_ops[names.HIDDEN_STATE_CNN]
        if self.hidden_state_cnn_flag:
            self.config.output_hidden_states = True
            self.convs = torch.nn.Sequential(
                torch.nn.Conv2d(7, 7, (3, 3)),
                torch.nn.ReLU(),
                torch.nn.Conv2d(7, 7, (4, 4)),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(7, 7, (5, 5)),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(7, 7, (13, 1)),
                torch.nn.ReLU()
            )
            self.fc = torch.nn.Linear(188, self.config.dim)

    def forward(self, input_ids: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None,
                tfidfs: Optional[torch.Tensor] = None, keyword_ids: Optional[torch.Tensor] = None,
                dropout_prob: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None, inputs_embeds: Optional[torch.Tensor] = None,
                labels: Optional[torch.LongTensor] = None, output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None) -> Union[
        SequenceClassifierOutput, Tuple[torch.Tensor, ...]]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.training and self.loss_based_replacement_flag and self.loss_based_replacement_flag > numpy.random.random():
            def current_model_loss(text, label, inner_attention_mask):
                with torch.no_grad():
                    tokenizer = self.tokenizer
                    inner_input_ids = torch.tensor([tokenizer(text, padding='max_length', truncation=True, max_length=tokenizer.max_length)['input_ids']], device=self.device)
                    inner_labels = torch.tensor([label], device=self.device)

                    inner_model = copy.deepcopy(self.distilbert).cuda()
                    inner_pre_classifier = copy.deepcopy(self.pre_classifier).cuda()
                    inner_dropout = copy.deepcopy(self.dropout).cuda()
                    inner_classifier = copy.deepcopy(self.classifier).cuda()


                    inner_model_output = inner_model(
                        input_ids=inner_input_ids,
                        attention_mask=inner_attention_mask,
                        head_mask=head_mask,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                    )
                    inner_hidden_state = inner_model_output[0]  # (bs, embed_len, dim)
                    inner_pooled_output = inner_hidden_state[:, 0]  # (bs, dim)
                    inner_pooled_output = inner_pre_classifier(inner_pooled_output)  # (bs, dim)
                    inner_pooled_output = nn.ReLU()(inner_pooled_output)  # (bs, dim)
                    inner_pooled_output = inner_dropout(inner_pooled_output)  # (bs, dim)
                    inner_logits = inner_classifier(inner_pooled_output)  # (bs, num_labels)
                    inner_criterion = torch.nn.CrossEntropyLoss()
                    inner_loss = inner_criterion(inner_logits, inner_labels)
                return inner_loss

            tokenizer = self.tokenizer
            texts = []
            for ids in input_ids:
                texts.append(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(ids)))

            def aug_transform(text, label, inner_attention_mask):
                split_text = text.split()
                split_text = split_text[1: split_text.index('[SEP]')]
                text = " ".join(split_text)
                new_text = loss_based_replacement(text, label, current_model_loss, inner_attention_mask)
                return new_text

            os.environ["TOKENIZERS_PARALLELISM"] = "false"

            import gevent
            jobs = [gevent.spawn(aug_transform, text, labels[i], attention_mask[i]) for i, text in enumerate(texts)]
            gevent.joinall(jobs)
            new_texts = [job.value for job in jobs]

            input_ids = torch.tensor(tokenizer(new_texts, padding='max_length', truncation=True, max_length=tokenizer.max_length)['input_ids'], device=self.device)


        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            keyword_ids=keyword_ids,
            dropout_prob=dropout_prob,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_state = distilbert_output[0]  # (bs, embed_len, dim)

        if self.hidden_state_pooling_flag:
            batch_size = input_ids.size(0)
            all_hidden_states = torch.stack(distilbert_output[1])  # (layer_num, bs, seq_len, dim)
            all_hidden_states = all_hidden_states.transpose(0, 1)  # (bs, layer_num, seq_len, dim)
            hcls = all_hidden_states[:,1:,0,:] # (bs, layer_num, dim)
            Wh = torch.broadcast_to(self.Wh, (batch_size, self.Wh.size(0), self.Wh.size(1)))
            q = torch.broadcast_to(self.q, (batch_size, hcls.size(1),hcls.size(2)))
            sa_output = self.attention_pool(
                query=q,
                key=hcls,
                value=hcls,
                mask=torch.ones((hcls.size(0),1,1,hcls.size(1)), device=self.device)
            ) # (bs, layer_num, dim)
            sa_output = sa_output[0]
            hidden_state = torch.matmul(Wh, sa_output) # (bs, layer_num, dim)

        if self.hidden_state_cnn_flag:
            x = self.convs(torch.stack(distilbert_output[1]).transpose(0, 1))
            pooled_output = self.fc(x.squeeze())
        else:
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
            logits=logits
        ) # Encoder

class DistilBERT(DistilBertModel, ABC):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)

        self.keyword_enhance_flag = config.aug_ops[names.KEYWORD_ENHANCE]
        self.tfidf_word_dropout_flag = config.aug_ops[names.TFIDF_WORD_DROPOUT]

        self.embeddings = KeywordEnhancedEmbeddings(config)
        self.transformer = KeywordEnhancedTransformer(config)
        if self.keyword_enhance_flag:
            self.keyword_bert_model = None

    def forward(self, input_ids: Optional[torch.Tensor] = None,  attention_mask: Optional[torch.Tensor] = None,
                tfidfs: Optional[torch.Tensor] = None, keyword_ids: Optional[torch.Tensor] = None,
                dropout_prob: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None, inputs_embeds: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None) -> Union[BaseModelOutput, Tuple[torch.Tensor, ...]]:
        # Note: when adding parameter, add it to KeywordEnhancedBertForSequenceClassification.forward, too!

        # lazy load
        if self.keyword_enhance_flag and not self.keyword_bert_model:
            self.keyword_bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased',
                                                                      num_labels=self.config.num_labels,
                                                                      mirror='tuna', ).cuda()
        if self.keyword_enhance_flag:
            keyword_hidden_state = self.keyword_bert_model(input_ids=keyword_ids[:, :1]).last_hidden_state
        if self.training and self.tfidf_word_dropout_flag:
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
            attention_mask = torch.ones(input_shape, device=device)  # (bs, embed_length)
        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)  # (bs, embed_length, dim)

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
        self.config = config
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

            if self.config.aug_ops[names.KEYWORD_ENHANCE] and i == 0:
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