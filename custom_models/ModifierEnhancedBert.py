import copy
from abc import ABC
from typing import Optional, Tuple, Union, List

import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedTokenizer
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.distilbert.modeling_distilbert import DistilBertPreTrainedModel, Embeddings, TransformerBlock, \
    Transformer
import transformers


class ModifierEnhancedBert(transformers.DistilBertModel, ABC):
    def __init__(self, config: PretrainedConfig, tokenizer: PreTrainedTokenizer):
        DistilBertPreTrainedModel.__init__(self, config)
        self.modifier_finder = ModifierFinder(tokenizer)
        self.embeddings = Embeddings(config)  # Embeddings
        self.transformer = ModifierEnhancedTransformer(config)  # Encoder

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, input_ids: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None, inputs_embeds: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None) -> Union[BaseModelOutput, Tuple[torch.Tensor, ...]]:
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

        modifier_lists = self.modifier_finder.find_modifiers(input_ids)
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)  # (bs, seq_length, dim)
        return self.transformer.forward(
            x=inputs_embeds,
            attn_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            modifier_lists=modifier_lists
        )

class ModifierEnhancedTransformer(Transformer):
    def __init__(self, config: PretrainedConfig):
        nn.Module.__init__(self)
        self.n_layers = config.n_layers
        module_list = [TransformerBlock(config) for _ in range(config.n_layers)]
        module_list.insert(0, ModifierEnhancedTransformerBlock(config))
        self.layer = nn.ModuleList(module_list)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False,
                output_hidden_states: bool = False, return_dict: Optional[bool] = None,
                modifier_lists: List[List] = None) -> Union[BaseModelOutput, Tuple[torch.Tensor, ...]]:
        # 对每个样本，将所有修饰词过一遍SingleLayerBERT得到hidden state，每个样本所有形容词的hidden state加成一个
        modifier_hidden_states = []
        for instance_modifiers in modifier_lists:
            instance_hidden_state = torch.zeros()
            for modifier in instance_modifiers:
                instance_hidden_state += SingleLayerModifierEnhancedBert(modifier)
            modifier_hidden_states.append(instance_hidden_state)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_state = x
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)
            if i == 0:
                layer_outputs = layer_module.forward(
                    x=hidden_state, attn_mask=attn_mask, head_mask=head_mask[i], output_attentions=output_attentions,
                    modifier_hidden_states=modifier_hidden_states
                )
            else:
                layer_outputs = layer_module(
                    x=hidden_state, attn_mask=attn_mask, head_mask=head_mask[i], output_attentions=output_attentions
                )
            hidden_state = layer_outputs[-1]

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


class ModifierEnhancedTransformerBlock(TransformerBlock):
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False,
                modifier_hidden_states = None) -> Tuple[torch.Tensor, ...]:
        # Self-Attention
        sa_output = self.attention(
            query=x,
            key=x,
            value=x,
            mask=attn_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        if output_attentions:
            sa_output, sa_weights = sa_output  # (bs, seq_length, dim), (bs, n_heads, seq_length, seq_length)
        else:  # To handle these `output_attentions` or `output_hidden_states` cases returning tuples
            assert type(sa_output) == tuple
            sa_output = sa_output[0]

        for modifier_hidden_state in modifier_hidden_states:
            sa_output += modifier_hidden_state * 0.1

        sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)

        # Feed Forward Network
        ffn_output = self.ffn(sa_output)  # (bs, seq_length, dim)
        ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)

        output = (ffn_output,)
        if output_attentions:
            output = (sa_weights,) + output
        return output

class SingleLayerModifierEnhancedBert(transformers.DistilBertModel, ABC):
    def __init__(self, config: PretrainedConfig, origin_model: ModifierEnhancedBert):
        DistilBertPreTrainedModel.__init__(self, config)
        self.embeddings = Embeddings(config)  # Embeddings
        self.transformer = SingleLayerModifierEnhancedTransformer(config, origin_transformer=origin_model.transformer)  # Encoder

        # Initialize weights and apply final processing
        self.post_init()

class SingleLayerModifierEnhancedTransformer(Transformer):
    def __init__(self, config: PretrainedConfig, origin_transformer: ModifierEnhancedTransformer):
        nn.Module.__init__(self)
        self.n_layers = config.n_layers
        module_list = [copy.deepcopy(origin_transformer.layer[0])]
        self.layer = nn.ModuleList(module_list)

class ModifierFinder:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def find_modifiers(self, input_ids: Optional[torch.Tensor] = None):
        return [transformers.DistilBertTokenizer.convert_ids_to_tokens(self.tokenizer, id_input.to_list()) for id_input
                in input_ids]