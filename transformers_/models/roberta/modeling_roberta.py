# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch RoBERTa model."""
import copy
import numpy as np
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN, gelu
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_roberta import RobertaConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "roberta-base"
_CONFIG_FOR_DOC = "RobertaConfig"
_TOKENIZER_FOR_DOC = "RobertaTokenizer"

ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "roberta-base",
    "roberta-large",
    "roberta-large-mnli",
    "distilroberta-base",
    "roberta-base-openai-detector",
    "roberta-large-openai-detector",
    # See all RoBERTa models at https://huggingface.co/models?filter=roberta
]


class RobertaEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            self.register_buffer(
                "token_type_ids",
                torch.zeros(self.position_ids.size(), dtype=torch.long),
                persistent=False,
            )

        # End copy
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0,
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
            
        # initializing <g> with the avg of target tokens
#         mask = (VDC_info == 1).long().unsqueeze(2)
#         pooled = (inputs_embeds*mask).sum(dim=1)/mask.sum(dim=1)
#         inputs_embeds[:, 1] = pooled
        
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)

# Copied from transformers.models.bert.modeling_bert.BertSelfAttention with Bert->Roberta
class RobertaSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        VDC_info = None,
        sgg_info = None,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)
        
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)
        
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key
        
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # attention_scores.size() = torch.tensor([32,12,128,128])
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Normalize the attention scores to probabilities.
        
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        
        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertSelfAttention with Bert->Roberta
class RobertaSelfAttention_t_star(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.query_t_star = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
#         self.key_t_star = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
#         self.value_t_star = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        VDC_info = None,
        sgg_info = None,
    ) -> Tuple[torch.Tensor]:
        
#         mixed_query_layer = self.query(hidden_states)
#         query_layer = self.transpose_for_scores(mixed_query_layer)
        
        t_positions = (VDC_info == 1).float()
        not_t_positions = (VDC_info != 1).float()
        
        ## Avg pooling
        q_1 = self.query(hidden_states)
        t_query_layer = self.query_t_star(hidden_states) * t_positions.unsqueeze(2)
        mixed_query_layer = q_1 * not_t_positions.unsqueeze(2) + q_1 * t_positions.unsqueeze(2) * 0.5 + t_query_layer * 0.5
        query_layer = self.transpose_for_scores(mixed_query_layer)
        
#         k_1 = self.key(hidden_states)
#         t_key_layer = self.key_t_star(hidden_states) * t_positions.unsqueeze(2)
#         mixed_key_layer = k_1 * not_t_positions.unsqueeze(2) + k_1 * t_positions.unsqueeze(2) * 0.5 + t_key_layer * 0.5
#         key_layer = self.transpose_for_scores(mixed_key_layer)
        
#         v_1 = self.value(hidden_states)
#         t_value_layer = self.value_t_star(hidden_states) * t_positions.unsqueeze(2)
#         mixed_value_layer = v_1 * not_t_positions.unsqueeze(2) + v_1 * t_positions.unsqueeze(2) * 0.5 + t_value_layer * 0.5
#         value_layer = self.transpose_for_scores(mixed_value_layer)
        
        
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key
        
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # attention_scores.size() = torch.tensor([32,12,128,128])
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Normalize the attention scores to probabilities.
        
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        
        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs
    
    
# Copied from transformers.models.bert.modeling_bert.BertSelfAttention with Bert->Roberta
class RobertaSelfAttention_Linear_Q(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder
        
        ######## new-Q-1
# #         self.query_vic = nn.Linear(config.hidden_size, 4)
#         self.query_vic_0 = nn.Parameter(torch.FloatTensor(768))
# #         self.query_vic_1 = nn.Parameter(torch.FloatTensor(768))
# #         self.query_vic_2 = nn.Parameter(torch.FloatTensor(768))
# #         self.query_vic_3 = nn.Parameter(torch.FloatTensor(768))
#         self.query_vic_0.data.normal_(mean=0.0, std=config.initializer_range)
# #         self.query_vic_1.data.normal_(mean=0.0, std=config.initializer_range)
# #         self.query_vic_2.data.normal_(mean=0.0, std=config.initializer_range)
# #         self.query_vic_3.data.normal_(mean=0.0, std=config.initializer_range)
        ########
    
        ######## new-Q-2
        self.query_vic_0 = nn.Linear(config.hidden_size, config.hidden_size)
#         self.query_vic_1 = nn.Linear(config.hidden_size, config.hidden_size)
#         self.query_vic_2 = nn.Linear(config.hidden_size, config.hidden_size)
#         self.query_vic_3 = nn.Linear(config.hidden_size, config.hidden_size)
        
        # 선택: newly initialize them or use the pre-trained query as initial
#         self.query_vic_0.weight.data = self.query.weight.data
#         self.query_vic_0.bias.data = self.query.bias.data
#         self.query_vic_1.weight.data = self.query.weight.data
#         self.query_vic_1.bias.data = self.query.bias.data
#         self.query_vic_2.weight.data = self.query.weight.data
#         self.query_vic_2.bias.data = self.query.bias.data
#         self.query_vic_3.weight.data = self.query.weight.data
#         self.query_vic_3.bias.data = self.query.bias.data

        # 선택: newly initialize
#         self.query_vic_0.weight.data.normal_(mean=0.0, std=config.initializer_range)
#         if self.query_vic_0.bias is not None:
#             self.query_vic_0.bias.data.zero_()
        ########
        
        self.VIC_gate = nn.Linear(768, 768)
        self.VIC_gate.weight.data.normal_(mean=0.0, std=config.initializer_range)
        if self.VIC_gate.bias is not None:
            self.VIC_gate.bias.data.zero_()
            
        self.VIC_gate_2 = nn.Linear(768, 48)
        self.VIC_gate_2.weight.data.normal_(mean=0.0, std=config.initializer_range)
        if self.VIC_gate_2.bias is not None:
            self.VIC_gate_2.bias.data.zero_()
            
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        ######## 
        
    def sg_pool(self, layer, x: torch.Tensor) -> torch.Tensor:
        cat = torch.mean(x, dim=1)
        pooled_output = layer(cat)
        pooled_output = self.tanh(pooled_output)

        attention_scores = torch.matmul(pooled_output.unsqueeze(1), x.transpose(1,2))
        attention_scores = attention_scores.view(-1,2)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        final_output = torch.matmul(attention_probs.unsqueeze(1), x).view(-1, 768)
        
        return final_output
    
    def sg_concat_pool(self, layer, x: torch.Tensor) -> torch.Tensor:
        return self.tanh(layer(x.view(-1, 2*768)))
    
    def a_pool(self, layer, x:torch.Tensor) -> torch.Tensor:
        return self.tanh(layer(x))
    
    def all_pool(self, layer, x: torch.Tensor) -> torch.Tensor:
        c = torch.mean(x, dim=1)
        pooled_output = layer(c)
        pooled_output = self.tanh(pooled_output)

        attention_scores = torch.matmul(pooled_output.unsqueeze(1), x.transpose(1,2))
        attention_scores = attention_scores.view(-1,128)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        final_output = torch.matmul(attention_probs.unsqueeze(1), x).view(-1, 768)
        
        return final_output
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        target_idx = None,
        gcls_update = False,
        g_config = None,
        VIC_gate = None,
        VDC_gate = None,
        VDC_info = None,
        head_wise = None,
        sgg_info = None,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)
        
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)
        
        ######## new-Q-1
#         query_vic_0 = self.query_vic_0.unsqueeze(0).unsqueeze(1).repeat(query_layer.size(0), 1, 1).view(query_layer.size(0), 1, self.num_attention_heads, self.attention_head_size).permute(0, 2, 1, 3)
#         query_vic_1 = self.query_vic_1.unsqueeze(0).unsqueeze(1).repeat(query_layer.size(0), 1, 1).view(query_layer.size(0), 1, self.num_attention_heads, self.attention_head_size).permute(0, 2, 1, 3)
#         query_vic_2 = self.query_vic_2.unsqueeze(0).unsqueeze(1).repeat(query_layer.size(0), 1, 1).view(query_layer.size(0), 1, self.num_attention_heads, self.attention_head_size).permute(0, 2, 1, 3)
#         query_vic_3 = self.query_vic_3.unsqueeze(0).unsqueeze(1).repeat(query_layer.size(0), 1, 1).view(query_layer.size(0), 1, self.num_attention_heads, self.attention_head_size).permute(0, 2, 1, 3)
        
#         query_layer = torch.cat((query_layer, query_vic_0, query_vic_1, query_vic_2, query_vic_3), dim = 2)
#         query_layer = torch.cat((query_layer, query_vic_0), dim = 2)
        ########
        
        
#         query_vic_0 = self.sg_pool(layer = self.query_vic_0, x = hidden_states[:,:2,:])
        query_vic_0 = self.a_pool(layer = self.query_vic_0, x = hidden_states[:,2,:])
        
#         query_vic_0 = self.sg_concat_pool(layer = self.query_vic_0, x = hidden_states[:,:2,:])
        ######## Method 1
#         query_vic_0 = query_vic_0.unsqueeze(1).view(hidden_states.size(0), 1, 12, 64).permute(0,2,1,3)
#         query_layer = torch.cat((query_layer, query_vic_0), dim = 2)
        ########
        
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        
        
        ######## Method 1
#         attention_scores__ = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        
    
        ######## Method 2
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        
        
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key
        
        ######### Method 1
#         attention_scores_vic = attention_scores__[:, :, -1:, :]
#         attention_probs_vic = nn.functional.softmax(attention_scores_vic, dim=-1)
        
#         context_layer_vic = torch.matmul(attention_probs_vic, value_layer)
#         context_layer_vic = context_layer_vic.permute(0, 2, 1, 3).contiguous()
#         new_context_layer_shape_vic = context_layer_vic.size()[:-2] + (self.all_head_size,)
#         context_layer_vic = context_layer_vic.view(new_context_layer_shape_vic)
#         VIC_gate = self.VIC_gate(context_layer_vic)
#         VIC_gate = self.tanh(VIC_gate)
#         VIC_gate = self.VIC_gate_2(VIC_gate)
#         VIC_gate = self.sigmoid(VIC_gate)
        
#         attention_scores= attention_scores__[:,:, :-1,:]
#         attention_scores[:,:,:2,:2] += VIC_gate.log().transpose(1,2).view(-1,12,2,2)
        #########
        
        ######### Method 2
        VIC_gate = self.VIC_gate(query_vic_0)
        VIC_gate = self.tanh(VIC_gate)
        VIC_gate = self.VIC_gate_2(VIC_gate)
        VIC_gate = self.sigmoid(VIC_gate) +1e-10
        
        attention_scores[:,:,:2,:2] += VIC_gate.log().view(-1,12,2,2)
        #########
        
        
        #########
        
#         # Head-wise learning
#         if VIC_gate != None:
# #             assert attention_mask != None
# #             assert (VIC_gate >= 0.0).all()
#             VIC_gate = VIC_gate + 1e-10
#             if head_wise == True:
#                 attention_scores[:,:,:2,:2] += VIC_gate.log().view(-1,12,2,2)
#             else:
#                 attention_scores[:,:,:2,:2] += VIC_gate.log().view(-1,2,2).unsqueeze(1)
            
#         if VDC_gate != None:
# #             assert attention_mask == None
#             if head_wise == True:
#                 VDC_info = VDC_info.long().unsqueeze(1).repeat(1,12,1)
#                 VDC_gate = nn.functional.softmax(VDC_gate.reshape(VDC_info.size(0), 12, -1), dim = -1)
#                 #y = torch.gather(input = VDC_gate.reshape(-1, 12, 7), dim = -1, index = VDC_info)
#                 y = VDC_gate[torch.arange(VDC_info.size(0)).unsqueeze(1).unsqueeze(2), torch.arange(12).unsqueeze(1), VDC_info[torch.arange(VDC_info.size(0))]]
#                 #assert (y == y2).all()

#                 attention_scores[:,:,1,2:] += (y[:,:,2:] + 1e-10).log()
#             else:
#                 VDC_info = VDC_info.long()
#                 VDC_gate = nn.functional.softmax(VDC_gate, dim = -1)
#                 # y = torch.gather(input = VDC_gate.reshape(-1, 12, 7), dim = -1, index = VDC_info)
#                 y = VDC_gate[torch.arange(VDC_info.size(0)).unsqueeze(1), VDC_info[torch.arange(VDC_info.size(0))]]
#                 #assert (y == y2).all()

#                 attention_scores[:,:,1,2:] += (y[:, 2:] + 1e-10).log().unsqueeze(1)
                
        #########
        
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # attention_scores.size() = torch.tensor([32,12,128,128])
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Normalize the attention scores to probabilities.
        
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        
        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs
    
    
# Copied from transformers.models.bert.modeling_bert.BertSelfOutput
class RobertaSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertAttention with Bert->Roberta
class RobertaAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = RobertaSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = RobertaSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        VDC_info = None,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            VDC_info = VDC_info
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class RobertaAttention_t_star(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = RobertaSelfAttention_t_star(config, position_embedding_type=position_embedding_type)
        self.output = RobertaSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        VDC_info = None
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            VDC_info = VDC_info,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

    
# Copied from transformers.models.bert.modeling_bert.BertIntermediate
class RobertaIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput
class RobertaOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertLayer with Bert->Roberta
class RobertaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = RobertaAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = RobertaAttention(config, position_embedding_type="absolute")
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        VDC_info = None,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            VDC_info = VDC_info,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# Copied from transformers.models.bert.modeling_bert.BertLayer with Bert->Roberta
class RobertaLayer_t_star(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = RobertaAttention_t_star(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = RobertaAttention(config, position_embedding_type="absolute")
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)
        
#         self.intermediate_t_star = RobertaIntermediate(config)
#         self.output_t_star = RobertaOutput(config)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        VDC_info = None,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            VDC_info = VDC_info,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        
#         layer_output_t_star = apply_chunking_to_forward(
#             self.feed_forward_chunk_t_star, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
#         )
        
#         t_positions = (VDC_info == 1).float()
#         not_t_positions = (VDC_info != 1).float()
        
#         layer_output = layer_output * not_t_positions.unsqueeze(2) \
#                        + layer_output * t_positions.unsqueeze(2) * 0.5 \
#                        + layer_output_t_star * t_positions.unsqueeze(2) * 0.5
        
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
    
    def feed_forward_chunk_t_star(self, attention_output):
        intermediate_output = self.intermediate_t_star(attention_output)
        layer_output = self.output_t_star(intermediate_output, attention_output)
        return layer_output
    
    
class RobertaLayer_auto(nn.Module):
    def __init__(self, config, VDC_auto, VIC_auto, num_auto_layers, head_wise, auto_VDC_k, a_pooler):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = RobertaAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = RobertaAttention(config, position_embedding_type="absolute")
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)
        
        self.VDC_auto = VDC_auto
        self.VIC_auto = VIC_auto
        self.num_auto_layers = num_auto_layers
        self.head_wise = head_wise
        self.a_pooler = a_pooler
        
        self.VDC_gate, self.VDC_gate_2, self.VIC_gate, self.VIC_gate_2 = None, None, None, None
        
        if num_auto_layers == 0:
            if VDC_auto == True:
                if head_wise == True:
                    self.VDC_gate = nn.Parameter(torch.FloatTensor(12*(auto_VDC_k+2)))
                else:
                    self.VDC_gate = nn.Parameter(torch.FloatTensor(auto_VDC_k+2))
            
            if VIC_auto == True:
                if head_wise == True:
                    self.VIC_gate = nn.Parameter(torch.FloatTensor(12*4))
                else:
                    self.VIC_gate = nn.Parameter(torch.FloatTensor(4))
                    
        else:
            if VDC_auto == True:
                if head_wise == True:
                    if a_pooler in ['t_max_concat', 't_avg_concat', 's_g_a_concat']:
                        if num_auto_layers == 1:
                            self.VDC_gate = nn.Linear(3*768, 12*(auto_VDC_k+2))
                        elif num_auto_layers == 2:
                            self.VDC_gate = nn.Linear(3*768, 768)
                            self.VDC_gate_2 = nn.Linear(768, 12*(auto_VDC_k+2))
                            
                    elif a_pooler in ['s_g_concat']:
                        if num_auto_layers == 1:
                            self.VDC_gate = nn.Linear(2*768, 12*(auto_VDC_k+2))
                        elif num_auto_layers == 2:
                            self.VDC_gate = nn.Linear(2*768, 768)
                            self.VDC_gate_2 = nn.Linear(768, 12*(auto_VDC_k+2))
                            
                    elif a_pooler in ['t_avg_only', 't_avg_all', 't_max_only', 't_max_all', 'a', 's_g_a_avg']:
                        if num_auto_layers == 1:
                            self.VDC_gate = nn.Linear(768, 12*(auto_VDC_k+2))
                        elif num_auto_layers == 2:
                            self.VDC_gate = nn.Linear(768, 768)
                            self.VDC_gate_2 = nn.Linear(768, 12*(auto_VDC_k+2))
                else:
                    if a_pooler in ['t_max_concat', 't_avg_concat', 's_g_a_concat']:
                        if num_auto_layers == 1:
                            self.VDC_gate = nn.Linear(3*768, auto_VDC_k+2)
                        elif num_auto_layers == 2:
                            self.VDC_gate = nn.Linear(3*768, 768)
                            self.VDC_gate_2 = nn.Linear(768, auto_VDC_k+2)
                            
                    elif a_pooler in ['s_g_concat']:
                        if num_auto_layers == 1:
                            self.VDC_gate = nn.Linear(2*768, auto_VDC_k+2)
                        elif num_auto_layers == 2:
                            self.VDC_gate = nn.Linear(2*768, 768)
                            self.VDC_gate_2 = nn.Linear(768, auto_VDC_k+2)
                            
                    elif a_pooler in ['t_avg_only', 't_avg_all', 't_max_only', 't_max_all', 'a', 's_g_a_avg']:
                        if num_auto_layers == 1:
                            self.VDC_gate = nn.Linear(768, auto_VDC_k+2)
                        elif num_auto_layers == 2:
                            self.VDC_gate = nn.Linear(768, 768)
                            self.VDC_gate_2 = nn.Linear(768, auto_VDC_k+2)
                        
            if VIC_auto == True:
                if head_wise == True:
                    if a_pooler in ['t_max_concat', 't_avg_concat', 's_g_a_concat']:
                        if num_auto_layers == 1:
                            self.VIC_gate = nn.Linear(3*768, 12*4)
                        elif num_auto_layers == 2:
                            self.VIC_gate = nn.Linear(3*768, 768)
                            self.VIC_gate_2 = nn.Linear(768, 12*4)
                        elif num_auto_layers == 3:
                            self.VIC_gate = nn.Linear(3*768, 768)
                            self.VIC_gate_2 = nn.Linear(768, 768)
                            self.VIC_gate_3 = nn.Linear(768, 48)
                            
                    elif a_pooler in ['s_g_concat']:
                        if num_auto_layers == 1:
                            self.VIC_gate = nn.Linear(2*768, 12*4)
                        elif num_auto_layers == 2:
                            self.VIC_gate = nn.Linear(2*768, 768)
                            self.VIC_gate_2 = nn.Linear(768, 12*4)
                            
                    elif a_pooler in ['t_avg_only', 't_avg_all', 't_max_only', 't_max_all', 'a', 's_g_a_avg']:
                        if num_auto_layers == 1:
                            self.VIC_gate = nn.Linear(768, 12*4)
                        elif num_auto_layers == 2:
                            self.VIC_gate = nn.Linear(768, 768)
                            self.VIC_gate_2 = nn.Linear(768, 12*4)
                        elif num_auto_layers == 3:
                            self.VIC_gate = nn.Linear(768, 768)
                            self.VIC_gate_2 = nn.Linear(768, 768)
                            self.VIC_gate_3 = nn.Linear(768, 12*4)
                else:
                    if a_pooler in ['t_max_concat', 't_avg_concat', 's_g_a_concat']:
                        if num_auto_layers == 1:
                            self.VIC_gate = nn.Linear(3*768, 4)
                        elif num_auto_layers == 2:
                            self.VIC_gate = nn.Linear(3*768, 768)
                            self.VIC_gate_2 = nn.Linear(768, 4)
                        elif num_auto_layers == 3:
                            self.VIC_gate = nn.Linear(3*768, 768)
                            self.VIC_gate_2 = nn.Linear(768, 768)
                            self.VIC_gate_3 = nn.Linear(768, 4)
                            
                    elif a_pooler in ['s_g_concat']:
                        if num_auto_layers == 1:
                            self.VIC_gate = nn.Linear(2*768, 4)
                        elif num_auto_layers == 2:
                            self.VIC_gate = nn.Linear(2*768, 768)
                            self.VIC_gate_2 = nn.Linear(768, 4)
                            
                    elif a_pooler in ['t_avg_only', 't_avg_all', 't_max_only', 't_max_all', 'a', 's_g_a_avg']:
                        if num_auto_layers == 1:
                            self.VIC_gate = nn.Linear(768, 4)
                        elif num_auto_layers == 2:
                            self.VIC_gate = nn.Linear(768, 768)
                            self.VIC_gate_2 = nn.Linear(768, 4)
                        elif num_auto_layers == 3:
                            self.VIC_gate = nn.Linear(768, 768)
                            self.VIC_gate_2 = nn.Linear(768, 768)
                            self.VIC_gate_3 = nn.Linear(768, 4)
            
        print('='*77)
        print('Automation info: ')
        print('-'*77)
        print('VDC_auto: ', self.VDC_auto)
        print('VIC_auto: ', self.VIC_auto)
        print('head_wise: ', self.head_wise)
        print('num_auto_layers: ', self.num_auto_layers)
        print('auto_VDC_k: ', auto_VDC_k)
        print('a_pooler: ', self.a_pooler)
        print('-'*77)
        
        if self.num_auto_layers >=1 :
            if self.VDC_gate != None:
                print('self.VDC_gate.weight.size(): ', self.VDC_gate.weight.size())
            else:
                print('self.VDC_gate: ', self.VDC_gate)

            if self.VDC_gate_2 != None:
                print('self.VDC_gate_2.weight.size(): ', self.VDC_gate_2.weight.size())
            else:
                print('self.VDC_gate_2: ', self.VDC_gate_2)

            if self.VIC_gate != None:
                print('self.VIC_gate.weight.size(): ', self.VIC_gate.weight.size())
            else:
                print('self.VIC_gate: ', self.VIC_gate)

            if self.VIC_gate_2 != None:
                print('self.VIC_gate_2.weight.size(): ', self.VIC_gate_2.weight.size())
            else:
                print('self.VIC_gate_2: ', self.VIC_gate_2)
            print('-'*77)
            
        else:
            if self.VDC_gate != None:
                print('self.VDC_gate.data.size(): ', self.VDC_gate.data.size())
            else:
                print('self.VDC_gate: ', self.VDC_gate)

            if self.VIC_gate != None:
                print('self.VIC_gate.data.size(): ', self.VIC_gate.data.size())
            else:
                print('self.VIC_gate: ', self.VIC_gate)

            print('-'*77)
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        if num_auto_layers == 0:
            if VDC_auto == True:
                self.VDC_gate.data.normal_(mean=0.0, std=config.initializer_range)
                
        elif num_auto_layers == 1:
            if VDC_auto == True:
                self.VDC_gate.weight.data.normal_(mean=0.0, std=config.initializer_range)
                if self.VDC_gate.bias is not None:
                    self.VDC_gate.bias.data.zero_()
            if VIC_auto == True:
                self.VIC_gate.weight.data.normal_(mean=0.0, std=config.initializer_range)
                if self.VIC_gate.bias is not None:
                    self.VIC_gate.bias.data.zero_()
                    
        elif num_auto_layers == 2:
            if VDC_auto == True:
                self.VDC_gate.weight.data.normal_(mean=0.0, std=config.initializer_range)
                if self.VDC_gate.bias is not None:
                    self.VDC_gate.bias.data.zero_()
                    
                self.VDC_gate_2.weight.data.normal_(mean=0.0, std=config.initializer_range)
                if self.VDC_gate_2.bias is not None:
                    self.VDC_gate_2.bias.data.zero_()
                    
            if VIC_auto == True:
                self.VIC_gate.weight.data.normal_(mean=0.0, std=config.initializer_range)
                if self.VIC_gate.bias is not None:
                    self.VIC_gate.bias.data.zero_()
                    
                self.VIC_gate_2.weight.data.normal_(mean=0.0, std=config.initializer_range)
                if self.VIC_gate_2.bias is not None:
                    self.VIC_gate_2.bias.data.zero_()
                    
        elif num_auto_layers == 3:
            if VIC_auto == True:
                self.VIC_gate.weight.data.normal_(mean=0.0, std=config.initializer_range)
                if self.VIC_gate.bias is not None:
                    self.VIC_gate.bias.data.zero_()
                    
                self.VIC_gate_2.weight.data.normal_(mean=0.0, std=config.initializer_range)
                if self.VIC_gate_2.bias is not None:
                    self.VIC_gate_2.bias.data.zero_()
                    
                self.VIC_gate_3.weight.data.normal_(mean=0.0, std=config.initializer_range)
                if self.VIC_gate_3.bias is not None:
                    self.VIC_gate_3.bias.data.zero_()
            
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        target_idx = None,
        gcls_update = False,
        g_config = None,
        VDC_info = None,
        automation_visualization = None,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        
        ## Automating VDC & VIC
        mask = (VDC_info == 1).long().unsqueeze(2)
        
        #################################
        if self.a_pooler == 't_avg_only':
            pooled = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1)
            V_input = self.dropout(pooled)
        elif self.a_pooler == 't_avg_all':
            mask[:, :2] = 1 # Allows considering s and g
            pooled = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1)
            V_input = self.dropout(pooled)
        elif self.a_pooler == 't_max_only':
            index_2 = (VDC_info != 1).long()*-float('inf')
            index_2[index_2 != index_2] = 0 # change 0 * -inf = nan to zero
            mask_2 = index_2.unsqueeze(2)
            pooled = torch.max(hidden_states*mask + mask_2, dim = 1)[0]
            V_input = self.dropout(pooled)
        elif self.a_pooler == 't_max_all':
            mask[:, :2] = 1 # Allows considering s and g
            index_2 = (VDC_info != 1).long()*-float('inf')
            index_2[:,:2] = 0 # Allows considering s and g
            index_2[index_2 != index_2] = 0 # change 0 * -inf = nan to zero
            mask_2 = index_2.unsqueeze(2)
            pooled = torch.max(hidden_states*mask + mask_2, dim = 1)[0]
            V_input = self.dropout(pooled)
        elif self.a_pooler == 't_avg_concat':
            pooled = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1)
            concat_vector = torch.cat((hidden_states[:,:2], pooled.unsqueeze(1)), dim = 1).reshape(-1, 3*768)
            V_input = self.dropout(concat_vector)
        elif self.a_pooler == 't_max_concat':
            index_2 = (VDC_info != 1).long()*-float('inf')
            index_2[index_2 != index_2] = 0 # change 0 * -inf = nan to zero
            mask_2 = index_2.unsqueeze(2)
            pooled = torch.max(hidden_states*mask + mask_2, dim = 1)[0]
            concat_vector = torch.cat((hidden_states[:,:2], pooled.unsqueeze(1)), dim = 1).reshape(-1, 3*768)
            V_input = self.dropout(concat_vector)
            
        elif self.a_pooler == 's_g_concat':
            V_input = hidden_states[:, :2].view(-1, 2*768)
            
        elif self.a_pooler == 'a':
            V_input = hidden_states[:, 2].view(-1, 768)
            
        elif self.a_pooler == 's_g_a_concat':
            V_input = hidden_states[:, :3].view(-1, 3*768)
            
        elif self.a_pooler == 's_g_a_avg':
            V_input = torch.mean(hidden_states[:, :3], dim= 1)
        #################################
        
        VDC_gate, VIC_gate = None, None
        if self.VDC_auto == True:
            if self.num_auto_layers == 0:
#                 w = self.VDC_gate.data
#                 w = w.clamp(0.0,1.0)
                VDC_gate = self.VDC_gate.unsqueeze(0).repeat(V_input.size(0),1)
            elif self.num_auto_layers == 1:
                VDC_gate = self.VDC_gate(V_input)
            elif self.num_auto_layers == 2:
                VDC_gate = self.VDC_gate(V_input)
                VDC_gate = self.tanh(VDC_gate)
                VDC_gate = self.VDC_gate_2(VDC_gate)
                
        if self.VIC_auto == True:
            if self.num_auto_layers == 0:
                VIC_gate = self.VIC_gate
                VIC_gate = self.sigmoid(VIC_gate)
            elif self.num_auto_layers == 1:
                VIC_gate = self.VIC_gate(V_input)
                VIC_gate = self.sigmoid(VIC_gate)
            elif self.num_auto_layers == 2:
                VIC_gate = self.VIC_gate(V_input)
                VIC_gate = self.tanh(VIC_gate)
                VIC_gate = self.VIC_gate_2(VIC_gate)
                VIC_gate = self.sigmoid(VIC_gate)
                
        # 사실 위에 두 개 합치는게 낫긴 함
        
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            target_idx = target_idx,
            gcls_update = gcls_update,
            g_config = g_config,
            VIC_gate = VIC_gate,
            VDC_gate = VDC_gate,
            VDC_info = VDC_info,
            head_wise = self.head_wise,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)
        
        vis_vdc, vis_vic = None, None
        if automation_visualization == True:
            if self.VDC_auto == True:
                if self.head_wise == True:
                    vis_vdc = nn.functional.softmax(VDC_gate.clone().reshape(VDC_info.size(0), 12, -1), dim = -1)
                else:
                    vis_vdc = nn.functional.softmax(VDC_gate.clone(), dim = -1)
                    
            if self.VIC_auto == True:
                if self.head_wise == True:
                    vis_vic = VIC_gate.clone().reshape(VDC_info.size(0), 12, 4)
                else:
                    vis_vic = VIC_gate.clone()
            return outputs, (vis_vdc, vis_vic)
        
        else:
            return outputs, None

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
    
    
class RobertaLayer_auto_2(nn.Module):
    def __init__(self, config, VDC_auto, VIC_auto, num_auto_layers, head_wise, auto_VDC_k, a_pooler):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = RobertaAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = RobertaAttention(config, position_embedding_type="absolute")
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)
            
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        target_idx = None,
        gcls_update = False,
        g_config = None,
        VDC_info = None,
        automation_visualization = None,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            target_idx = target_idx,
            gcls_update = gcls_update,
            g_config = g_config,
            VIC_gate = VIC_gate,
            VDC_gate = VDC_gate,
            VDC_info = VDC_info,
            head_wise = self.head_wise,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)
        
        return outputs, None

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
    
# Copied from transformers.models.bert.modeling_bert.BertEncoder with Bert->Roberta
class RobertaEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([RobertaLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        VDC_info = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        
        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    VDC_info = VDC_info,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

    
class RobertaEncoder_t_star(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
#         self.layer = nn.ModuleList([RobertaLayer_t_star(config) for _ in range(config.num_hidden_layers)])
        
        self.layer = nn.ModuleList([RobertaLayer_t_star(config) for _ in range(3)]) \
                    +nn.ModuleList([RobertaLayer(config) for _ in range(9)])
        
        
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        VDC_info = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        
        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    VDC_info = VDC_info,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

    
class RobertaEncoder_gcls(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([RobertaLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        VDC_info = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        
#         all_s_outputs = () if sg_output else None
#         all_g_outputs = () if sg_output else None
        
        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                
#             if sg_output:
#                 all_s_outputs = all_s_outputs + (hidden_states[:,0],)
#                 all_g_outputs = all_g_outputs + (hidden_states[:,1],)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                att_mask = attention_mask[i].clone()
                
#                 if i > 700000 :
#                     gcls_update = True
#                 else:
#                     gcls_update = False
                gcls_update = False
                
                layer_outputs = layer_module(
                    hidden_states,
                    att_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    VDC_info = VDC_info,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
#         if sg_output:
#             all_s_outputs = all_s_outputs + (hidden_states[:,0],)
#             all_g_outputs = all_g_outputs + (hidden_states[:,1],)
            
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
    
class RobertaEncoder_gcls_auto(nn.Module):
    def __init__(self, config, VDC_auto, VIC_auto, num_auto_layers, head_wise, auto_VDC_k, a_pooler):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([RobertaLayer_auto(config, VDC_auto = VDC_auto, VIC_auto = VIC_auto, 
                                                      num_auto_layers = num_auto_layers, head_wise = head_wise,
                                                      auto_VDC_k = auto_VDC_k, a_pooler = a_pooler) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
        
        self.VDC_auto = VDC_auto
        self.VIC_auto = VIC_auto
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        VDC_info = None,
        automation_visualization = None,
        sg_output = False,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        
        all_s_outputs = () if sg_output else None
        all_g_outputs = () if sg_output else None

        next_decoder_cache = () if use_cache else None
        
        auto_VDC_results = None
        auto_VIC_results = None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                
            if sg_output:
                all_s_outputs = all_s_outputs + (hidden_states[:,0],)
                all_g_outputs = all_g_outputs + (hidden_states[:,1],)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                if self.VDC_auto == False:
                    att_mask = attention_mask[i].clone()
                else:
                    att_mask = None
                
                gcls_update = False
                layer_outputs, visualization_results = layer_module(
                    hidden_states,
                    att_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    gcls_update = gcls_update,
                    VDC_info = VDC_info,
                    automation_visualization = automation_visualization,
                )
                
            
            if automation_visualization:
                if self.VDC_auto:
                    if i == 0:
                        auto_VDC_results = visualization_results[0].unsqueeze(0)
                    else:
                        auto_VDC_results = torch.cat((auto_VDC_results, visualization_results[0].unsqueeze(0)), dim = 0)
                if self.VIC_auto:
                    if i == 0:
                        auto_VIC_results = visualization_results[1].unsqueeze(0)
                    else:
                        auto_VIC_results = torch.cat((auto_VIC_results, visualization_results[1].unsqueeze(0)), dim = 0)
            
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            
        if sg_output:
            all_s_outputs = all_s_outputs + (hidden_states[:,0],)
            all_g_outputs = all_g_outputs + (hidden_states[:,1],)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        ), all_s_outputs, all_g_outputs, (auto_VDC_results, auto_VIC_results)
    

# Copied from transformers.models.bert.modeling_bert.BertPooler
class RobertaPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class RobertaPooler_TD(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, VDC_info=None) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        
#         # 직접 검토 (1):
#         target_in_sent_embed = torch.zeros(hidden_states.size()[0], hidden_states.size()[-1]).to(hidden_states.device)
#         for i in range(hidden_states.size()[0]):
#             x = (extended_attention_mask_pooler[12][i][0][1][:] == 0.0).nonzero(as_tuple=True)[0]
            
#             # 가장 기본적인 RoBERTa-TD
#             target_embed = hidden_states[i][x[0]:x[-1]+1]
#             # 1. RoBERTa-TD-avg
# #             target_in_sent_embed[i] = target_embed.mean(dim = 0)
#             # 2. RoBERTa-TD-max
#             target_in_sent_embed[i] = torch.max(target_embed, dim=0)[0]
        
        #########################
        # avg pooling
        index = VDC_info == 1
        index = index.long()
        
        mask = index.unsqueeze(2)
        pooled = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1)
        
        # max_pooling
#         index = VDC_info == 1
#         index = index.long()
#         index_2 = (VDC_info != 1).long()*-float('inf')
#         index_2[index_2 != index_2] = 0 # change 0 * -inf = nan to zero
#         mask = index.unsqueeze(2)
#         mask_2 = index_2.unsqueeze(2)
#         pooled = torch.max(hidden_states*mask + mask_2, dim = 1)[0]
        #########################
    
#         for i in range(target_in_sent_embed.size(0)):
#             assert torch.sum((target_in_sent_embed[i] - pooled[i])**2) < 0.000001 -> 이거 코딩 수준 실화? ㅋㅋ
        
        ##### s랑 t 같이 하는거 체크하기 위해 여기에 잠깐 도입
        st_vector = torch.cat((hidden_states[:, 0].unsqueeze(1), pooled.unsqueeze(1)), dim=1)
        cat = torch.mean(st_vector, dim = 1)
        pooled_output = self.dense(cat)
        pooled_output = self.activation(pooled_output)
        
        attention_scores = torch.matmul(pooled_output.unsqueeze(1), st_vector.transpose(1,2))
        attention_scores = attention_scores.view(-1,2)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        pooled_output = torch.matmul(attention_probs.unsqueeze(1), st_vector).view(-1, 768)
        
        #####
        
#         pooled = self.dropout(pooled)
        
#         pooled_output = self.dense(pooled)
#         pooled_output = self.activation(pooled_output)
        
        return pooled_output
    
    
class RobertaPooler_gcls(nn.Module):
    def __init__(self, config, g_pooler):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.g_pooler = g_pooler
        self.dense_2 = None
        if g_pooler in ['s_g_t_max_concat', 's_g_t_avg_concat']:
            self.dense_2 = nn.Linear(3*config.hidden_size, config.hidden_size)
        elif g_pooler in ['s_g_concat']:
            self.dense_2 = nn.Linear(2*config.hidden_size, config.hidden_size)
        elif g_pooler in ['s_g_att', 's_g_t_avg']:
            self.dense_2 = nn.Linear(config.hidden_size, config.hidden_size)
        
        if self.dense_2 != None:
            self.dense_2.weight.data.normal_(mean=0.0, std=config.initializer_range)
            if self.dense_2.bias is not None:
                self.dense_2.bias.data.zero_()
        
    def forward(self, hidden_states: torch.Tensor, VDC_info=None) -> torch.Tensor:
        if self.g_pooler == 's':
            cat = hidden_states[:,0]
            pooled_output = self.dense(cat)
            final_output = self.activation(pooled_output)
        elif self.g_pooler == 's_g_avg':
            first_second_token_tensor = hidden_states[:, :2]
            cat = torch.mean(first_second_token_tensor, dim = 1)
            pooled_output = self.dense(cat)
            final_output = self.activation(pooled_output)
        elif self.g_pooler == 's_g_g_avg':    
            sec_g_token = hidden_states[range(hidden_states.size(0)), sgg_info]
            sgg_tokens = torch.cat((hidden_states[:, :2], sec_g_token.unsqueeze(1)), dim=1)
            cat = torch.mean(sgg_tokens, dim = 1)
            pooled_output = self.dense(cat)
            final_output = self.activation(pooled_output)
        elif self.g_pooler == 's_g_max':
            first_second_token_tensor = hidden_states[:, :2]
            cat = torch.max(first_second_token_tensor, dim=1)[0]
            pooled_output = self.dense(cat)
            final_output = self.activation(pooled_output)
            
        elif self.g_pooler == 's_g_att':
            
            first_second_token_tensor = hidden_states[:, :2]
            cat = torch.mean(first_second_token_tensor, dim=1)
            pooled_output = self.dense(cat)
            pooled_output = self.activation(pooled_output)

            attention_scores = torch.matmul(pooled_output.unsqueeze(1), first_second_token_tensor.transpose(1,2))
            attention_scores = attention_scores.view(-1,2)
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)
            final_output = torch.matmul(attention_probs.unsqueeze(1), first_second_token_tensor).view(-1, 768)
#             final_output = self.activation(final_output)
            
            

        elif self.g_pooler == 's_g_t_att':
            index = VDC_info == 1
            index = index.long()
            mask = index.unsqueeze(2)
            t_vector = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1)
            
            sgt_vector = torch.cat((hidden_states[:, :2], t_vector.unsqueeze(1)), dim=1)
            cat = torch.mean(sgt_vector, dim=1)
            pooled_output = self.dense(cat)
            pooled_output = self.activation(pooled_output)

            attention_scores = torch.matmul(pooled_output.unsqueeze(1), sgt_vector.transpose(1,2))
            attention_scores = attention_scores.view(-1,3)
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)
            final_output = torch.matmul(attention_probs.unsqueeze(1), sgt_vector).view(-1, 768)
            final_output = self.activation(final_output)
            
        elif self.g_pooler == 's_g_t_max_att':
            index = VDC_info == 1
            index = index.long()
            index_2 = (VDC_info != 1.0).long()*-float('inf')
            index_2[index_2 != index_2] = 0 # change 0 * -inf = nan to zero
            mask = index.unsqueeze(2)
            mask_2 = index_2.unsqueeze(2)
            t_vector = torch.max(hidden_states*mask + mask_2, dim = 1)[0]
            
            sgt_vector = torch.cat((hidden_states[:, :2], t_vector.unsqueeze(1)), dim=1)
            cat = torch.mean(sgt_vector, dim=1)
            pooled_output = self.dense(cat)
            pooled_output = self.activation(pooled_output)

            attention_scores = torch.matmul(pooled_output.unsqueeze(1), sgt_vector.transpose(1,2))
            attention_scores = attention_scores.view(-1,3)
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)
            final_output = torch.matmul(attention_probs.unsqueeze(1), sgt_vector).view(-1, 768)
            final_output = self.activation(final_output)
        
        elif self.g_pooler == 's_g_t_max':
            index = VDC_info == 1
            index = index.long()
            index[:, :2] = 1 # Allows considering s and g
            index_2 = (VDC_info != 1.0).long()*-float('inf')
            index_2[index_2 != index_2] = 0 # change 0 * -inf = nan to zero
            mask = index.unsqueeze(2)
            mask_2 = index_2.unsqueeze(2)
            t_vector = torch.max(hidden_states*mask + mask_2, dim = 1)[0]
            
            pooled = self.dropout(t_vector)
        
            pooled_output = self.dense(pooled)
            final_output = self.activation(pooled_output)
            
        elif self.g_pooler == 'g_t_max':
            index = VDC_info == 1
            index = index.long()
            index[:, 1] = 1 # g
            index_2 = (VDC_info != 1.0).long()*-float('inf')
            index_2[index_2 != index_2] = 0 # change 0 * -inf = nan to zero
            mask = index.unsqueeze(2)
            mask_2 = index_2.unsqueeze(2)
            t_vector = torch.max(hidden_states*mask + mask_2, dim = 1)[0]
            
            pooled = self.dropout(t_vector)
        
            pooled_output = self.dense(pooled)
            final_output = self.activation(pooled_output)
            
        elif self.g_pooler == 'g_t_max_att':
            index = VDC_info == 1
            index = index.long()
            index_2 = (VDC_info != 1.0).long()*-float('inf')
            index_2[index_2 != index_2] = 0 # change 0 * -inf = nan to zero
            mask = index.unsqueeze(2)
            mask_2 = index_2.unsqueeze(2)
            t_vector = torch.max(hidden_states*mask + mask_2, dim = 1)[0]
            
            first_second_token_tensor = torch.cat((hidden_states[:, 1].unsqueeze(1), t_vector.unsqueeze(1)), dim=1)
            cat = torch.mean(first_second_token_tensor, dim=1)
            pooled_output = self.dense(cat)
            pooled_output = self.activation(pooled_output)
            
            attention_scores = torch.matmul(pooled_output.unsqueeze(1), first_second_token_tensor.transpose(1,2))
            attention_scores = attention_scores.view(-1,2)
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)
            final_output = torch.matmul(attention_probs.unsqueeze(1), first_second_token_tensor).view(-1, 768)
            
#             final_output = self.activation(final_output)
            
        elif self.g_pooler == 'g_t_avg':
            index = VDC_info == 1
            index = index.long()
            mask = index.unsqueeze(2)
            t_vector = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1)
            
            gt_vector = torch.cat((hidden_states[:, 1].unsqueeze(1), t_vector.unsqueeze(1)), dim=1)
            cat = torch.mean(gt_vector, dim = 1)
            pooled_output = self.dense(cat)
            final_output = self.activation(pooled_output)
            
        elif self.g_pooler == 'g_t_att':
            index = VDC_info == 1
            index = index.long()
            mask = index.unsqueeze(2)
            t_vector = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1)
            
            first_second_token_tensor = torch.cat((hidden_states[:, 1].unsqueeze(1), t_vector.unsqueeze(1)), dim=1)
            cat = torch.mean(first_second_token_tensor, dim=1)
            pooled_output = self.dense(cat)
            pooled_output = self.activation(pooled_output)
            
            attention_scores = torch.matmul(pooled_output.unsqueeze(1), first_second_token_tensor.transpose(1,2))
            attention_scores = attention_scores.view(-1,2)
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)
            final_output = torch.matmul(attention_probs.unsqueeze(1), first_second_token_tensor).view(-1, 768)
            
#             final_output = self.dropout(final_output)
            
        elif self.g_pooler == 's_t_att':
            index = VDC_info == 1
            index = index.long()
            mask = index.unsqueeze(2)
            t_vector = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1)
            
            first_second_token_tensor = torch.cat((hidden_states[:, 0].unsqueeze(1), t_vector.unsqueeze(1)), dim=1)
            cat = torch.mean(first_second_token_tensor, dim=1)
            pooled_output = self.dense(cat)
            pooled_output = self.activation(pooled_output)
            
            attention_scores = torch.matmul(pooled_output.unsqueeze(1), first_second_token_tensor.transpose(1,2))
            attention_scores = attention_scores.view(-1,2)
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)
            final_output = torch.matmul(attention_probs.unsqueeze(1), first_second_token_tensor).view(-1, 768)
                
        elif self.g_pooler == 's_g_concat':
            first_second_token_tensor = hidden_states[:, :2].reshape(-1, 2*768)
            pooled_output = self.dense_2(first_second_token_tensor)
            final_output = self.activation(pooled_output)
        elif self.g_pooler == 's_g_t_avg_concat':  # concatanating s, g, and t_avg
            first_second_token_tensor = hidden_states[:, :2]
            
            index = VDC_info == 1
            index = index.long()
            mask = index.unsqueeze(2)
            pooled = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1)
            
            final_output = torch.cat((first_second_token_tensor, pooled.unsqueeze(1)), dim = 1).reshape(-1,3*768)
            final_output = self.dropout(final_output)
            final_output = self.dense_2(final_output)
            final_output = self.activation(final_output)
            
        elif self.g_pooler == 's_g_t_avg':  # averaging s, g, and t_avg
            first_second_token_tensor = hidden_states[:, :2]
            
            index = VDC_info == 1
            index = index.long()
            mask = index.unsqueeze(2)
            pooled = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1)
            cat = torch.cat((first_second_token_tensor, pooled.unsqueeze(1)), dim = 1)
            cat_avg = torch.mean(cat, dim = 1)
            pooled_output = self.dense(cat_avg)
            pooled_output = self.dropout(pooled_output)
            final_output = self.activation(pooled_output)
          
        elif self.g_pooler == 's_g_t_max_concat':  # concatanating s, g, and t_max
            first_second_token_tensor = hidden_states[:, :2]
            index = VDC_info == 1
            index = index.long()
            index_2 = (VDC_info != 1.0).long()*-float('inf')
            index_2[index_2 != index_2] = 0 # change 0 * -inf = nan to zero
            mask = index.unsqueeze(2)
            mask_2 = index_2.unsqueeze(2)
            pooled = torch.max(hidden_states*mask + mask_2, dim = 1)[0]
            
            final_output = torch.cat((first_second_token_tensor, pooled.unsqueeze(1)), dim = 1).reshape(-1,3*768)
            final_output = self.dropout(final_output)
            final_output = self.dense_2(final_output)
            final_output = self.activation(final_output)
            
        
            
        elif self.g_pooler == 'g':
            first_token_tensor = hidden_states[:,1]
            pooled_output = self.dense(first_token_tensor)
            final_output = self.activation(pooled_output)
        elif self.g_pooler == 's':
            first_token_tensor = hidden_states[:,0]
            pooled_output = self.dense(first_token_tensor)
            final_output = self.activation(pooled_output)
        
        elif self.g_pooler == 't_avg_only':
            index = VDC_info == 1
            index = index.long()

            mask = index.unsqueeze(2)
            pooled = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1)
            
            pooled = self.dropout(pooled)
            pooled_output = self.dense(pooled)
            final_output = self.activation(pooled_output)
            
        elif self.g_pooler == 't_avg_all':
            # s and g vectors
            first_second_token_tensor = hidden_states[:, :2]
            
            index = VDC_info == 1.0
            index[:,:2] = 1
            index = index.long()

            mask = index.unsqueeze(2)
            pooled = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1)
                
            pooled = self.dropout(pooled)
            pooled_output = self.dense(pooled)
            final = self.activation(pooled_output)
            
        elif self.g_pooler == 't_avg_concat':
            first_second_token_tensor = hidden_states[:, :2]
            
            index = VDC_info == 1
            index = index.long()
            
            mask = index.unsqueeze(2)
            pooled = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1)
            
            target_in_sent_embed = torch.cat((first_second_token_tensor, pooled.unsqueeze(1)), dim = 1).reshape(-1,3*768)
            target_in_sent_embed = self.dropout(target_in_sent_embed)
            pooled_output = self.dense_2(target_in_sent_embed)
            final_output = self.activation(pooled_output)
            
        elif self.g_pooler == 't_max_avg':
            # s and g vectors
            first_second_token_tensor = hidden_states[:, :2]
            
            # target token vectors
            target_in_sent_embed = torch.zeros(hidden_states.size()[0], hidden_states.size()[-1]).to(hidden_states.device)
            for i in range(hidden_states.size()[0]):
                x = (VDC_info == 1).nonzero(as_tuple=True)[0]

                # 가장 기본적인 RoBERTa-TD
                target_embed = hidden_states[i][x[0]:x[-1]+1]
                target_embed = torch.cat((first_second_token_tensor[i], torch.max(target_embed, dim=0)[0].unsqueeze(0)), dim = 0)
                target_in_sent_embed[i] = target_embed.mean(dim = 0)

            target_in_sent_embed = self.dropout(target_in_sent_embed)
            pooled_output = self.dense(target_in_sent_embed)
            final_output = self.activation(pooled_output)
            
        elif self.g_pooler == 't_max_max':
            # s and g vectors
            first_second_token_tensor = hidden_states[:, :2]
            
            # target token vectors
            target_in_sent_embed = torch.zeros(hidden_states.size()[0], hidden_states.size()[-1]).to(hidden_states.device)
            for i in range(hidden_states.size()[0]):
                x = (VDC_info == 1).nonzero(as_tuple=True)[0]

                # 가장 기본적인 RoBERTa-TD
                target_embed = hidden_states[i][x[0]:x[-1]+1]
                target_embed = torch.cat((first_second_token_tensor[i], target_embed), dim=0)
                target_in_sent_embed[i] = torch.max(target_embed, dim=0)[0]

            target_in_sent_embed = self.dropout(target_in_sent_embed)
            pooled_output = self.dense(target_in_sent_embed)
            final_output = self.activation(pooled_output)
            
        elif self.g_pooler == 't_max_only':
            index = VDC_info == 1
            index = index.long()
            index_2 = (VDC_info != 1).long()*-float('inf')
            index_2[index_2 != index_2] = 0 # change 0 * -inf = nan to zero
            mask = index.unsqueeze(2)
            mask_2 = index_2.unsqueeze(2)
            pooled = torch.max(hidden_states*mask + mask_2, dim = 1)[0]
            
            pooled = self.dropout(pooled)
        
            pooled_output = self.dense(pooled)
            final_output = self.activation(pooled_output)
            
        elif self.g_pooler == 't_max_concat':
            # s and g vectors
            first_second_token_tensor = hidden_states[:, :2]
            
            index = VDC_info == 1
            index_2 = (VDC_info != 1).long()*-float('inf')
            index_2[index_2 != index_2] = 0
            index = index.long()
            mask = index.unsqueeze(2)
            mask_2 = index_2.unsqueeze(2)
            pooled = torch.max(hidden_states*mask + mask_2, dim = 1)[0]
            
            target_in_sent_embed = torch.cat((first_second_token_tensor, pooled.unsqueeze(1)), dim = 1).reshape(-1,3*768)
            target_in_sent_embed = self.dropout(target_in_sent_embed)
            pooled_output = self.dense_2(target_in_sent_embed)
            final_output = self.activation(pooled_output)
            
        elif self.g_pooler == 't_all_att':
            first_second_token_tensor = hidden_states[:, :2]
            target_in_sent_embed = torch.zeros(hidden_states.size()[0], hidden_states.size()[-1]).to(hidden_states.device)
            
            for i in range(hidden_states.size()[0]):
                x = (VDC_info == 1).nonzero(as_tuple=True)[0]

                # 가장 기본적인 RoBERTa-TD
                target_embed = hidden_states[i][x[0]:x[-1]+1]
                
                y = torch.cat((first_second_token_tensor[i], target_embed), dim = 0)
                y_dim = y.size(0)
                cat = torch.mean(y, dim=0)
                pooled_output = self.dense(cat)
                pooled_output = self.activation(pooled_output)
                
                attention_scores = torch.matmul(pooled_output.unsqueeze(0), y.transpose(0,1))
                attention_scores = attention_scores.view(-1, y_dim)
                attention_probs = nn.functional.softmax(attention_scores, dim=-1)
                target_in_sent_embed[i] = torch.matmul(attention_probs.unsqueeze(0), y).view(-1, 768)
            
            final_output = self.dropout(target_in_sent_embed)
            
        return final_output
        
class RobertaPooler_gcls_2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor, g_token_pos) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        
        gcls_tokens_tensor = hidden_states[:,:3]
        cat = torch.mean(gcls_tokens_tensor, dim = 1)
        
        pooled_output = self.dense(cat)
        final_output = self.activation(pooled_output)
        
#         first_second_token_tensor = hidden_states[:, :2]
#         cat = torch.mean(first_second_token_tensor, dim=1)
#         pooled_output = self.dense(cat)
#         pooled_output = self.activation(pooled_output)
        
#         attention_scores = torch.matmul(pooled_output.unsqueeze(1), first_second_token_tensor.transpose(1,2))
#         attention_scores = attention_scores.view(-1,2)
#         attention_probs = nn.functional.softmax(attention_scores, dim=-1)
#         final_output = torch.matmul(attention_probs.unsqueeze(1), first_second_token_tensor).view(-1, 768)
        
        return final_output
    
class RobertaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = RobertaConfig
    base_model_prefix = "roberta"
    supports_gradient_checkpointing = True

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, RobertaEncoder):
            module.gradient_checkpointing = value

    def update_keys_to_ignore(self, config, del_keys_to_ignore):
        """Remove some keys from ignore list"""
        if not config.tie_word_embeddings:
            # must make a new list, or the class variable gets modified!
            self._keys_to_ignore_on_save = [k for k in self._keys_to_ignore_on_save if k not in del_keys_to_ignore]
            self._keys_to_ignore_on_load_missing = [
                k for k in self._keys_to_ignore_on_load_missing if k not in del_keys_to_ignore
            ]


ROBERTA_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`RobertaConfig`]): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

ROBERTA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`RobertaTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare RoBERTa Model transformer outputting raw hidden-states without any specific head on top.",
    ROBERTA_START_DOCSTRING,
)

class RobertaModel(RobertaPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in *Attention is
    all you need*_ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
    Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.

    .. _*Attention is all you need*: https://arxiv.org/abs/1706.03762

    """

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->Roberta
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoder(config)

        self.pooler = RobertaPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        
        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)
        
        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


@add_start_docstrings(
    """RoBERTa Model with a `language modeling` head on top for CLM fine-tuning.""", ROBERTA_START_DOCSTRING
)

class RobertaModel_TD(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->Roberta
    def __init__(self, config, add_pooling_layer=False):
        super().__init__(config)
        self.config = config

        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoder(config)

        self.pooler = RobertaPooler_TD(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        VDC_info = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        
        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)
        
        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            VDC_info = VDC_info,
        )
        sequence_output = encoder_outputs[0]
        
#         assert self.pooler == None
        pooled_output = self.pooler(sequence_output, VDC_info = VDC_info) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


@add_start_docstrings(
    """RoBERTa Model with a `language modeling` head on top for CLM fine-tuning.""", ROBERTA_START_DOCSTRING
)


class RobertaModel_TD_t_star(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->Roberta
    def __init__(self, config, add_pooling_layer=False):
        super().__init__(config)
        self.config = config

        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoder_t_star(config)

        self.pooler = RobertaPooler_TD(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        VDC_info = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        
        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)
        
        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            VDC_info = VDC_info,
        )
        sequence_output = encoder_outputs[0]
        
#         assert self.pooler == None
        pooled_output = self.pooler(sequence_output, VDC_info = VDC_info) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


@add_start_docstrings(
    """RoBERTa Model with a `language modeling` head on top for CLM fine-tuning.""", ROBERTA_START_DOCSTRING
)


class RobertaModel_gcls(RobertaPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in *Attention is
    all you need*_ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
    Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.

    .. _*Attention is all you need*: https://arxiv.org/abs/1706.03762

    """

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->Roberta
    def __init__(self, config, g_pooler, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoder_gcls(config)
        
        self.pooler = RobertaPooler_gcls(config, g_pooler) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        extended_attention_mask = None,
        VDC_info = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
#         extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None
        
#         original_attention_mask = extended_attention_mask.clone()    # torch.Size([32, 1, 1, 128])
 
#         extended_attention_mask = extended_attention_mask.repeat(1,1,input_ids.size(1),1)    # torch.Size([32, 1, 128, 128])
        
#         extended_attention_mask_ = {}
        
#         for item in layer_L_set:
#             extended_att_mask = extended_attention_mask.clone()
#             for i in range(input_ids.size(0)):
#                 extended_att_mask[i, 0, 1, :] =  (1 - gcls_attention_mask[i][item]) * -10000.0
# #                 extended_att_mask[i,0,0,:2] = torch.tensor([0, -10000.0], dtype =torch.float)

#             extended_attention_mask_[item] = extended_att_mask
        
            
        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            VDC_info = VDC_info,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output, VDC_info = VDC_info) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


@add_start_docstrings(
    """RoBERTa Model with a `language modeling` head on top for CLM fine-tuning.""", ROBERTA_START_DOCSTRING
)

class RobertaModel_gcls_auto(RobertaPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in *Attention is
    all you need*_ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
    Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.

    .. _*Attention is all you need*: https://arxiv.org/abs/1706.03762

    """

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->Roberta
    def __init__(self, config, VDC_auto, VIC_auto, num_auto_layers, head_wise, auto_VDC_k, a_pooler, g_pooler, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoder_gcls_auto(config, VDC_auto = VDC_auto, VIC_auto = VIC_auto, 
                                                num_auto_layers = num_auto_layers, head_wise = head_wise, 
                                                auto_VDC_k = auto_VDC_k, a_pooler = a_pooler)
        
        self.pooler = RobertaPooler_gcls(config, g_pooler) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        extended_attention_mask = None,
        VDC_info = None,
        automation_visualization = None,
        sg_output = False,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
#         extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None
        
#         original_attention_mask = extended_attention_mask.clone()    # torch.Size([32, 1, 1, 128])
 
#         extended_attention_mask = extended_attention_mask.repeat(1,1,input_ids.size(1),1)    # torch.Size([32, 1, 128, 128])
        
#         extended_attention_mask_ = {}
        
#         for item in layer_L_set:
#             extended_att_mask = extended_attention_mask.clone()
#             for i in range(input_ids.size(0)):
#                 extended_att_mask[i, 0, 1, :] =  (1 - gcls_attention_mask[i][item]) * -10000.0
# #                 extended_att_mask[i,0,0,:2] = torch.tensor([0, -10000.0], dtype =torch.float)

#             extended_attention_mask_[item] = extended_att_mask
        
            
        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs, s_, g_, visualization_results = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            VDC_info = VDC_info,
            automation_visualization = automation_visualization,
            sg_output = sg_output,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output, VDC_info = VDC_info) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        ), s_, g_, visualization_results


@add_start_docstrings(
    """RoBERTa Model with a `language modeling` head on top for CLM fine-tuning.""", ROBERTA_START_DOCSTRING
)

class RobertaForCausalLM(RobertaPreTrainedModel):
    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        if not config.is_decoder:
            logger.warning("If you want to use `RobertaLMHeadModel` as a standalone, add `is_decoder=True.`")

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)

        # The LM head weights require special treatment only when they are tied with the word embeddings
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Tuple[Tuple[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).

        Returns:

        Example:

        ```python
        >>> from transformers import RobertaTokenizer, RobertaForCausalLM, RobertaConfig
        >>> import torch

        >>> tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        >>> config = RobertaConfig.from_pretrained("roberta-base")
        >>> config.is_decoder = True
        >>> model = RobertaForCausalLM.from_pretrained("roberta-base", config=config)

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> prediction_logits = outputs.logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=prediction_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past}

    def _reorder_cache(self, past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past


@add_start_docstrings("""RoBERTa Model with a `language modeling` head on top.""", ROBERTA_START_DOCSTRING)
class RobertaForMaskedLM(RobertaPreTrainedModel):
    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `RobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)

        # The LM head weights require special treatment only when they are tied with the word embeddings
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        mask="<mask>",
        expected_output="' Paris'",
        expected_loss=0.1,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        self.bias = self.decoder.bias


@add_start_docstrings(
    """
    RoBERTa Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    ROBERTA_START_DOCSTRING,
)
class RobertaForSequenceClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        config.num_labels = 3
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint="cardiffnlp/twitter-roberta-base-emotion",
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="'optimism'",
        expected_loss=0.08,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

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
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Roberta Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    ROBERTA_START_DOCSTRING,
)

class RobertaForSequenceClassification_gcls(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, g_pooler):
        super().__init__(config)
        config.num_labels = 3
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel_gcls(config, add_pooling_layer=True, g_pooler = g_pooler)
        self.classifier = RobertaClassificationHead_gcls(config)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint="cardiffnlp/twitter-roberta-base-emotion",
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="'optimism'",
        expected_loss=0.08,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        extended_attention_mask = None,
        VDC_info = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        sg_loss = None
        
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            extended_attention_mask = extended_attention_mask,
            VDC_info = VDC_info,
        )
        sequence_output = outputs[1]    # torch.tensor([32,768]), outputs[0].size() = torch.tensor([32,128,768])
        logits = self.classifier(sequence_output)

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
        
#         if sg_output:
#             sg_loss = ()
#             loss_fct_2 = MSELoss()
#             for l in range(12):
#                 sg_loss = sg_loss + (loss_fct_2(s_outputs[l], g_outputs[l]), )
                
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return sg_loss, SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Roberta Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    ROBERTA_START_DOCSTRING,
)

class RobertaForSequenceClassification_gcls_auto(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, VDC_auto, VIC_auto, num_auto_layers, head_wise, auto_VDC_k, a_pooler, g_pooler):
        super().__init__(config)
        config.num_labels = 3
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel_gcls_auto(config, add_pooling_layer=True, VDC_auto = VDC_auto, VIC_auto = VIC_auto,
                                              num_auto_layers = num_auto_layers, head_wise = head_wise, auto_VDC_k = auto_VDC_k,
                                              a_pooler = a_pooler, g_pooler = g_pooler)
        self.classifier = RobertaClassificationHead_gcls(config)
            
        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint="cardiffnlp/twitter-roberta-base-emotion",
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="'optimism'",
        expected_loss=0.08,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        extended_attention_mask = None,
        VDC_info = None,
        automation_visualization = None,
        sg_output = False,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs, s_outputs, g_outputs, visualization_result = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            extended_attention_mask = extended_attention_mask,
            VDC_info = VDC_info,
            automation_visualization = automation_visualization,
            sg_output = sg_output,
        )
        
        sequence_output = outputs[1]    # torch.tensor([32,768]), outputs[0].size() = torch.tensor([32,128,768])
        logits = self.classifier(sequence_output)

        loss = None
        sg_loss = None
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
                
        if sg_output:
            sg_loss = ()
            loss_fct_2 = MSELoss()
            for l in range(12):
                sg_loss = sg_loss + (loss_fct_2(s_outputs[l], g_outputs[l]), )

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return visualization_result, sg_loss, SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Roberta Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    ROBERTA_START_DOCSTRING,
)


class RobertaForSequenceClassification_TD(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        config.num_labels = 3
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel_TD(config, add_pooling_layer=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.dense = nn.Linear(config.hidden_size, config.num_labels)
        
        #### for att-based pooling
#         self.dense_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        ####
        
        self.classifier = RobertaClassificationHead_gcls(config)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint="cardiffnlp/twitter-roberta-base-emotion",
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="'optimism'",
        expected_loss=0.08,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        VDC_info = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        features = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            VDC_info = VDC_info,
        )[1]
        
        # avg pooling
#         index = VDC_info == 1
#         index = index.long()
        
#         mask = index.unsqueeze(2)
#         pooled = (features*mask).sum(dim=1)/mask.sum(dim=1)
        
#         target_in_sent_embed = torch.zeros(input_ids.size()[0], features.size()[-1]).to(input_ids.device)
#         for i in range(input_ids.size()[0]):
#             x = (extended_attention_mask[12][i][0][1][:] == 0.0).nonzero(as_tuple=True)[0]
            
#             # 가장 기본적인 RoBERTa-TD
#             target_embed = features[i][x[0]:x[-1]+1]
#             # 1. RoBERTa-TD-avg
#             # target_in_sent_embed[i] = target_embed.mean(dim = 0)
#             # 2. RoBERTa-TD-max
# #             target_in_sent_embed[i] = torch.max(target_embed, dim=0)[0]
#             # 3. RoBERTa-TD-att
#             c_vec = target_embed.mean(dim=0)
#             c_vec = self.dense_1(c_vec)
#             c_vec = self.activation(c_vec)
            
#             attention_scores = torch.matmul(c_vec.unsqueeze(0), target_embed.transpose(0,1))
#             attention_scores = attention_scores.view(-1,target_embed.size(0))
#             attention_probs = nn.functional.softmax(attention_scores, dim=-1)
#             target_in_sent_embed[i] = torch.matmul(attention_probs, target_embed).view(-1, 768)
            
# #             # Including [CLS]
# #             TD = torch.max(target_embed, dim=0)[0]
# #             cls_vec = features[i][0]
            
# #             ##############
# #             two_tensors = torch.cat((TD.unsqueeze(0), cls_vec.unsqueeze(0)), 0)
# #             c_vec = torch.mean(two_tensors, dim = 0)
# #             c_vec = self.dense_1(c_vec)
# #             c_vec = self.activation(c_vec)

# #             attention_scores = torch.matmul(c_vec.unsqueeze(0), two_tensors.transpose(0,1))
# #             attention_scores = attention_scores.view(-1,2)
# #             attention_probs = nn.functional.softmax(attention_scores, dim=-1)
# #             target_in_sent_embed[i] = torch.matmul(attention_probs, two_tensors).view(-1, 768)
#             ##############
            
            
# #             target_in_sent_embed[i] = (TD +cls_vec)/2.0
# #             target_embed = torch.cat((features[i][0].unsqueeze(0), features[i][x[0]-1:x[-1]+1-1]), 0)
            
# #             target_in_sent_embed[i] = torch.cat((features[i][0], torch.max(target_embed, dim=0)[0]), -1)
            
#         target_in_sent_embed = self.dropout(target_in_sent_embed)
        
        logits = self.classifier(features)
        
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
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


@add_start_docstrings(
    """
    Roberta Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    ROBERTA_START_DOCSTRING,
)


class RobertaForSequenceClassification_TD_t_star(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        config.num_labels = 3
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel_TD_t_star(config, add_pooling_layer=True)
        self.classifier = RobertaClassificationHead_gcls(config)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint="cardiffnlp/twitter-roberta-base-emotion",
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="'optimism'",
        expected_loss=0.08,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        VDC_info = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        features = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            VDC_info = VDC_info,
        )[1]
        
        logits = self.classifier(features)
        
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
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


@add_start_docstrings(
    """
    Roberta Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    ROBERTA_START_DOCSTRING,
)


class RobertaForMultipleChoice(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.roberta(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Roberta Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    ROBERTA_START_DOCSTRING,
)
class RobertaForTokenClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint="Jean-Baptiste/roberta-large-ner-english",
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="['O', 'ORG', 'ORG', 'O', 'O', 'O', 'O', 'O', 'LOC', 'O', 'LOC', 'LOC']",
        expected_loss=0.01,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class RobertaClassificationHead_gcls(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features 
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
@add_start_docstrings(
    """
    Roberta Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    ROBERTA_START_DOCSTRING,
)
class RobertaForQuestionAnswering(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint="deepset/roberta-base-squad2",
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="' puppet'",
        expected_loss=0.86,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx
