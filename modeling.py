# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
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
"""PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import math
import six
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
                vocab_size,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_act="gelu",
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                max_position_embeddings=512,
                type_vocab_size=16,
                initializer_range=0.02):
        """Constructs BertConfig.

        Args:
            vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BERTLayerNorm(nn.Module):
    def __init__(self, config, variance_epsilon=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BERTLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(config.hidden_size))
        self.beta = nn.Parameter(torch.zeros(config.hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class BERTEmbeddings(nn.Module):
    def __init__(self, config):
        super(BERTEmbeddings, self).__init__()
        """Construct the embedding module from word, position and token_type embeddings.
        """
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BERTSelfAttention(nn.Module):
    def __init__(self, config):
        super(BERTSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
#         self.W_g = nn.Linear(config.hidden_size, config.hidden_size)
#         self.gate_0 = nn.Linear(config.hidden_size, config.hidden_size)
#         self.gate_1 = nn.Linear(config.hidden_size, config.hidden_size)
#         self.gate_2 = nn.Linear(config.hidden_size, config.hidden_size)
#         self.gate_3 = nn.Linear(config.hidden_size, config.hidden_size)
        
#         self.sigmoid = nn.Sigmoid()
#         self.tanh = nn.Tanh()

    def gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, layer_i = None, VDC_info=None, AAW_mask_token=None, Q_g= None,
                K_g= None, V_g= None, current_VDC = None, path_info = None,  path_params = None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        # g infront of X
#         mixed_query_layer[:,1] = Q_g(mixed_query_layer[:,1].clone())
        
        ## path_params 를 어디에 더해야 하나?
        
        
#         mixed_key_layer[:,1] = K_g(mixed_key_layer[:,1].clone())
#         mixed_value_layer[:,1] = V_g(mixed_value_layer[:,1].clone())
#         for i in range(hidden_states.size(0)):
#             mixed_query_layer[i,1] = Q_g[int(current_VDC[i][layer_i])](mixed_query_layer[i,1].clone())
#             mixed_key_layer[i,1] = Q_g[int(current_VDC[i][layer_i])](mixed_key_layer[i,1].clone())
#             mixed_value_layer[i,1] = Q_g[int(current_VDC[i][layer_i])](mixed_value_layer[i,1].clone())
        
        
#         mixed_query_layer[:,1] = self.gate_0(self.tanh(mixed_query_layer[:,1].clone()))
        
        # g-update 할 때는 we want the [g] to be aware of the target.
#         index = VDC_info <= 1
#         index = index.long()
#         mask = index.unsqueeze(2)
#         t0 = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1) 
        
#         index = VDC_info <= 2
#         index = index.long()
#         mask = index.unsqueeze(2)
#         t1 = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1) 
        
#         index = VDC_info <= 3
#         index = index.long()
#         mask = index.unsqueeze(2)
#         t2 = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1) 
        
#         index = VDC_info <= 4
#         index = index.long()
#         mask = index.unsqueeze(2)
#         t3 = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1) 
        
#         index = VDC_info <= 5
#         index = index.long()
#         mask = index.unsqueeze(2)
#         t4 = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1) 
        
#         index = VDC_info <= 6
#         index = index.long()
#         mask = index.unsqueeze(2)
#         t5 = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1) 
        
#         gate_0 = self.sigmoid(self.gate_0(t0)) # B, 768
#         gate_1 = self.sigmoid(self.gate_1(t1)) # B, 768
#         gate_2 = self.sigmoid(self.gate_2(t2)) # B, 768
#         gate_3 = self.sigmoid(self.gate_3(t3)) # B, 768
#         gate_4 = self.sigmoid(self.gate(t4)) # B, 768
#         gate_5 = self.sigmoid(self.gate(t5)) # B, 768
        
        # 먼저 value만 바꿔보자.
#         mm0 = (VDC_info==1).long().unsqueeze(2)
#         mm1 = (VDC_info==2).long().unsqueeze(2)
#         mm2 = (VDC_info==3).long().unsqueeze(2)
#         mm3 = (VDC_info==4).long().unsqueeze(2)
#         mm4 = (VDC_info==5).long().unsqueeze(2)
#         mm5 = (VDC_info==6).long().unsqueeze(2)
        
#         value_reg = gate_0.unsqueeze(1)*mixed_value_layer[:,:]*mm0 +\
#                     gate_1.unsqueeze(1)*mixed_value_layer[:,:]*mm1 +\
#                     gate_2.unsqueeze(1)*mixed_value_layer[:,:]*mm2 +\
#                     gate_3.unsqueeze(1)*mixed_value_layer[:,:]*mm3
#                     gate_4.unsqueeze(1)*mixed_value_layer[:,:]*mm4 +\
#                     gate_5.unsqueeze(1)*mixed_value_layer[:,:]*mm5
        
        
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # value 작업.
#         value_layer_reg = self.transpose_for_scores(value_reg)
        
        
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
#         print('attention_scores.size(): ' , attention_scores.size())
#         print('attention_mask.size(): ', attention_mask.size())
#         print('attention_mask[0,0,1,:20]: ', attention_mask[0,0,0,:20])
        
        attention_scores = attention_scores + attention_mask
        
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        
#         print('='*77)
#         print('attention_probs.size(): ', attention_probs.size())
#         print('-'*77)
#         print('attention_probs[0,0,1,:20]: ', attention_probs[0,0,1,:20])
#         print('-'*77)
#         print('attention_probs[0,0,0,:20]: ', attention_probs[0,0,0,:20])
#         print('='*77)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        if layer_i>=10000:
            # Soft VDC 쓰는 경우
#             print('attention_mask[0,0,1,:30]: ', attention_mask[0,0,1,:30])
            hidden_states_g = self.tanh(w_g_layer(hidden_states))
#             attention_mask[:,0,1,:] = 0.0
            
#             attention_mask_ = attention_mask.clone()
#             attention_mask_[:,0,1,:] = (VDC_info == 100).float()*-10000.0
            
            mixed_query_layer_g = self.query(hidden_states_g)
            mixed_key_layer_g = self.key(hidden_states_g)
            mixed_value_layer_g = self.value(hidden_states_g)
            
            query_layer_g = self.transpose_for_scores(mixed_query_layer_g)
            key_layer_g = self.transpose_for_scores(mixed_key_layer_g)
            value_layer_g = self.transpose_for_scores(mixed_value_layer_g)
            
            attention_scores_g = torch.matmul(query_layer_g, key_layer_g.transpose(-1, -2))
            attention_scores_g = attention_scores_g / math.sqrt(self.attention_head_size)
            
            attention_scores_g = attention_scores_g + attention_mask
            # soft-VDC => 어차피 0.0 더하는 경우 위 line comment 가능.
            
            attention_probs_g = nn.Softmax(dim=-1)(attention_scores_g)
            
            attention_probs_g = self.dropout(attention_probs_g)
            
            context_layer_g = torch.matmul(attention_probs_g, value_layer_g)
            context_layer_g = context_layer_g.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape_g = context_layer_g.size()[:-2] + (self.all_head_size,)
            context_layer_g = context_layer_g.view(*new_context_layer_shape)
        
            context_layer[:,1] = context_layer_g[:,1]
        
        
        
        # value 작업
#         context_layer_reg = torch.matmul(attention_probs, value_layer_reg)
#         context_layer_reg = context_layer_reg.permute(0, 2, 1, 3).contiguous()
#         new_context_layer_reg_shape = context_layer_reg.size()[:-2] + (self.all_head_size,)
#         context_layer_reg = context_layer_reg.view(*new_context_layer_reg_shape)
#         context_layer[:,1] = context_layer_reg[:,1]
        
        
        return context_layer


class BERTSelfOutput(nn.Module):
    def __init__(self, config):
        super(BERTSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BERTAttention(nn.Module):
    def __init__(self, config):
        super(BERTAttention, self).__init__()
        self.self = BERTSelfAttention(config)
        self.output = BERTSelfOutput(config)

    def forward(self, input_tensor, attention_mask, layer_i = None, VDC_info = None, AAW_mask_token = None, Q_g = None,
                K_g= None, V_g=None, current_VDC = None, path_info = None,  path_params = None):
        self_output = self.self(input_tensor, attention_mask, layer_i = layer_i, VDC_info = VDC_info,
                                AAW_mask_token = AAW_mask_token, Q_g = Q_g,K_g = K_g,V_g = V_g, current_VDC = current_VDC,
                                path_info = path_info,  path_params = path_params)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BERTIntermediate(nn.Module):
    def __init__(self, config):
        super(BERTIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BERTOutput(nn.Module):
    def __init__(self, config):
        super(BERTOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BERTLayer(nn.Module):
    def __init__(self, config, plus_layer=None, minus_layer=None):
        super(BERTLayer, self).__init__()
        self.attention = BERTAttention(config)
        self.intermediate = BERTIntermediate(config)
        self.output = BERTOutput(config)
        
        self.dgedt_1 = nn.Linear(768, 768)
        self.dgedt_2 = nn.Linear(768, 768)
        self.sigmoid = nn.Sigmoid()
        self.cos = nn.CosineSimilarity(dim=1)
        self.plus_layer = plus_layer
        self.minus_layer = minus_layer
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        if plus_layer != None and minus_layer != None: 
            if self.plus_layer[0] == -1 and len(self.plus_layer) == 1:
                self.plus_layer = []
            if self.minus_layer[0] == -1 and len(self.minus_layer) == 1:
                self.minus_layer = []

    def forward(self, hidden_states, attention_mask, layer_i = None, VDC_info = None, AAW_mask_token = None, Q_g = None,
                 K_g = None, V_g = None, current_VDC = None, path_info = None, path_params = None):
        attention_output = self.attention(hidden_states, attention_mask, layer_i = layer_i, VDC_info = VDC_info,
                                          AAW_mask_token=AAW_mask_token, Q_g = Q_g, K_g=K_g, V_g=V_g, current_VDC = current_VDC,
                                          path_info = path_info,  path_params = path_params)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        cos_sim = 0
        
        return layer_output, 0.5*cos_sim


class BERTEncoder(nn.Module):
    def __init__(self, config):
        super(BERTEncoder, self).__init__()
        layer = BERTLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states, _ = layer_module(hidden_states, attention_mask)
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

class BERTEncoder_gcls(nn.Module):
    def __init__(self, config, plus_layer, minus_layer):
        super(BERTEncoder_gcls, self).__init__()
        layer = BERTLayer(config, plus_layer, minus_layer)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])
        
        self.Q_g = nn.Linear(config.hidden_size, config.hidden_size)
        
#         self.path = nn.ModuleList([nn.Parameter(torch.FloatTensor(300)) for _ in range(1+2+4)])

    def forward(self, hidden_states, extended_attention_mask=None, VDC_info = None, AAW_mask_token = None, current_VDC = None,
                path_info = None):
        all_encoder_layers = []
        cos_sim_agg = 0
        for i, layer_module in enumerate(self.layer):
#             if i > 3:
#                 ext_att_mask = extended_attention_mask[i].clone()
#                 ext_att_mask[:,:,0,1] = 0.0
#                 ext_att_mask[:,:,1,0] = 0.0
#                 hidden_states = layer_module(hidden_states, ext_att_mask)
#             else:
#                 hidden_states = layer_module(hidden_states, extended_attention_mask[i])
                
            hidden_states, cos_sim = layer_module(hidden_states, extended_attention_mask[i], layer_i = i, VDC_info = VDC_info,
                                                  AAW_mask_token = AAW_mask_token, Q_g = self.Q_g, K_g = None,
                                                  V_g = None, current_VDC = current_VDC, path_info = path_info, 
                                                  path_params = None)
            all_encoder_layers.append(hidden_states)
            cos_sim_agg += cos_sim
        return all_encoder_layers, cos_sim_agg

class BERTPooler(nn.Module):
    def __init__(self, config):
        super(BERTPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
    
class BERTPooler_TD(nn.Module):
    def __init__(self, config):
        super(BERTPooler_TD, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states, VDC_info = None):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
#         first_token_tensor = hidden_states[:, 0]
#         pooled_output = self.dense(first_token_tensor)
#         pooled_output = self.activation(pooled_output)
        index = VDC_info == 1
        index = index.long()
        mask = index.unsqueeze(2)
        t_vector = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1)
        
        
#         index = VDC_info == 1
#         index = index.long()
#         index_2 = (VDC_info != 1.0).long()*-float('inf')
#         index_2[index_2 != index_2] = 0 # change 0 * -inf = nan to zero
#         mask = index.unsqueeze(2)
#         mask_2 = index_2.unsqueeze(2)
#         t_vector = torch.max(hidden_states*mask + mask_2, dim = 1)[0]
        
        pooled_output = self.dense(0.5*t_vector+0.5*hidden_states[:,0])
        pooled_output = self.activation(pooled_output)
        
        return pooled_output

class BERTPooler_gcls(nn.Module):
    def __init__(self, config, g_pooler = None):
        super(BERTPooler_gcls, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        if g_pooler in ['s_g_att_ec', 's_g_att_ec_2', 's_g_t_avg_att_ec', 's_g_t_max_att_ec']:
            self.dense_2 = nn.Linear(config.hidden_size, 1)
        elif g_pooler in ['t_avg']:
            self.dense_2 = None
        elif g_pooler in ['g_t_avg_att']:
            self.dense_2 = nn.Linear(config.hidden_size, 1)
        elif g_pooler in ['g_t_avg_ec', 's_g_sg_ec']:
            self.dense_2 = nn.Linear(config.hidden_size, 1)
        elif g_pooler in ['s_g_t_avg_ec']:
            self.dense_2 = nn.Linear(config.hidden_size, 1)
        elif g_pooler in ['dgedt_1', 'dgedt_2', 'dgedt_3']:
            self.dgedt_1 = nn.Linear(config.hidden_size, config.hidden_size)
            self.dgedt_2 = nn.Linear(config.hidden_size, config.hidden_size)
            self.sigmoid = nn.Sigmoid()
        elif g_pooler in ['s_g_avg_var_1']:
            self.dense_2 = nn.Linear(config.hidden_size, config.hidden_size)
        elif g_pooler in ['s_g_t_avg_avg_var_1', 's_g_t_max_avg_var_1', 's_g_g2', 's_g2_t_avg', 's_g2_t2_avg',
                          's_g_sg_avg', 's_g_max_avg']:
            self.dense_2 = nn.Linear(config.hidden_size, config.hidden_size)
            self.dense_3 = nn.Linear(config.hidden_size, config.hidden_size)
        elif g_pooler in ['s_g_t_avg_avg_var_2']:
            self.dense_2 = nn.Linear(config.hidden_size, config.hidden_size)
            self.dense_3 = nn.Linear(config.hidden_size, config.hidden_size)
            self.dense_4 = nn.Linear(config.hidden_size, config.hidden_size)
        elif g_pooler in ['s_g_t_avg_avg_var_3']:
            self.dense_2 = nn.Linear(config.hidden_size, config.hidden_size)
            self.dense_3 = nn.Linear(config.hidden_size, config.hidden_size)
        elif g_pooler in ['g_t_avg_avg_var_1', 's_g_g2', 's_g2_t_avg', 's_g2_t2_avg']:
            self.dense_2 = nn.Linear(config.hidden_size, config.hidden_size)
        elif g_pooler in ['s_g_g2_t_avg_avg_var_1']:
            self.dense_2 = nn.Linear(config.hidden_size, config.hidden_size)
            self.dense_3 = nn.Linear(config.hidden_size, config.hidden_size)
            self.dense_4 = nn.Linear(config.hidden_size, config.hidden_size)
        elif g_pooler in ['s_g_g2_t_avg_avg_var_1_no_pre_dense']:
            self.dense_1 = nn.Linear(config.hidden_size, config.hidden_size)
            self.dense_2 = nn.Linear(config.hidden_size, config.hidden_size)
            self.dense_3 = nn.Linear(config.hidden_size, config.hidden_size)
            self.dense_4 = nn.Linear(config.hidden_size, config.hidden_size)
        elif g_pooler in ['s_g_g2_t_avg_t2_avg_avg_var_1']:
            self.dense_2 = nn.Linear(config.hidden_size, config.hidden_size)
            self.dense_3 = nn.Linear(config.hidden_size, config.hidden_size)
            self.dense_4 = nn.Linear(config.hidden_size, config.hidden_size)
            self.dense_5 = nn.Linear(config.hidden_size, config.hidden_size)
        elif g_pooler in ['sggcn_1']:
            self.dense_1 = nn.Linear(config.hidden_size, config.hidden_size)
            self.dense_2 = nn.Linear(3*config.hidden_size, config.hidden_size)
        elif g_pooler in ['sggcn_2']:
            self.dense_2 = nn.Linear(3*config.hidden_size, config.hidden_size)
        elif g_pooler in ['s_g_inter_t_avg_avg_var1']:
            self.dense_2 = nn.Linear(config.hidden_size, config.hidden_size)
            self.dense_3 = nn.Linear(config.hidden_size, config.hidden_size)
            self.dense_4 = nn.Linear(config.hidden_size, config.hidden_size)
            self.dense_layers = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(10)]) 
            self.dense_agg_1 = nn.Linear(config.hidden_size, config.hidden_size)
            self.dense_agg_2 = nn.Linear(config.hidden_size, config.hidden_size)
        elif g_pooler in ['s_g_concat']:
            self.dense_2 = nn.Linear(2*config.hidden_size, config.hidden_size)
        elif g_pooler in ['s_g_t_avg_concat']:
            self.dense_2 = nn.Linear(3*config.hidden_size, config.hidden_size)    
        
            
    def att_pool(self, vectors, layer):
        cat = torch.mean(vectors, dim=1)
        pooled_output = layer(cat)
        pooled_output = self.activation(pooled_output)

        attention_scores = torch.matmul(pooled_output.unsqueeze(1), vectors.transpose(1,2))
        attention_scores = attention_scores.view(-1,vectors.size(1))
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        final_output = torch.matmul(attention_probs.unsqueeze(1), vectors).view(-1, 768)
        
        return final_output
        
    def ec_pool(self, vectors, layer):
        scores = layer(vectors).view(-1, vectors.size(1))
        probs = nn.functional.softmax(scores, dim=1)
        final_output =  torch.matmul(probs.unsqueeze(1), vectors).view(-1, 768)
        final_output = self.activation(final_output)
        
        return final_output
    
    def avg_var(self, vectors):
        for ii in range(vectors.size(1)):
            if ii ==0:
                final_output = self.dense_layers[ii](vectors[:,ii])
            else:
                final_output += self.dense_layers[ii](vectors[:,ii])
        
        return final_output/vectors.size(1)
    
    def avg_pool(self, vectors):
        return torch.mean(vectors, dim=1)
        
    def forward(self, hidden_states, inter_g = None, g_pooler=None, VDC_info=None):
        if g_pooler == 'sggcn_1':
            s = hidden_states[:,0] 
            g = hidden_states[:,1]
            index = VDC_info == 1
            index = index.long()
            mask = index.unsqueeze(2)
            t = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1) 
            
#             index = VDC_info == 1
#             index = index.long()
#             index_2 = (VDC_info != 1.0).long()*-float('inf')
#             index_2[index_2 != index_2] = 0 # change 0 * -inf = nan to zero
#             mask = index.unsqueeze(2)
#             mask_2 = index_2.unsqueeze(2)
#             t_avg = torch.max(hidden_states*mask + mask_2, dim = 1)[0]
            
            gate = self.sigmoid(self.dense_1((s+g+t)/3.0)) # B, 768
            
            index = VDC_info != 100
            index = index.long()
            index_2 = (VDC_info == 100).long()*-float('inf')
            index_2[index_2 != index_2] = 0 # change 0 * -inf = nan to zero
            mask = index.unsqueeze(2)
            mask_2 = index_2.unsqueeze(2)
            
            X = gate.unsqueeze(1) * hidden_states[:,:]
            X_ = torch.max(X*mask[:,:] + mask_2[:,:], dim = 1)[0]
            
            final = torch.concat((s, g, X_), dim=1)
            
#             cat = 0.5*final_s + 0.5*final_g
            pooled_output = self.dense_2(final)
            pooled_output = self.activation(pooled_output)
            return pooled_output
        
        elif g_pooler == 's':
            pooled_output = self.dense(hidden_states[:, 0])
            pooled_output = self.activation(pooled_output)
            return pooled_output
        
        elif g_pooler == 'g':
            pooled_output = self.dense(hidden_states[:, 1])
            pooled_output = self.activation(pooled_output)
            return pooled_output
        
        elif g_pooler == 's_g_max':
            index = VDC_info == 1000 # g infront of X
            index = index.long()
            mask = index.unsqueeze(2)
            g_vector = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1)
            
            first_second_token_tensor = torch.cat((hidden_states[:,0].unsqueeze(1), g_vector.unsqueeze(1)), dim=1)
            cat = torch.max(first_second_token_tensor, dim=1)[0]
            pooled_output = self.dense(cat)
            pooled_output = self.activation(pooled_output)
            return pooled_output
        
            
        elif g_pooler == 's_g_avg':
            index = VDC_info == 1000 # g infront of X
            index = index.long()
            mask = index.unsqueeze(2)
            g_vector = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1)
            
            first_second_token_tensor = torch.cat((hidden_states[:,0].unsqueeze(1), g_vector.unsqueeze(1)), dim=1 )
            cat = torch.mean(first_second_token_tensor, dim = 1)
            pooled_output = self.dense(cat)
            pooled_output = self.activation(pooled_output)
            return pooled_output
        
        elif g_pooler == 's_g_sg_avg':
            index = VDC_info == 1000 # g infront of X
            index = index.long()
            mask = index.unsqueeze(2)
            g_vector = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1)
            
            pooled_output = self.activation((1/3)*self.dense(hidden_states[:,0])+
                                            (1/3)*self.dense_2(g_vector)+
                                            (1/3)*self.dense_3(hidden_states[:,0]*g_vector))
            
            return pooled_output
        
        elif g_pooler == 's_g_max_avg':
            index = VDC_info == 1000 # g infront of X
            index = index.long()
            mask = index.unsqueeze(2)
            g_vector = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1)
            
            max_ = torch.max(torch.concat((hidden_states[:,0].unsqueeze(1),
                                           g_vector.unsqueeze(1)), dim=1), dim = 1)[0]
            
            pooled_output = self.activation((1/3)*self.dense(hidden_states[:,0])+
                                            (1/3)*self.dense_2(g_vector)+
                                            (1/3)*self.dense_3(max_))
            
            return pooled_output
        
        elif g_pooler == 's_g_sg_ec':
            index = VDC_info == 1000 # g infront of X
            index = index.long()
            mask = index.unsqueeze(2)
            g_vector = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1)
            
            full_vector = torch.cat((hidden_states[:,0].unsqueeze(1), g_vector.unsqueeze(1),
                                    (hidden_states[:,0]*g_vector).unsqueeze(1)),
                                    dim=1)
            
            return self.ec_pool(full_vector, self.dense_2)
        
        
        
        elif g_pooler == 's_g2_avg':
            index_2 = VDC_info == -1
            index_2 = index_2.long()
            mask = index_2.unsqueeze(2)
            g2_vector = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1)
            
            pooled_output = self.activation(0.5*self.dense(hidden_states[:,0])+0.5*self.dense(g2_vector))
            return pooled_output
        
        
        elif g_pooler == 't_avg':
            index = VDC_info == 1
            index = index.long()
            mask = index.unsqueeze(2)
            t_vector = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1)
            
            return self.activation(self.dense(t_vector))
        
        elif g_pooler == 'g_t_avg_att':
            index = VDC_info == 1
            index = index.long()
            mask = index.unsqueeze(2)
            t_vector = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1)
            
            full_vector = torch.cat((hidden_states[:,1].unsqueeze(1), t_vector.unsqueeze(1)), dim=1)
            
            return self.att_pool(full_vector, self.dense)
        
        elif g_pooler == 'g_t_avg_ec':
            index = VDC_info == 1
            index = index.long()
            mask = index.unsqueeze(2)
            t_vector = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1)
            
            full_vector = torch.cat((hidden_states[:,1].unsqueeze(1), t_vector.unsqueeze(1)), dim=1)
            
            return self.ec_pool(full_vector, self.dense_2)
        
        elif g_pooler == 's_g_t_avg_ec':
            index = VDC_info == 1
            index = index.long()
            mask = index.unsqueeze(2)
            t_vector = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1)
            
            full_vector = torch.cat((hidden_states[:,1].unsqueeze(1), 
                                     hidden_states[:,1].unsqueeze(1), t_vector.unsqueeze(1)), dim=1)
            
            return self.ec_pool(full_vector, self.dense_2)
        
        elif g_pooler == 's_g_avg_var_1':
            pooled_output = self.activation(0.5*self.dense(hidden_states[:,0])+0.5*self.dense_2(hidden_states[:,1]))
            return pooled_output
        
        elif g_pooler == 's_g_t_avg_avg_var_1':
            index = VDC_info == 1
            index = index.long()
            mask = index.unsqueeze(2)
            t_vector = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1)
            
            index = VDC_info == 1000 # g infront of X
            index = index.long()
            mask = index.unsqueeze(2)
            g_vector = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1)
            
            pooled_output = self.activation((1/3)*self.dense(hidden_states[:,0])+
                                            (1/3)*self.dense_2(g_vector)+
                                            (1/3)*self.dense_3(t_vector))
            
            return pooled_output
        
        elif g_pooler == 's_g_t_max_avg_var_1':
            index = VDC_info == 1
            index = index.long()
            index_2 = (VDC_info != 1.0).long()*-float('inf')
            index_2[index_2 != index_2] = 0 # change 0 * -inf = nan to zero
            mask = index.unsqueeze(2)
            mask_2 = index_2.unsqueeze(2)
            t_vector = torch.max(hidden_states*mask + mask_2, dim = 1)[0]
            
            index = VDC_info == 1000 # g infront of X
            index = index.long()
            mask = index.unsqueeze(2)
            g_vector = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1)
            
            pooled_output = self.activation((1/3)*self.dense(hidden_states[:,0])+
                                            (1/3)*self.dense_2(g_vector)+
                                            (1/3)*self.dense_3(t_vector))
            
            return pooled_output
        
        elif g_pooler == 's_g_t_avg_avg_var_2':
            index = VDC_info == 1
            index = index.long()
            mask = index.unsqueeze(2)
            t_vector = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1)
            
            index = VDC_info == 1000 # g infront of X
            index = index.long()
            mask = index.unsqueeze(2)
            g_vector = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1)
            
            pooled_output = self.activation((1/3)*self.dense_2(hidden_states[:,0])+
                                            (1/3)*self.dense_3(g_vector)+
                                            (1/3)*self.dense_4(t_vector))
            
            pooled_output = self.activation(self.dense(pooled_output))
            
            return pooled_output
        
        elif g_pooler == 's_g_t_avg_avg_var_3':
            index = VDC_info == 1
            index = index.long()
            mask = index.unsqueeze(2)
            t_vector = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1)
            
            index = VDC_info == 1000 # g infront of X
            index = index.long()
            mask = index.unsqueeze(2)
            g_vector = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1)
            
            pooled_output = self.activation((1/3)*self.dense(hidden_states[:,0])+
                                            (1/3)*self.dense(g_vector)+
                                            (1/3)*self.dense_2(t_vector))
            
            
            return pooled_output
        
        elif g_pooler == 'g_t_avg_avg_var_1':
            index = VDC_info == 1
            index = index.long()
            mask = index.unsqueeze(2)
            t_vector = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1)
            
            pooled_output = self.activation((1/2)*self.dense_2(hidden_states[:,1])+
                                            (1/2)*self.dense(t_vector))
            
            return pooled_output
        
        
        elif g_pooler == 's_g_g2':
            
            index_2 = VDC_info == -1
            index_2 = index_2.long()
            mask = index_2.unsqueeze(2)
            g2_vector = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1)
            
            pooled_output = self.activation((1/3)*self.dense(hidden_states[:,0])+
                                            (1/3)*self.dense_2(hidden_states[:,1])+
                                            (1/3)*self.dense_3(g2_vector))
            
            return pooled_output
        
        elif g_pooler == 's_g2_t_avg':
            index = VDC_info == 1
            index = index.long()
            mask = index.unsqueeze(2)
            t_vector = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1)
            
            index_2 = VDC_info == -1
            index_2 = index_2.long()
            mask = index_2.unsqueeze(2)
            g2_vector = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1)
            
            pooled_output = self.activation((1/3)*self.dense(hidden_states[:,0])+
                                            (1/3)*self.dense_2(g2_vector)+
                                            (1/3)*self.dense_3(t_vector))
            
            return pooled_output
        
        elif g_pooler == 's_g2_t2_avg':
            index = VDC_info == -2
            index = index.long()
            mask = index.unsqueeze(2)
            t2_vector = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1)
            
            index_2 = VDC_info == -1
            index_2 = index_2.long()
            mask = index_2.unsqueeze(2)
            g2_vector = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1)
            
            pooled_output = self.activation((1/3)*self.dense(hidden_states[:,0])+
                                            (1/3)*self.dense_2(g2_vector)+
                                            (1/3)*self.dense_3(t2_vector))
            
            return pooled_output
         
        elif g_pooler == 's_g_g2_t_avg_avg_var_1':
            index = VDC_info == 1
            index = index.long()
            mask = index.unsqueeze(2)
            t_vector = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1)
            
            index_1 = VDC_info == 1000
            index_1 = index_1.long()
            mask_1 = index_1.unsqueeze(2)
            g_vector = (hidden_states*mask_1).sum(dim=1)/mask_1.sum(dim=1)
            
            
            index_2 = VDC_info == -1
            index_2 = index_2.long()
            mask_2 = index_2.unsqueeze(2)
            g2_vector = (hidden_states*mask_2).sum(dim=1)/mask_2.sum(dim=1)
            
#             g_vector = hidden_states[:,1]
#             intermediate_g = torch.cat((inter_g, g_vector.unsqueeze(1)),dim=1)
            
            pooled_output = self.activation((1/4)*self.dense(hidden_states[:,0])+
                                            (1/4)*self.dense_2(g_vector)+
                                            (1/4)*self.dense_3(t_vector)+
                                            (1/4)*self.dense_4(g2_vector))
            
            return pooled_output
        
        
        
        elif g_pooler == 's_g_inter_t_avg_avg_var1':
            index = VDC_info == 1
            index = index.long()
            mask = index.unsqueeze(2)
            t_vector = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1)
            
#             index_2 = VDC_info == -1
#             index_2 = index_2.long()
#             mask = index_2.unsqueeze(2)
#             g2_vector = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1)
            
#             index_3 = VDC_info == -2
#             index_3 = index_3.long()
#             mask = index_3.unsqueeze(2)
#             t2_vector = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1)
            
#             g_vector = torch.max(torch.cat((hidden_states[:,0].unsqueeze(1), inter_g), dim=1), dim=1)[0]
            g_vector = self.activation((1/2)*self.dense_agg_1(hidden_states[:,1])+
                                            (1/2)*self.dense_agg_2(inter_g.squeeze(1)))
            
                  
            full_vector = torch.cat((hidden_states[:,0].unsqueeze(1),
                                     g_vector.unsqueeze(1),
                                     t_vector.unsqueeze(1)),dim=1)
            
#             final_output = self.ec_pool(full_vector, self.dense_1)
            
#             final_output = self.avg_var(full_vector)
#             final_output = self.activation((1/4)*self.dense(hidden_states[:,0])+
#                                             (1/4)*self.dense_2(g_vector)+
#                                             (1/4)*self.dense_3(t_vector)+
#                                             (1/4)*self.dense_4(g2_vector))
            
            final_output = self.avg_var(full_vector)
            final_output = self.activation(final_output)
            return final_output
        
        elif g_pooler == 's_g_g2_t_avg_avg_var_1_no_pre_dense':
            index = VDC_info == 1
            index = index.long()
            mask = index.unsqueeze(2)
            t_vector = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1)
            
            index_2 = VDC_info == -1
            index_2 = index_2.long()
            mask = index_2.unsqueeze(2)
            g2_vector = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1)
            
            pooled_output = self.activation((1/4)*self.dense_1(hidden_states[:,0])+
                                            (1/4)*self.dense_2(hidden_states[:,1])+
                                            (1/4)*self.dense_3(t_vector)+
                                            (1/4)*self.dense_4(g2_vector))
            
            return pooled_output
         
        elif g_pooler == 's_g_g2_t_avg_t2_avg_avg_var_1':
            index = VDC_info == 1
            index = index.long()
            mask = index.unsqueeze(2)
            t_vector = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1)
            
            index_2 = VDC_info == -1
            index_2 = index_2.long()
            mask = index_2.unsqueeze(2)
            g2_vector = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1)
            
            index_3 = VDC_info == -2
            index_3 = index_3.long()
            mask = index_3.unsqueeze(2)
            t2_vector = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1)
            
            pooled_output = self.activation((1/5)*self.dense(hidden_states[:,0])+
                                            (1/5)*self.dense_2(hidden_states[:,1])+
                                            (1/5)*self.dense_3(t_vector)+
                                            (1/5)*self.dense_4(g2_vector)+
                                            (1/5)*self.dense_5(t2_vector))
            
            return pooled_output
        
        elif g_pooler == 's_g_t_avg_avg':
            index = VDC_info == 1
            index = index.long()
            mask = index.unsqueeze(2)
            t_vector = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1)
            
            sgt_vector = torch.cat((hidden_states[:, :2], t_vector.unsqueeze(1)), dim=1)
            cat = torch.mean(sgt_vector, dim = 1)
            pooled_output = self.dense(cat)
            pooled_output = self.activation(pooled_output)
            return pooled_output
        
        elif g_pooler == 's_g_t_max_avg':
            index = VDC_info == 1
            index = index.long()
            index_2 = (VDC_info != 1.0).long()*-float('inf')
            index_2[index_2 != index_2] = 0 # change 0 * -inf = nan to zero
            mask = index.unsqueeze(2)
            mask_2 = index_2.unsqueeze(2)
            t_vector = torch.max(hidden_states*mask + mask_2, dim = 1)[0]
            
            sgt_vector = torch.cat((hidden_states[:, :2], t_vector.unsqueeze(1)), dim=1)
            cat = torch.mean(sgt_vector, dim = 1)
            pooled_output = self.dense(cat)
            pooled_output = self.activation(pooled_output)
            return pooled_output
        
        elif g_pooler == 's_g_t_max_att_1':
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

            attention_scores = torch.matmul(pooled_output.unsqueeze(1), sgt_vector[:,:2].transpose(1,2))
            attention_scores = attention_scores.view(-1,2)
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)
            final_output = torch.matmul(attention_probs.unsqueeze(1), sgt_vector[:,:2]).view(-1, 768)
            final_output = self.activation(final_output)
            
            return final_output
        
        elif g_pooler == 's_g_t_max_att_2':
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
            
#         elif g_pooler == 's_g_max':
#             first_second_token_tensor = hidden_states[:, :2]
#             cat = torch.max(first_second_token_tensor, dim=1)[0]
#             pooled_output = self.dense(cat)
#             pooled_output = self.activation(pooled_output)
#             return pooled_output
        
        elif g_pooler == 's_g_att':
            T = 1.0
            first_second_token_tensor = hidden_states[:, :2]
            cat = torch.mean(first_second_token_tensor, dim=1)
            pooled_output = self.dense(cat)
            pooled_output = self.activation(pooled_output)

            attention_scores = torch.matmul(pooled_output.unsqueeze(1), first_second_token_tensor.transpose(1,2))
            attention_scores = attention_scores.view(-1,2)
            attention_probs = nn.functional.softmax(attention_scores/T, dim=-1)
            final_output = torch.matmul(attention_probs.unsqueeze(1), first_second_token_tensor).view(-1, 768)
            
        elif g_pooler == 's_g_att_var_1':
            # Uses s,g,t to compute the weights between s and g
            index = VDC_info == 1
            index = index.long()
            mask = index.unsqueeze(2)
            t_vector = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1)
            
            sgt_vector = torch.cat((hidden_states[:, :2], t_vector.unsqueeze(1)), dim=1)
            cat = torch.mean(sgt_vector, dim=1)
            pooled_output = self.dense(cat)
            pooled_output = self.activation(pooled_output)

            attention_scores = torch.matmul(pooled_output.unsqueeze(1), sgt_vector[:,:2].transpose(1,2))
            attention_scores = attention_scores.view(-1,2)
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)
            final_output = torch.matmul(attention_probs.unsqueeze(1), sgt_vector[:,:2]).view(-1, 768)
            final_output = self.activation(final_output)
            
        elif g_pooler == 's_g_att_var_2':
            # Uses s,g,t to compute the weights between s, g, and t.
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

#         elif g_pooler == 's_g_att_var_3':
#             # Uses top 10 tokens to predict the weights between (s,g)
#             # What if we just choose top 10 tokens at the end..? 
#             # 구현하기..!

        elif g_pooler == 's_g_concat':
            index = VDC_info == 1000 # g infront of X
            index = index.long()
            mask = index.unsqueeze(2)
            g_vector = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1)
            
            first_second_token_tensor = torch.cat((hidden_states[:,0], g_vector), dim=1)
            pooled_output = self.activation(self.dense_2(first_second_token_tensor))
            final_output = self.activation(self.dense(pooled_output))
            
            return final_output
        
        elif g_pooler == 's_g_t_avg_concat':
            index = VDC_info == 1
            index = index.long()
            mask = index.unsqueeze(2)
            t_vector = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1)
            
            index = VDC_info == 1000 # g infront of X
            index = index.long()
            mask = index.unsqueeze(2)
            g_vector = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1)
            
            first_second_token_tensor = torch.cat((hidden_states[:,0], g_vector, t_vector), dim=1)
            pooled_output = self.activation(self.dense_2(first_second_token_tensor))
            final_output = self.activation(self.dense(pooled_output))
            
            return final_output
            
        elif g_pooler == 's_g_att_ec': # individual dense (O) + activation (O)
            T = 0.5 # Temperature
            first_second_token_tensor = hidden_states[:, :2]
            scores = self.dense_2(first_second_token_tensor).view(-1, 2)
            probs = nn.functional.softmax(scores/T, dim=1)
            final_output =  torch.matmul(probs.unsqueeze(1), first_second_token_tensor).view(-1, 768)
            final_output = self.activation(final_output)
            
        elif g_pooler == 's_g_t_avg_att_ec': # individual dense (O) + activation (O)
            T = 0.5 # Temperature
            
            index = VDC_info == 1
            index = index.long()
            mask = index.unsqueeze(2)
            t_vector = (hidden_states*mask).sum(dim=1)/mask.sum(dim=1)
            
            sgt_vector = torch.cat((hidden_states[:, :2], t_vector.unsqueeze(1)), dim=1)
            scores = self.dense_2(sgt_vector).view(-1, 3)
            probs = nn.functional.softmax(scores/T, dim=1)
            final_output =  torch.matmul(probs.unsqueeze(1), sgt_vector).view(-1, 768)
            final_output = self.activation(final_output)
            
        elif g_pooler == 's_g_t_max_att_ec': # individual dense (O) + activation (O)
            T = 0.5 # Temperature
            
            index = VDC_info == 1
            index = index.long()
            index_2 = (VDC_info != 1.0).long()*-float('inf')
            index_2[index_2 != index_2] = 0 # change 0 * -inf = nan to zero
            mask = index.unsqueeze(2)
            mask_2 = index_2.unsqueeze(2)
            t_vector = torch.max(hidden_states*mask + mask_2, dim = 1)[0]
            
            sgt_vector = torch.cat((hidden_states[:, :2], t_vector.unsqueeze(1)), dim=1)
            scores = self.dense_2(sgt_vector).view(-1, 3)
            probs = nn.functional.softmax(scores/T, dim=1)
            final_output =  torch.matmul(probs.unsqueeze(1), sgt_vector).view(-1, 768)
            final_output = self.activation(final_output)
            
            
        elif g_pooler == 'dgedt_1': # individual dense (O) + activation (O)
            T = 1.0 # Temperature
            coef_1 = self.sigmoid((torch.matmul(hidden_states[:,0].unsqueeze(1),
                                                   self.dgedt_1(hidden_states[:,1]).unsqueeze(2))*T).view(-1,1))
            coef_2 = self.sigmoid((torch.matmul(hidden_states[:,1].unsqueeze(1),
                                                   self.dgedt_2(hidden_states[:,0]).unsqueeze(2))*T).view(-1,1))
            final_output =  coef_1*hidden_states[:,0] + coef_2*hidden_states[:,1]
            
        elif g_pooler == 'dgedt_2': # individual dense (O) + activation (O)
            T = 1.0 # Temperature
            coef_1 = torch.matmul(hidden_states[:,0].unsqueeze(1),
                                                   self.dgedt_1(hidden_states[:,1]).unsqueeze(2)).view(-1,1)
            coef_2 = torch.matmul(hidden_states[:,1].unsqueeze(1),
                                                   self.dgedt_2(hidden_states[:,0]).unsqueeze(2)).view(-1,1)
            coeffs = torch.cat((coef_1, coef_2), dim=1)
            probs = nn.functional.softmax(coeffs/T, dim=1)
            final_output =  probs[:,0:1]*hidden_states[:,0] + probs[:,1:2]*hidden_states[:,1]
                                  
        return final_output


class BertModel(nn.Module):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])

    config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config: BertConfig):
        """Constructor for BertModel.

        Args:
            config: `BertConfig` instance.
        """
        super(BertModel, self).__init__()
        self.embeddings = BERTEmbeddings(config)
        self.encoder = BERTEncoder(config)
        self.pooler = BERTPooler(config)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        all_encoder_layers = self.encoder(embedding_output, extended_attention_mask)
        sequence_output = all_encoder_layers[-1]
        pooled_output = self.pooler(sequence_output)
        return all_encoder_layers, pooled_output
    
class BertModel_TD(nn.Module):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])

    config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config: BertConfig):
        """Constructor for BertModel.

        Args:
            config: `BertConfig` instance.
        """
        super(BertModel_TD, self).__init__()
        self.embeddings = BERTEmbeddings(config)
        self.encoder = BERTEncoder(config)
        self.pooler = BERTPooler_TD(config)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, VDC_info = None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        all_encoder_layers = self.encoder(embedding_output, extended_attention_mask)
        sequence_output = all_encoder_layers[-1]
        pooled_output = self.pooler(sequence_output, VDC_info = VDC_info)
        return all_encoder_layers, pooled_output
    

class BertModel_gcls(nn.Module):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])

    config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config: BertConfig, g_pooler, plus_layer, minus_layer):
        """Constructor for BertModel.

        Args:
            config: `BertConfig` instance.
        """
        super(BertModel_gcls, self).__init__()
        self.embeddings = BERTEmbeddings(config)
        self.encoder = BERTEncoder_gcls(config, plus_layer, minus_layer)
        self.pooler = BERTPooler_gcls(config, g_pooler = g_pooler)
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, extended_attention_mask = None, g_pooler=None,
                VDC_info = None, AAW_mask_token = None, current_VDC = None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        embedding_output = self.embeddings(input_ids, token_type_ids)
        all_encoder_layers, cos_sim_agg = self.encoder(embedding_output, extended_attention_mask=extended_attention_mask,
                                                       VDC_info = VDC_info, AAW_mask_token = AAW_mask_token,
                                                       current_VDC = current_VDC)
        sequence_output = all_encoder_layers[-1]
        
#         cosine_similarity = self.cos(sequence_output[:,0].clone(), sequence_output[:,1].clone())
#         index = VDC_info == 1
#         index = index.long()
#         mask = index.unsqueeze(2)
#         t = (sequence_output*mask).sum(dim=1)/mask.sum(dim=1) 
        
#         avg = (1/2)*sequence_output[:,0] + (1/2)*sequence_output[:,1]
#         cosine_similarity = (1/2)*(
#                             torch.mean(self.cos(sequence_output[:,0], avg), dim=0) +
#                             torch.mean(self.cos(sequence_output[:,1], avg), dim=0))
        
        
#         cosine_similarity = F.mse_loss(sequence_output[:,0].clone(), sequence_output[:,1].clone())
        # Using the intermediate [g] vectors
        inter_g = all_encoder_layers[10][:,1].unsqueeze(1)
#         inter_g = torch.cat((all_encoder_layers[8][:,1].unsqueeze(1),
#                              all_encoder_layers[9][:,1].unsqueeze(1),
#                              all_encoder_layers[10][:,1].unsqueeze(1)), dim = 1)
        
        pooled_output = self.pooler(sequence_output, inter_g=inter_g, g_pooler = g_pooler, VDC_info = VDC_info)
        
        return all_encoder_layers, pooled_output, 0.0

class BertForSequenceClassification(nn.Module):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])

    config = BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_labels):
        super(BertForSequenceClassification, self).__init__()
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()
        self.apply(init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits

        
class BertForSequenceClassification_TD(nn.Module):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])

    config = BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_labels):
        super(BertForSequenceClassification_TD, self).__init__()
        self.bert = BertModel_TD(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()
        self.apply(init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None, VDC_info = None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, VDC_info = VDC_info)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits
        
        
class BertForSequenceClassification_gcls(nn.Module):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])

    config = BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_labels, g_pooler, plus_layer, minus_layer):
        super(BertForSequenceClassification_gcls, self).__init__()
        self.bert = BertModel_gcls(config, g_pooler = g_pooler, plus_layer=plus_layer, minus_layer=minus_layer)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                # Original
#                 module.bias.data.zero_()
                
                # In RoBERTa:
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
                
                
        self.apply(init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None, extended_attention_mask=None, g_pooler = None,
                VDC_info = None, AAW_mask_token = None, current_VDC = None):
        all_encoder_layers, pooled_output, cosine_similarity = self.bert(input_ids, token_type_ids, attention_mask, extended_attention_mask=extended_attention_mask, g_pooler = g_pooler, VDC_info = VDC_info, AAW_mask_token = AAW_mask_token,
                                                                        current_VDC = current_VDC)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            
            return loss, logits
        else:
            return logits

class BertForSequenceClassification_gcls_MoE(nn.Module):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])

    config = BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_labels):
        super(BertForSequenceClassification_gcls_MoE, self).__init__()
        self.bert = BertModel_gcls_MoE(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()
        self.apply(init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None, gcls_attention_mask=None, layer_L = None, MoE_layer = None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, gcls_attention_mask, layer_L, MoE_layer)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits

class BertForSequenceClassification_MoE(nn.Module):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])

    config = BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_labels):
        super(BertForSequenceClassification_MoE, self).__init__()
        self.bert = BertModel_MoE(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()
        self.apply(init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None, gcls_attention_mask=None, layer_L = None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, gcls_attention_mask, layer_L)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits

class BertForQuestionAnswering(nn.Module):
    """BERT model for Question Answering (span extraction).
    This module is composed of the BERT model with a linear layer on top of
    the sequence output that computes start_logits and end_logits

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])

    config = BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    model = BertForQuestionAnswering(config)
    start_logits, end_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__()
        self.bert = BertModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()
        self.apply(init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, start_positions=None, end_positions=None):
        all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_output = all_encoder_layers[-1]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        else:
            return start_logits, end_logits
