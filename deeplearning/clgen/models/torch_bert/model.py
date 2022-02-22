# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model. """

import math
import os
import typing

import numpy as np

from deeplearning.clgen.util import pytorch
from deeplearning.clgen.util.pytorch import torch
from deeplearning.clgen.models.torch_bert import activations
from deeplearning.clgen.models.torch_bert import config
from deeplearning.clgen.models.torch_bert import modeling_utils
from deeplearning.clgen.models.torch_bert import compiler

from deeplearning.clgen.util import logging as l

# import tensorrt as trt
# import pycuda.autoinit
# import pycuda.driver as cuda

def mish(x):
  return x * torch.tanh(torch.nn.functional.softplus(x))

ACT2FN = {
  "gelu": activations.gelu,
  "relu": torch.nn.functional.relu,
  "swish": activations.swish,
  "gelu_new": activations.gelu_new,
  "mish": mish
}

class BertEmbeddings(torch.nn.Module):
  """Construct the embeddings from word, position and token_type embeddings.
  """

  def __init__(self, config):
    super().__init__()
    self.word_embeddings = torch.nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
    self.position_embeddings = torch.nn.Embedding(config.max_position_embeddings, config.hidden_size)
    # self.token_type_embeddings = torch.nn.Embedding(config.type_vocab_size, config.hidden_size)

    # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
    # any TensorFlow checkpoint file
    self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

    # position_ids (1, len position emb) is contiguous in memory and exported when serialized
    self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

  def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
    if input_ids is not None:
      input_shape = input_ids.size()
    else:
      input_shape = inputs_embeds.size()[:-1]

    seq_length = input_shape[1]

    if position_ids is None:
      position_ids = self.position_ids[:, :seq_length]

    if token_type_ids is None:
      token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

    if inputs_embeds is None:
      inputs_embeds = self.word_embeddings(input_ids)
    position_embeddings = self.position_embeddings(position_ids)
    # token_type_embeddings = self.token_type_embeddings(token_type_ids)

    embeddings = inputs_embeds + position_embeddings # + token_type_embeddings
    embeddings = self.LayerNorm(embeddings)
    embeddings = self.dropout(embeddings)
    return embeddings


class BertSelfAttention(torch.nn.Module):
  def __init__(self, config):
    super().__init__()
    if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
      raise ValueError(
        "The hidden size (%d) is not a multiple of the number of attention "
        "heads (%d)" % (config.hidden_size, config.num_attention_heads)
      )

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    self.query = torch.nn.Linear(config.hidden_size, self.all_head_size)
    self.key = torch.nn.Linear(config.hidden_size, self.all_head_size)
    self.value = torch.nn.Linear(config.hidden_size, self.all_head_size)

    self.dropout = torch.nn.Dropout(config.attention_probs_dropout_prob)

  def transpose_for_scores(self, x):
    new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
    x = x.view(*new_x_shape)
    return x.permute(0, 2, 1, 3)

  def forward(
    self,
    hidden_states,
    attention_mask=None,
    head_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    output_attentions=False,
  ):
    mixed_query_layer = self.query(hidden_states)

    # If this is instantiated as a cross-attention module, the keys
    # and values come from an encoder; the attention mask needs to be
    # such that the encoder's padding tokens are not attended to.
    if encoder_hidden_states is not None:
      mixed_key_layer = self.key(encoder_hidden_states)
      mixed_value_layer = self.value(encoder_hidden_states)
      attention_mask = encoder_attention_mask
    else:
      mixed_key_layer = self.key(hidden_states)
      mixed_value_layer = self.value(hidden_states)

    query_layer = self.transpose_for_scores(mixed_query_layer)
    key_layer = self.transpose_for_scores(mixed_key_layer)
    value_layer = self.transpose_for_scores(mixed_value_layer)

    # Take the dot product between "query" and "key" to get the raw attention scores.
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    if attention_mask is not None:
      # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
      attention_scores = attention_scores + attention_mask

    # Normalize the attention scores to probabilities.
    attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = self.dropout(attention_probs)

    # Mask heads if we want to
    if head_mask is not None:
      attention_probs = attention_probs * head_mask

    context_layer = torch.matmul(attention_probs, value_layer)

    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(*new_context_layer_shape)

    outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
    return outputs


class BertSelfOutput(torch.nn.Module):
  def __init__(self, config):
    super().__init__()
    self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
    self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

  def forward(self, hidden_states, input_tensor):
    hidden_states = self.dense(hidden_states)
    hidden_states = self.dropout(hidden_states)
    hidden_states = self.LayerNorm(hidden_states + input_tensor)
    return hidden_states


class BertAttention(torch.nn.Module):
  def __init__(self, config):
    super().__init__()
    self.self = BertSelfAttention(config)
    self.output = BertSelfOutput(config)
    self.pruned_heads = set()

  def prune_heads(self, heads):
    if len(heads) == 0:
      return
    heads, index = modeling_utils.find_pruneable_heads_and_indices(
      heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
    )

    # Prune linear layers
    self.self.query = modeling_utils.prune_linear_layer(self.self.query, index)
    self.self.key = modeling_utils.prune_linear_layer(self.self.key, index)
    self.self.value = modeling_utils.prune_linear_layer(self.self.value, index)
    self.output.dense = modeling_utils.prune_linear_layer(self.output.dense, index, dim=1)

    # Update hyper params and store pruned heads
    self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
    self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
    self.pruned_heads = self.pruned_heads.union(heads)

  def forward(
    self,
    hidden_states,
    attention_mask=None,
    head_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    output_attentions=False,
  ):
    self_outputs = self.self(
      hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, output_attentions,
    )
    attention_output = self.output(self_outputs[0], hidden_states)
    outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
    return outputs


class BertIntermediate(torch.nn.Module):
  def __init__(self, config):
    super().__init__()
    self.dense = torch.nn.Linear(config.hidden_size, config.intermediate_size)
    if isinstance(config.hidden_act, str):
      self.intermediate_act_fn = ACT2FN[config.hidden_act]
    else:
      self.intermediate_act_fn = config.hidden_act

  def forward(self, hidden_states):
    hidden_states = self.dense(hidden_states)
    hidden_states = self.intermediate_act_fn(hidden_states)
    return hidden_states


class BertOutput(torch.nn.Module):
  def __init__(self, config):
    super().__init__()
    self.dense = torch.nn.Linear(config.intermediate_size, config.hidden_size)
    self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

  def forward(self, hidden_states, input_tensor):
    hidden_states = self.dense(hidden_states)
    hidden_states = self.dropout(hidden_states)
    hidden_states = self.LayerNorm(hidden_states + input_tensor)
    return hidden_states


class BertLayer(torch.nn.Module):
  def __init__(self, config):
    super().__init__()
    self.chunk_size_feed_forward = config.chunk_size_feed_forward
    self.seq_len_dim = 1
    self.attention = BertAttention(config)
    self.is_decoder = config.is_decoder
    self.add_cross_attention = config.add_cross_attention
    if self.add_cross_attention:
      assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
      self.crossattention = BertAttention(config)
    self.intermediate = BertIntermediate(config)
    self.output = BertOutput(config)

  def forward(
    self,
    hidden_states,
    attention_mask=None,
    head_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    output_attentions=False,
  ):
    self_attention_outputs = self.attention(
      hidden_states, attention_mask, head_mask, output_attentions=output_attentions,
    )
    attention_output = self_attention_outputs[0]
    outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

    if self.is_decoder and encoder_hidden_states is not None:
      assert hasattr(
        self, "crossattention"
      ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
      cross_attention_outputs = self.crossattention(
        attention_output,
        attention_mask,
        head_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        output_attentions,
      )
      attention_output = cross_attention_outputs[0]
      outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

    layer_output = modeling_utils.apply_chunking_to_forward(
      self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
    )
    outputs = (layer_output,) + outputs
    return outputs

  def feed_forward_chunk(self, attention_output):
    intermediate_output = self.intermediate(attention_output)
    layer_output = self.output(intermediate_output, attention_output)
    return layer_output


class BertEncoder(torch.nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.layer = torch.nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

  def forward(
    self,
    hidden_states,
    attention_mask=None,
    head_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    output_attentions=False,
    output_hidden_states=False,
  ):
    all_hidden_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None
    for i, layer_module in enumerate(self.layer):
      if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

      if getattr(self.config, "gradient_checkpointing", False):

        def create_custom_forward(module):
          def custom_forward(*inputs):
            return module(*inputs, output_attentions)

          return custom_forward

        layer_outputs = torch.utils.checkpoint.checkpoint(
          create_custom_forward(layer_module),
          hidden_states,
          attention_mask,
          head_mask[i],
          encoder_hidden_states,
          encoder_attention_mask,
        )
      else:
        layer_outputs = layer_module(
          hidden_states,
          attention_mask,
          head_mask[i],
          encoder_hidden_states,
          encoder_attention_mask,
          output_attentions,
        )
      hidden_states = layer_outputs[0]
      if output_attentions:
        all_attentions = all_attentions + (layer_outputs[1],)

    if output_hidden_states:
      all_hidden_states = all_hidden_states + (hidden_states,)

    return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

class BertPooler(torch.nn.Module):
  def __init__(self, config):
    super().__init__()
    self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
    self.activation = torch.nn.Tanh()

  def forward(self, hidden_states):
    # We "pool" the model by simply taking the hidden state corresponding
    # to the first token.
    first_token_tensor = hidden_states[:, 0]
    pooled_output = self.dense(first_token_tensor)
    pooled_output = self.activation(pooled_output)
    return pooled_output


class BertPredictionHeadTransform(torch.nn.Module):
  def __init__(self, config):
    super().__init__()
    self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
    if isinstance(config.hidden_act, str):
      self.transform_act_fn = ACT2FN[config.hidden_act]
    else:
      self.transform_act_fn = config.hidden_act
    self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

  def forward(self, hidden_states):
    hidden_states = self.dense(hidden_states)
    hidden_states = self.transform_act_fn(hidden_states)
    hidden_states = self.LayerNorm(hidden_states)
    return hidden_states


class BertLMPredictionHead(torch.nn.Module):
  def __init__(self, config):
    super().__init__()
    self.transform = BertPredictionHeadTransform(config)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    self.decoder = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    self.bias = torch.nn.Parameter(torch.zeros(config.vocab_size))

    # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
    self.decoder.bias = self.bias

  def forward(self, hidden_states):
    hidden_states = self.transform(hidden_states)
    hidden_states = self.decoder(hidden_states)
    return hidden_states


class BertOnlyMLMHead(torch.nn.Module):
  def __init__(self, config):
    super().__init__()
    self.predictions = BertLMPredictionHead(config)

  def forward(self, sequence_output):
    prediction_scores = self.predictions(sequence_output)
    return prediction_scores


class BertOnlyNSPHead(torch.nn.Module):
  def __init__(self, config):
    super().__init__()
    self.seq_relationship = torch.nn.Linear(config.hidden_size, 2)

  def forward(self, pooled_output):
    seq_relationship_score = self.seq_relationship(pooled_output)
    return seq_relationship_score


class BertPreTrainingHeads(torch.nn.Module):
  def __init__(self, config):
    super().__init__()
    self.predictions = BertLMPredictionHead(config)
    self.seq_relationship = torch.nn.Linear(config.hidden_size, 2)

  def forward(self, sequence_output, pooled_output):
    prediction_scores = self.predictions(sequence_output)
    seq_relationship_score = self.seq_relationship(pooled_output)
    return prediction_scores, seq_relationship_score


class BertPreTrainedModel(modeling_utils.PreTrainedModel):
  """ An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
  """

  config_class = config.BertConfig
  base_model_prefix = "bert"
  authorized_missing_keys = [r"position_ids"]

  def _init_weights(self, module):
    """ Initialize the weights """
    if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
      # Slightly different from the TF version which uses truncated_normal for initialization
      # cf https://github.com/pytorch/pytorch/pull/5617
      module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    elif isinstance(module, torch.nn.LayerNorm):
      module.bias.data.zero_()
      module.weight.data.fill_(1.0)
    if isinstance(module, torch.nn.Linear) and module.bias is not None:
      module.bias.data.zero_()

  def get_output(self,
                 input_ids,
                 attention_mask,
                 position_ids,
                 token_type_ids       = None,
                 head_mask            = None,
                 inputs_embeds        = None,
                 output_attentions    = None,
                 output_hidden_states = None,
                 ) -> typing.Tuple[torch.FloatTensor, torch.FloatTensor]:
    raise NotImplementedError("Abstract class")

class BertModel(BertPreTrainedModel):
  """

  The model can behave as an encoder (with only self-attention) as well
  as a decoder, in which case a layer of cross-attention is added between
  the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
  Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

  To behave as an decoder the model needs to be initialized with the
  :obj:`is_decoder` argument of the configuration set to :obj:`True`.
  To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
  argument and :obj:`add_cross_attention` set to :obj:`True`; an
  :obj:`encoder_hidden_states` is then expected as an input to the forward pass.

  .. _`Attention is all you need`:
    https://arxiv.org/abs/1706.03762

  """

  def __init__(self, config):
    super().__init__(config)
    self.config = config

    self.embeddings = BertEmbeddings(config)
    self.encoder = BertEncoder(config)
    self.pooler = BertPooler(config)

    self.init_weights()

  def get_input_embeddings(self):
    return self.embeddings.word_embeddings

  def set_input_embeddings(self, value):
    self.embeddings.word_embeddings = value

  def _prune_heads(self, heads_to_prune):
    """ Prunes heads of the model.
      heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
      See base class PreTrainedModel
    """
    for layer, heads in heads_to_prune.items():
      self.encoder.layer[layer].attention.prune_heads(heads)

  def forward(
    self,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    output_attentions=None,
    output_hidden_states=None,
  ):
    r"""
    encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
      Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
      if the model is configured as a decoder.
    encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
      Mask to avoid performing attention on the padding token indices of the encoder input. This mask
      is used in the cross-attention if the model is configured as a decoder.
      Mask values selected in ``[0, 1]``:
      ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
      output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )

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
      attention_mask = torch.ones(input_shape, device=device)
    if token_type_ids is None:
      token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

    # If a 2D ou 3D attention mask is provided for the cross-attention
    # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
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
      input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
    )
    encoder_outputs = self.encoder(
      embedding_output,
      attention_mask=extended_attention_mask,
      head_mask=head_mask,
      encoder_hidden_states=encoder_hidden_states,
      encoder_attention_mask=encoder_extended_attention_mask,
      output_attentions=output_attentions,
      output_hidden_states=output_hidden_states,
    )
    sequence_output = encoder_outputs[0]
    pooled_output = self.pooler(sequence_output)

    return (sequence_output, pooled_output) + encoder_outputs[1:]

class BertForPreTraining(BertPreTrainedModel):
  def __init__(self, config, tokenizer = None, use_categorical = False, temperature = None):
    super().__init__(config)

    self.bert = BertModel(config)
    self.cls  = BertOnlyMLMHead(config)

    if self.config.reward_compilation >= 0 or self.config.is_sampling:
      self.compile_sampler = compiler.CompilationSampler(
        tokenizer, use_categorical, temperature
      )
    else:
      self.compile_sampler = None

    self.init_weights()

  def get_output_embeddings(self):
    return self.cls.predictions.decoder

  def get_output(self,
                 input_ids,
                 attention_mask,
                 position_ids,
                 token_type_ids       = None,
                 head_mask            = None,
                 inputs_embeds        = None,
                 output_attentions    = None,
                 output_hidden_states = None,
                 ) -> typing.Tuple[torch.FloatTensor, torch.FloatTensor]:
    outputs = self.bert(
      input_ids            = input_ids,
      attention_mask       = attention_mask,
      position_ids         = position_ids,
      token_type_ids       = token_type_ids,
      head_mask            = head_mask,
      inputs_embeds        = inputs_embeds,
      output_attentions    = output_attentions,
      output_hidden_states = output_hidden_states,
    )
    sequence_output, pooled_output = outputs[:2]
    # prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)
    prediction_scores = self.cls(sequence_output)
    # return prediction_scores, seq_relationship_score, outputs[0], outputs[1]
    return prediction_scores, outputs[0], outputs[1]

  def forward(
    self,
    input_ids        = None,
    attention_mask   = None,
    token_type_ids   = None,
    position_ids     = None,
    head_mask        = None,
    inputs_embeds    = None,
    masked_lm_labels = None,
    next_sentence_labels = None,
    workload             = None,
    output_attentions    = None,
    output_hidden_states = None,
    is_validation        = False,
    is_live              = False,
    step                 = -1,
    **kwargs
  ):
    r"""
    labels (``torch.LongTensor`` of shape ``(batch_size, sequence_length)``, `optional`, defaults to :obj:`None`):
      Labels for computing the masked language modeling loss.
      Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
      Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
      in ``[0, ..., config.vocab_size]``
    next_sentence_labels (``torch.LongTensor`` of shape ``(batch_size,)``, `optional`, defaults to :obj:`None`):
      Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair (see :obj:`input_ids` docstring)
      Indices should be in ``[0, 1]``.
      ``0`` indicates sequence B is a continuation of sequence A,
      ``1`` indicates sequence B is a random sequence.
    kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
      Used to hide legacy arguments that have been deprecated.

  Returns:

  Examples::

    >>> from transformers import BertTokenizer, BertForPreTraining
    >>> import torch

    >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    >>> model = BertForPreTraining.from_pretrained('bert-base-uncased')

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    >>> outputs = model(**inputs)

    >>> prediction_logits = outptus.prediction_logits
    >>> seq_relationship_logits = outputs.seq_relationship_logits
    """
    if workload is not None:
      input_ids, attention_mask, position_ids = workload
    
    device = input_ids.get_device()
    device = device if device >= 0 else 'cpu'

    if workload is not None:
      # prediction_scores, seq_relationship_score, hidden_states, attentions = self.get_output(
      prediction_scores, hidden_states, attentions = self.get_output(
        input_ids[0], attention_mask[0], position_ids[0]
      )
      return self.compile_sampler.generateSampleWorkload(
        self,
        device,
        input_ids,
        attention_mask,
        prediction_scores,
        position_ids[0],
      )

    # prediction_scores, seq_relationship_score, hidden_states, attentions = self.get_output(
    prediction_scores, hidden_states, attentions = self.get_output(
      input_ids, attention_mask, position_ids, token_type_ids, head_mask,
      inputs_embeds, output_attentions, output_hidden_states 
    )
    if not is_validation and self.compile_sampler and step >= self.config.reward_compilation and not self.config.is_sampling:
      samples, compile_flag, masked_lm_labels = self.compile_sampler.generateTrainingBatch(
        self,
        device,
        input_ids.cpu(),
        prediction_scores.cpu(),
        torch.clone(position_ids),
        masked_lm_labels.cpu().numpy(),
      )
      loss_fct = torch.nn.CrossEntropyLoss()
      masked_lm_loss     = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
      # next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_labels.view(-1))
      total_loss = masked_lm_loss # + next_sentence_loss
      return {
        'masked_lm_loss'          : masked_lm_loss,
        # 'next_sentence_loss'      : next_sentence_loss,
        'total_loss'              : total_loss,
        'prediction_logits'       : prediction_scores,
        'seq_relationship_logits' : seq_relationship_score,
        'hidden_states'           : hidden_states,
        'attentions'              : attentions,
        'compile_status'          : torch.LongTensor(compile_flag).to(device),
        'generated_samples'       : torch.LongTensor(samples).to(device),
        'batch_compilation_rate'  : torch.full((1,), float(sum(compile_flag)) / len(compile_flag), dtype = torch.float).to(device),
        # 'sample_indices'          : [0],
      }
    elif not is_validation and self.compile_sampler and self.config.is_sampling:
      samples, sample_indices, scores_history = self.compile_sampler.generateSampleBatch(
        self,
        device,
        input_ids,
        prediction_scores,
        position_ids,
        is_live,
      )
      if is_live:
        return {
          'prediction_scores' : scores_history, # This is mainly used for live sampling. Else, watch out!
          'generated_samples' : samples,
          'sample_indices'    : sample_indices,
        }
      else:
        return {
          'generated_samples': samples,
          # 'sample_indices'   : sample_indices,
        }
    else:
      loss_fct = torch.nn.CrossEntropyLoss()
      masked_lm_loss     = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
      # next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_labels.view(-1))
      total_loss = masked_lm_loss # + next_sentence_loss

      return {
        'masked_lm_loss'          : masked_lm_loss,
        # 'next_sentence_loss'      : next_sentence_loss,
        'total_loss'              : total_loss,
        'prediction_logits'       : prediction_scores,
        'seq_relationship_logits' : seq_relationship_score,
        'hidden_states'           : hidden_states,
        'attentions'              : attentions,
      }


class BertForPreTrainingTRT(BertForPreTraining):
  def __init__(self, config, tokenizer = None, use_categorical = False, temperature = None):
    super().__init__(config, tokenizer=tokenizer, use_categorical=use_categorical, temperature=temperature)
    self.forward = self._forward_pytorch
    self.get_output = self._get_output_pytorch
    self.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

  def init_engine(self, cache, device_id, batch_size, sequence_length, vocab_size, max_position_embeddings):
    self.engine_path = cache.path / f'active_bert.{device_id}.engine'
    self.model_onnx_path = cache.path / f'active_bert.{device_id}.onnx'
    if not self.engine_path.exists():
      self._create_engine(batch_size, sequence_length, vocab_size, max_position_embeddings)

    self.runtime = trt.Runtime(self.TRT_LOGGER)
    with open(self.engine_path, 'rb') as f:
      self.engine = self.runtime.deserialize_cuda_engine(f.read())

    self.stream = cuda.Stream()
    self.inputs = []
    self.outputs = []
    self.bindings = []
    for binding in self.engine:
      shape = self.engine.get_binding_shape(binding)
      size = trt.volume(shape)# * batch_size
      dtype = trt.nptype(self.engine.get_binding_dtype(binding))
      host_mem = cuda.pagelocked_empty(size, dtype).reshape(shape)
      device_mem = cuda.mem_alloc(host_mem.nbytes)
      # Append the device buffer to device bindings.
      self.bindings.append(int(device_mem))
      # Append to the appropriate list.
      if self.engine.binding_is_input(binding):
        self.inputs.append((host_mem, device_mem))
      else:
        self.outputs.append((host_mem, device_mem))

    # Override the pytorch module () operator
    self.__call__ = self._forward_trt
    self.forward = self._forward_trt
    self.get_output = self._get_output_trt


  def _create_engine(self, batch_size, sequence_length, vocab_size, max_position_embeddings):
    with torch.no_grad():
      dims = (batch_size, sequence_length)
      input_ids = torch.autograd.Variable(torch.randint(vocab_size, dims)).cuda()
      attention_mask = torch.autograd.Variable(torch.ones(dims)).cuda()
      position_ids = torch.autograd.Variable(torch.randint(max_position_embeddings, dims)).cuda()
      args = (input_ids, attention_mask, position_ids)

      inputs = ['input_ids', 'attention_mask', 'position_ids']
      outputs = ['prediction_scores']
      dynamic_axes = {
          'input_ids': {0: 'batch'}, 
          'attention_mask': {0: 'batch'}, 
          'position_ids': {0: 'batch'},
          'prediction_scores':{0: 'batch'}
          }
      #out = torch.onnx.export(self.sample.model, args=args, f=model_onnx_path, input_names=inputs, output_names=outputs, dynamic_axes=dynamic_axes)
      out = torch.onnx.export(self, args=args, f=self.model_onnx_path, input_names=inputs, output_names=outputs)

    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(self.TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, self.TRT_LOGGER) as parser:
      with open(self.model_onnx_path, 'rb') as model_onnx:
        if not parser.parse(model_onnx.read()):
          for error in range(parser.num_errors):
            print(parser.get_error(error))

    with trt.Builder(self.TRT_LOGGER) as builder, builder.create_builder_config() as config:
      config.max_workspace_size = 1 << 29 # This determines the amount of memory available to the builder when building an optimized engine and should generally be set as high as possible.
      with builder.build_engine(network, config) as engine:
        with open(self.engine_path, 'wb') as f:
          f.write(engine.serialize())


  def _get_output_pytorch(self,
                 input_ids,
                 attention_mask,
                 position_ids,
                 token_type_ids       = None,
                 head_mask            = None,
                 inputs_embeds        = None,
                 output_attentions    = None,
                 output_hidden_states = None,
                 ) -> typing.Tuple[torch.FloatTensor, torch.FloatTensor]:
    outputs = self.bert(
      input_ids            = input_ids,
      attention_mask       = attention_mask,
      position_ids         = position_ids,
      token_type_ids       = token_type_ids,
      head_mask            = head_mask,
      inputs_embeds        = inputs_embeds,
      output_attentions    = output_attentions,
      output_hidden_states = output_hidden_states,
    )
    sequence_output, pooled_output = outputs[:2]
    prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)
    return prediction_scores, seq_relationship_score, outputs[0], outputs[1]

  def _forward_pytorch(
    self,
    input_ids,
    attention_mask,
    position_ids
  ):
    prediction_scores, _, _, _ = self._get_output_pytorch(input_ids, attention_mask, position_ids)
    return prediction_scores

  def _get_output_trt(self,
                 input_ids,
                 attention_mask,
                 position_ids
                 ) -> typing.Tuple[torch.FloatTensor, torch.FloatTensor]:
    np.copyto(self.inputs[0][0], input_ids.cpu())
    np.copyto(self.inputs[1][0], attention_mask.cpu())
    np.copyto(self.inputs[2][0], position_ids.cpu())

    for inp in self.inputs:
      cuda.memcpy_htod_async(inp[1], inp[0], self.stream) 
    self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
    cuda.memcpy_dtoh_async(self.outputs[0][0], self.outputs[0][1], self.stream) 
    self.stream.synchronize()
    return torch.tensor(self.outputs[0][0]).cpu(), None, None, None

  def _forward_trt(
    self,
    input_ids        = None,
    attention_mask   = None,
    token_type_ids   = None,
    position_ids     = None,
    head_mask        = None,
    inputs_embeds    = None,
    masked_lm_labels = None,
    next_sentence_labels = None,
    output_attentions    = None,
    output_hidden_states = None,
    is_validation        = False,
    is_live              = False,
    step                 = -1,
    **kwargs
  ):
    if is_validation or not self.compile_sampler or not self.config.is_sampling:
      raise NotImplementedError
    with self.engine.create_execution_context() as self.context:

      prediction_scores, _, _, _ = self._get_output_trt(input_ids, attention_mask, position_ids)
      device = input_ids.get_device()
      samples, sample_indices, scores_history = self.compile_sampler.generateSampleBatch(
        self,
        input_ids.get_device(),
        input_ids.cpu(),
        prediction_scores.cpu(),
        position_ids,
        is_live,
      )
      return {
        'prediction_scores' : scores_history, # This is mainly used for live sampling. Else, watch out!
        'generated_samples' : samples,
        'sample_indices'    : sample_indices,
      }
