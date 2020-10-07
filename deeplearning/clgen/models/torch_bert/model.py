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

import logging
import math
import os
import typing
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple
import concurrent.futures
# import torch
# import torch.utils.checkpoint

from deeplearning.clgen.util import pytorch
from deeplearning.clgen.util.pytorch import torch
from deeplearning.clgen.preprocessors import opencl
from deeplearning.clgen.models.torch_bert.activations import gelu, gelu_new, swish
from deeplearning.clgen.models.torch_bert.config import BertConfig
from deeplearning.clgen.models.torch_bert.modeling_utils import (
  PreTrainedModel,
  apply_chunking_to_forward,
  find_pruneable_heads_and_indices,
  prune_linear_layer,
)

def mish(x):
  return x * torch.tanh(torch.nn.functional.softplus(x))

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new, "mish": mish}

BertLayerNorm = torch.nn.LayerNorm

class BertEmbeddings(torch.nn.Module):
  """Construct the embeddings from word, position and token_type embeddings.
  """

  def __init__(self, config):
    super().__init__()
    self.word_embeddings = torch.nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
    self.position_embeddings = torch.nn.Embedding(config.max_position_embeddings, config.hidden_size)
    self.token_type_embeddings = torch.nn.Embedding(config.type_vocab_size, config.hidden_size)

    # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
    # any TensorFlow checkpoint file
    self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
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
    token_type_embeddings = self.token_type_embeddings(token_type_ids)

    embeddings = inputs_embeds + position_embeddings + token_type_embeddings
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
    self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
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
    self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
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

    layer_output = apply_chunking_to_forward(
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
    return_dict=False,
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

    if not return_dict:
      return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
    # return BaseModelOutput(
    #   last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
    # )


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
    self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

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


class BertPreTrainedModel(PreTrainedModel):
  """ An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
  """

  config_class = BertConfig
  base_model_prefix = "bert"
  authorized_missing_keys = [r"position_ids"]

  def _init_weights(self, module):
    """ Initialize the weights """
    if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
      # Slightly different from the TF version which uses truncated_normal for initialization
      # cf https://github.com/pytorch/pytorch/pull/5617
      module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    elif isinstance(module, BertLayerNorm):
      module.bias.data.zero_()
      module.weight.data.fill_(1.0)
    if isinstance(module, torch.nn.Linear) and module.bias is not None:
      module.bias.data.zero_()


# @dataclass
class BertForPreTrainingOutput(typing.NamedTuple):
  """
  Output type of :class:`~transformers.BertForPreTrainingModel`.

  Args:
    loss (`optional`, returned when ``labels`` is provided, ``torch.FloatTensor`` of shape :obj:`(1,)`):
      Total loss as the sum of the masked language modeling loss and the next sequence prediction (classification) loss.
    prediction_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
      Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    seq_relationship_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, 2)`):
      Prediction scores of the next sequence prediction (classification) head (scores of True/False
      continuation before SoftMax).
    hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
      Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
      of shape :obj:`(batch_size, sequence_length, hidden_size)`.

      Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
      Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
      :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

      Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
      heads.
  """

  masked_lm_loss: Optional[torch.FloatTensor] = None
  next_sentence_loss: Optional[torch.FloatTensor] = None
  total_loss: Optional[torch.FloatTensor] = None
  prediction_logits: torch.FloatTensor = None
  seq_relationship_logits: torch.FloatTensor = None
  hidden_states: Optional[Tuple[torch.FloatTensor]] = None
  attentions: Optional[Tuple[torch.FloatTensor]] = None
  batch_compilation_rate: float = None
  compile_status: typing.List = None
  generated_samples: typing.List = None
  sample_indices: typing.List = None

BERT_START_DOCSTRING = r"""
  This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
  Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
  usage and behavior.

  Parameters:
    config (:class:`~transformers.BertConfig`): Model configuration class with all the parameters of the model.
      Initializing with a config file does not load the weights associated with the model, only the configuration.
      Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

BERT_INPUTS_DOCSTRING = r"""
  Args:
    input_ids (:obj:`torch.LongTensor` of shape :obj:`{0}`):
      Indices of input sequence tokens in the vocabulary.

      Indices can be obtained using :class:`transformers.BertTokenizer`.
      See :func:`transformers.PreTrainedTokenizer.encode` and
      :func:`transformers.PreTrainedTokenizer.__call__` for details.

      `What are input IDs? <../glossary.html#input-ids>`__
    attention_mask (:obj:`torch.FloatTensor` of shape :obj:`{0}`, `optional`, defaults to :obj:`None`):
      Mask to avoid performing attention on padding token indices.
      Mask values selected in ``[0, 1]``:
      ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.

      `What are attention masks? <../glossary.html#attention-mask>`__
    token_type_ids (:obj:`torch.LongTensor` of shape :obj:`{0}`, `optional`, defaults to :obj:`None`):
      Segment token indices to indicate first and second portions of the inputs.
      Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
      corresponds to a `sentence B` token

      `What are token type IDs? <../glossary.html#token-type-ids>`_
    position_ids (:obj:`torch.LongTensor` of shape :obj:`{0}`, `optional`, defaults to :obj:`None`):
      Indices of positions of each input sequence tokens in the position embeddings.
      Selected in the range ``[0, config.max_position_embeddings - 1]``.

      `What are position IDs? <../glossary.html#position-ids>`_
    head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
      Mask to nullify selected heads of the self-attention modules.
      Mask values selected in ``[0, 1]``:
      :obj:`1` indicates the head is **not masked**, :obj:`0` indicates the head is **masked**.
    inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
      Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
      This is useful if you want more control over how to convert `input_ids` indices into associated vectors
      than the model's internal embedding lookup matrix.
    output_attentions (:obj:`bool`, `optional`, defaults to :obj:`None`):
      If set to ``True``, the attentions tensors of all attention layers are returned. See ``attentions`` under returned tensors for more detail.
    output_hidden_states (:obj:`bool`, `optional`, defaults to :obj:`None`):
      If set to ``True``, the hidden states of all layers are returned. See ``hidden_states`` under returned tensors for more detail.
    return_dict (:obj:`bool`, `optional`, defaults to :obj:`None`):
      If set to ``True``, the model will return a :class:`~transformers.file_utils.ModelOutput` instead of a
      plain tuple.
"""

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
    return_dict=None,
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
      return_dict=return_dict,
    )
    sequence_output = encoder_outputs[0]
    pooled_output = self.pooler(sequence_output)

    if not return_dict:
      return (sequence_output, pooled_output) + encoder_outputs[1:]

    # return BaseModelOutputWithPooling(
    #   last_hidden_state=sequence_output,
    #   pooler_output=pooled_output,
    #   hidden_states=encoder_outputs.hidden_states,
    #   attentions=encoder_outputs.attentions,
    # )

class BertForPreTraining(BertPreTrainedModel):
  def __init__(self, config, atomizer = None, use_categorical = False, temperature = None):
    super().__init__(config)

    self.bert = BertModel(config)
    self.cls  = BertPreTrainingHeads(config)
    self.atomizer        = atomizer
    self.use_categorical = use_categorical
    self.temperature     = temperature
    if self.use_categorical:
      self.argmax = lambda x: torch.argmax(torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(
        temperature = self.temperature if self.temperature is not None else 1.0, logits = x)
      )
    else:
      self.argmax = lambda x: torch.argmax(x)

    self.init_weights()

  def get_output_embeddings(self):
    return self.cls.predictions.decoder

  def fillTrainSeq(self,
                   batch,
                   prediction_scores,
                   attention_mask,
                   ):
    """
    Takes a single sequence of the batch.
    """
    seq_length    = tuple(batch.shape)[0]
    holes_exist   = False

    allowed_incr = (seq_length - int(torch.where(batch==self.atomizer.padToken)[0][0])
                    if self.atomizer.padToken in batch
                    else 0)
    rem_adds = allowed_incr
    ignore_idx, no_hole, with_hole = set(), {}, {}

    for target_idx in torch.where((batch == self.atomizer.holeToken) | (batch == self.atomizer.maskToken))[0]:
      idx        = int(target_idx)
      prediction = self.argmax(prediction_scores[target_idx])

      if int(prediction) in {self.atomizer.endholeToken, self.atomizer.maskToken, self.atomizer.holeToken}:
        # Model predicted sth that will close the hole.
        rem_adds += 1
        ignore_idx.add(idx)
      elif batch[idx] == self.atomizer.maskToken or not rem_adds:
        # Asked position is a mask, or it is a hole and there is no room.
        no_hole[idx] = int(prediction)
      elif rem_adds:
        rem_adds   -= 1
        holes_exist = True
        with_hole[idx] = int(prediction)

    new_batch = []
    for idx, t in enumerate(batch):
      if idx in ignore_idx:
        continue
      elif idx in no_hole:
        new_batch.append(no_hole[idx])
      elif idx in with_hole:
        new_batch += [with_hole[idx], self.atomizer.holeToken]
      else:
        new_batch.append(t)

    if len(new_batch) < seq_length:
      new_batch += [self.atomizer.padToken] * (seq_length - len(new_batch))
    else:
      new_batch = new_batch[:len(new_batch) - (allowed_incr - rem_adds)]
    assert len(new_batch) == seq_length, "{} - {} - {}".format(len(new_batch), allowed_incr, rem_adds)
    new_batch = torch.LongTensor([new_batch])

    if self.atomizer.padToken not in new_batch[0]:
      attention_mask = torch.ones([1, seq_length], dtype = torch.int64)
    else:
      pad_idx = torch.where(new_batch[0] == self.atomizer.padToken)[0][0]
      attention_mask = torch.cat((
        torch.ones ([1, pad_idx],              dtype = torch.int64),
        torch.zeros([1, seq_length - pad_idx], dtype = torch.int64)
      ), axis = 1)

    return holes_exist, new_batch, attention_mask

  def checkIfBatchCompiles(self, sample):
    """Check if generated sample compiles"""
    try:
      stdout = opencl.Compile(self.atomizer.ArrayToCode(sample))
      return 1
    except ValueError:
      return 0

  def fillPredictionSeq(self,
                        input_ids,
                        predictions,
                        attention_mask,
                        sample_indices,
                        ):
    """
    Updates new_input_ids with the model's output prediction.
    The output, if still contains hole or mask tokens, is fed back
    to the model's input through the input_fn's sample_gen generator.
    """
    masked_lm_ids = [
                      [x for idx, x in enumerate(predictions[batch_idx])
                          if input_ids[batch_idx][idx] == self.atomizer.maskToken
                          or input_ids[batch_idx][idx] == self.atomizer.holeToken
                      ] for batch_idx in range(len(input_ids))
                    ]
    assert len(input_ids) == len(masked_lm_ids), "Inputs and predictions do not have the same batch size."

    updated_sequence = []
    there_is_target = False
    for batch_idx, _ in enumerate(input_ids):
      batch = []
      mask_id_index     = 0
      closed_hole_index = 0
      for idx, token in enumerate(input_ids[batch_idx]):
        if   token == self.atomizer.maskToken:
          there_is_target = True
          mt = masked_lm_ids[batch_idx][mask_id_index]
          if mt == self.atomizer.maskToken or mt == self.atomizer.holeToken:
            continue
          if len(sample_indices[batch_idx][mask_id_index]) > 0:
            while(sample_indices[batch_idx][mask_id_index + closed_hole_index][-1]) == self.atomizer.endholeToken:
              closed_hole_index += 1
          sample_indices[batch_idx][mask_id_index + closed_hole_index].append(int(mt.cpu().numpy()))
          mask_id_index += 1
          batch.append(mt)
        elif token == self.atomizer.holeToken:
          there_is_target = True
          mt = masked_lm_ids[batch_idx][mask_id_index]
          if mt == self.atomizer.maskToken or mt == self.atomizer.holeToken:
            continue
          if len(sample_indices[batch_idx][mask_id_index]) > 0:
            while(sample_indices[batch_idx][mask_id_index + closed_hole_index][-1]) == self.atomizer.endholeToken:
              closed_hole_index += 1
          sample_indices[batch_idx][mask_id_index + closed_hole_index].append(int(mt.cpu().numpy()))
          mask_id_index += 1
          if mt != self.atomizer.endholeToken:
            batch.append(mt)
            batch.append(self.atomizer.holeToken)
            # done = False
        else:
          batch.append(token)

      while len(batch) < len(input_ids[batch_idx]):
        batch.append(self.atomizer.padToken)
      batch = batch[:len(input_ids[batch_idx])]

      pad_idx = None
      if self.atomizer.padToken in batch:
        pad_idx = batch.index(self.atomizer.padToken)
      attention_mask[batch_idx] = (torch.full([len(input_ids[0])], 1, dtype = torch.int64)
                        if pad_idx is None else
                        torch.cat(
                            (torch.full([pad_idx], 1, dtype = torch.int64),
                             torch.full([len(input_ids[batch_idx]) - pad_idx], 0, dtype = torch.int64)
                            )
                          )
                        )

      batch = batch[:len(input_ids[0])]
      updated_sequence.append(batch)
    new_input_ids = torch.LongTensor(updated_sequence).to(pytorch.device)
    return there_is_target, new_input_ids, attention_mask, sample_indices

  def apply_batch(self, batch, prediction, attention, position_ids, masked_lm_label):

    holes, new_batch, new_attention = self.fillTrainSeq(
      batch, prediction, attention
    )
    while holes:
      outputs = self.bert(
        input_ids      = new_batch.to(pytorch.device),
        attention_mask = new_attention.to(pytorch.device),
        position_ids   = position_ids.to(pytorch.device),
      )
      seq_output, pooled_output = outputs[:2]
      new_prediction, new_seq_relationship_score = self.cls(seq_output, pooled_output)

      holes, new_batch, new_attention = self.fillTrainSeq(
        new_batch[0],
        new_prediction.detach().cpu()[0],
        new_attention,
      )
    compile_flag = self.checkIfBatchCompiles(new_batch[0].numpy())
    if compile_flag:
      for idx, t in enumerate(masked_lm_label):
        if t != -100:
          masked_lm_label[t] = -100
    return new_batch[0], compile_flag, masked_lm_label

  def forward(
    self,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    masked_lm_labels=None,
    next_sentence_labels=None,
    output_attentions=None,
    output_hidden_states=None,
    is_training = True,
    is_prediction = False,
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
    >>> model = BertForPreTraining.from_pretrained('bert-base-uncased', return_dict=True)

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    >>> outputs = model(**inputs)

    >>> prediction_logits = outptus.prediction_logits
    >>> seq_relationship_logits = outputs.seq_relationship_logits
    """
    compile_samples = (self.config.reward_compilation and is_training) or (is_prediction and not is_training)

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

    if compile_samples:
      if is_training:
        batch_size, sequence_length = tuple(input_ids.shape)
        with concurrent.futures.ThreadPoolExecutor() as executor:
          jobs = [executor.submit(self.apply_batch,
                                  input_ids        [i].cpu(),
                                  prediction_scores[i].detach().cpu(),
                                  attention_mask   [i].cpu(),
                                  position_ids     [i].cpu().unsqueeze(0),
                                  masked_lm_labels [i].cpu().numpy()
                              ) for i in range(batch_size)]

          results          = [j.result() for j in jobs]
          samples          = [x.numpy() for (x, _, _) in results]
          compile_flag     = [y         for (_, y, _) in results]
          masked_lm_labels = torch.LongTensor([z for (_, _, z) in results]).to(pytorch.device)
      else:
        compile_flag = [0] * len(input_ids)
        num_targets = sum([x for x in input_ids[0] if x == self.atomizer.maskToken or x == self.atomizer.holeToken])
        sample_indices = [[[] for i in range(num_targets)] for j in range(len(input_ids))]
        there_are_holes, new_input_ids, new_attention_mask, sample_indices = self.fillPredictionSeq(
          input_ids,
          [[self.argmax(x) for x in b] for b in prediction_scores],
          attention_mask,
          sample_indices,
        )
        while there_are_holes:
          new_outputs = self.bert(
            input_ids            = new_input_ids,
            attention_mask       = new_attention_mask,
            position_ids         = position_ids,
            token_type_ids       = token_type_ids,
            head_mask            = head_mask,
            inputs_embeds        = inputs_embeds,
            output_attentions    = output_attentions,
            output_hidden_states = output_hidden_states,
          )
          new_sequence_output, new_pooled_output = new_outputs[:2]
          new_prediction_scores, new_seq_relationship_score = self.cls(new_sequence_output, new_pooled_output)
          there_are_holes, new_input_ids, new_attention_mask, sample_indices = self.fillPredictionSeq(
            new_input_ids,
            [[self.argmax(x) for x in b] for b in new_prediction_scores],
            new_attention_mask,
            sample_indices,
          )
        for i in range(len(new_input_ids)):
          compile_flag[i] = self.checkIfBatchCompiles(new_input_ids[i].cpu().numpy())
          # if compile_flag[i]:
          #   masked_lm_labels[i][:] = -100

    loss_fct = torch.nn.CrossEntropyLoss()
    masked_lm_loss     = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
    next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_labels.view(-1))
    total_loss = masked_lm_loss + next_sentence_loss

    return BertForPreTrainingOutput(
      masked_lm_loss          = masked_lm_loss,
      next_sentence_loss      = next_sentence_loss,
      total_loss              = total_loss,
      prediction_logits       = prediction_scores,
      seq_relationship_logits = seq_relationship_score,
      hidden_states           = outputs[0],
      attentions              = outputs[1],
      compile_status          = compile_flag if compile_samples else [],
      generated_samples       = [x for en, x in enumerate(samples)] if compile_samples else [],
      batch_compilation_rate  = float(sum(compile_flag)) / len(compile_flag) if compile_samples else -1,
      sample_indices          = sample_indices if not is_training and is_prediction else []
    )

class BertLMHeadModel(BertPreTrainedModel):
  def __init__(self, config):
    super().__init__(config)

    if not config.is_decoder:
      l.getLogger().warning("If you want to use `BertLMHeadModel` as a standalone, add `is_decoder=True.`")

    self.bert = BertModel(config)
    self.cls = BertOnlyMLMHead(config)

    self.init_weights()

  def get_output_embeddings(self):
    return self.cls.predictions.decoder

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
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
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
    labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
      Labels for computing the left-to-right language modeling loss (next word prediction).
      Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
      Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
      in ``[0, ..., config.vocab_size]``

  Returns:

  Example::

    >>> from transformers import BertTokenizer, BertLMHeadModel, BertConfig
    >>> import torch

    >>> tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    >>> config = BertConfig.from_pretrained("bert-base-cased")
    >>> config.is_decoder = True
    >>> model = BertLMHeadModel.from_pretrained('bert-base-cased', config=config, return_dict=True)

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    >>> outputs = model(**inputs)

    >>> prediction_logits = outputs.logits
    """
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.bert(
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
    prediction_scores = self.cls(sequence_output)

    lm_loss = None
    if labels is not None:
      # we are doing next-token prediction; shift prediction scores and input ids by one
      shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
      labels = labels[:, 1:].contiguous()
      loss_fct = torch.nn.CrossEntropyLoss()
      lm_loss = loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

    if not return_dict:
      output = (prediction_scores,) + outputs[2:]
      return ((lm_loss,) + output) if lm_loss is not None else output

    # return CausalLMOutput(
    #   loss=lm_loss, logits=prediction_scores, hidden_states=outputs.hidden_states, attentions=outputs.attentions,
    # )

  def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
    input_shape = input_ids.shape

    # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
    if attention_mask is None:
      attention_mask = input_ids.new_ones(input_shape)

    return {"input_ids": input_ids, "attention_mask": attention_mask}

class BertForMaskedLM(BertPreTrainedModel):
  def __init__(self, config):
    super().__init__(config)

    if config.is_decoder:
      l.getLogger().warning(
        "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
        "bi-directional self-attention."
      )

    self.bert = BertModel(config)
    self.cls = BertOnlyMLMHead(config)

    self.init_weights()

  def get_output_embeddings(self):
    return self.cls.predictions.decoder

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
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    **kwargs
  ):
    r"""
    labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
      Labels for computing the masked language modeling loss.
      Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
      Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
      in ``[0, ..., config.vocab_size]``
    kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
      Used to hide legacy arguments that have been deprecated.
    """
    if "masked_lm_labels" in kwargs:
      warnings.warn(
        "The `masked_lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
        FutureWarning,
      )
      labels = kwargs.pop("masked_lm_labels")
    assert "lm_labels" not in kwargs, "Use `BertWithLMHead` for autoregressive language modeling task."
    assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.bert(
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
    prediction_scores = self.cls(sequence_output)

    masked_lm_loss = None
    if labels is not None:
      loss_fct = torch.nn.CrossEntropyLoss()  # -100 index = padding token
      masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

    if not return_dict:
      output = (prediction_scores,) + outputs[2:]
      return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

    # return MaskedLMOutput(
    #   loss=masked_lm_loss,
    #   logits=prediction_scores,
    #   hidden_states=outputs.hidden_states,
    #   attentions=outputs.attentions,
    # )

  def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
    input_shape = input_ids.shape
    effective_batch_size = input_shape[0]

    #  add a dummy token
    assert self.config.pad_token_id is not None, "The PAD token should be defined for generation"
    attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
    dummy_token = torch.full(
      (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
    )
    input_ids = torch.cat([input_ids, dummy_token], dim=1)

    return {"input_ids": input_ids, "attention_mask": attention_mask}

class BertForTokenClassification(BertPreTrainedModel):
  def __init__(self, config):
    super().__init__(config)
    self.num_labels = config.num_labels

    self.bert = BertModel(config)
    self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
    self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

    self.init_weights()

  def forward(
    self,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
  ):
    r"""
    labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
      Labels for computing the token classification loss.
      Indices should be in ``[0, ..., config.num_labels - 1]``.
    """
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.bert(
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
      loss_fct = torch.nn.CrossEntropyLoss()
      # Only keep active parts of the loss
      if attention_mask is not None:
        active_loss = attention_mask.view(-1) == 1
        active_logits = logits.view(-1, self.num_labels)
        active_labels = torch.where(
          active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
        )
        loss = loss_fct(active_logits, active_labels)
      else:
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

    if not return_dict:
      output = (logits,) + outputs[2:]
      return ((loss,) + output) if loss is not None else output

    # return TokenClassifierOutput(
    #   loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions,
    # )
