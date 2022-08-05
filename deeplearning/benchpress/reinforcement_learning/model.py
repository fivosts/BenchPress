# coding=utf-8
# Copyright 2022 Foivos Tsimpourlas.
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
"""
Modeling for reinforcement learning program synthesis.
"""
import pathlib
import typing
import math

from deeplearning.benchpress.reinforcement_learning import interactions
from deeplearning.benchpress.reinforcement_learning import data_generator
from deeplearning.benchpress.reinforcement_learning import config
from deeplearning.benchpress.models.torch_bert import model
from deeplearning.benchpress.models import language_models
from deeplearning.benchpress.corpuses import tokenizers
from deeplearning.benchpress.util import environment
from deeplearning.benchpress.util import pytorch
from deeplearning.benchpress.util import logging as l

torch = pytorch.torch

class PredictionHeadTransform(torch.nn.Module):
  def __init__(self,
               config     : config.QValuesConfig,
               dense_size : int
               ):
    super().__init__()
    self.dense = torch.nn.Linear(dense_size, config.hidden_size)
    if isinstance(config.hidden_act, str):
      self.transform_act_fn = model.ACT2FN[config.hidden_act]
    else:
      self.transform_act_fn = config.hidden_act
    self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

  def forward(self, hidden_states):
    hidden_states = self.dense(hidden_states)
    hidden_states = self.transform_act_fn(hidden_states)
    hidden_states = self.LayerNorm(hidden_states)
    return hidden_states

class ActionHead(torch.nn.Module):
  """Classification head for action prediction."""
  def __init__(self, config, output_dim: int = None):
    super().__init__()
    if output_dim is None:
      output_dim = len(interactions.ACTION_TYPE_SPACE) * config.max_position_embeddings
    self.transform = PredictionHeadTransform(config, dense_size = config.hidden_size)
    self.decoder   = torch.nn.Linear(config.hidden_size * config.max_position_embeddings, output_dim, bias = False)
    self.bias      = torch.nn.Parameter(torch.zeros(output_dim))
    self.decoder.bias = self.bias
    return

  def forward(self, decoder_out: torch.FloatTensor) -> torch.FloatTensor:
    transformed = self.transform(decoder_out)
    flat = transformed.reshape((transformed.shape[0], -1))
    action_logits = self.decoder(flat)
    return action_logits

class TokenHead(torch.nn.Module):
  """Classification head for token prediction."""
  def __init__(self, config, output_dim: int):
    super().__init__()
    self.transform = PredictionHeadTransform(config, dense_size = config.hidden_size)
    self.decoder   = torch.nn.Linear(config.hidden_size, output_dim, bias = False)
    self.bias      = torch.nn.Parameter(torch.zeros(output_dim))
    self.decoder.bias = self.bias
    return
  
  def forward(self, decoder_out: torch.FloatTensor) -> torch.FloatTensor:
    hidden_states = self.transform(decoder_out)
    token_logits = self.decoder(hidden_states)
    return token_logits

class ActionQV(torch.nn.Module):
  """Deep Q-Values for Action type prediction."""
  def __init__(self,
               language_model : language_models.Model,
               config         : config.QValuesConfig,
               is_critic      : bool = False
               ):
    super().__init__()
    ## Pre-trained Encoder LM.
    self.feature_encoder = language_model.backend.GetEncoderModule(
      vocab_size                   = config.feature_vocab_size,
      hidden_size                  = config.hidden_size,
      num_hidden_layers            = config.num_hidden_layers,
      num_attention_heads          = config.num_attention_heads,
      intermediate_size            = config.intermediate_size,
      hidden_act                   = config.hidden_act,
      hidden_dropout_prob          = config.hidden_dropout_prob,
      attention_probs_dropout_prob = config.attention_probs_dropout_prob,
      max_position_embeddings      = config.feature_sequence_length,
      type_vocab_size              = config.type_vocab_size,
      initializer_range            = config.initializer_range,
      layer_norm_eps               = config.layer_norm_eps,
      pad_token_id                 = config.feature_pad_idx,
      with_checkpoint              = False,
    )
    ## Decoder for token prediction, given features and source code encoded memory.
    self.source_decoder = language_model.backend.GetDecoderModule(
      with_checkpoint    = True,
      without_label_head = True,
    )
    output_dim = None
    if is_critic:
      output_dim = 1
    self.action_head = ActionHead(config, output_dim = output_dim)
    self.softmax     = torch.nn.Softmax(dim = -1)
    return

  def forward(self,
              encoder_feature_ids  : torch.LongTensor,
              encoder_feature_mask : torch.LongTensor,
              encoder_position_ids : torch.LongTensor,
              decoder_input_ids    : torch.LongTensor,
              decoder_input_mask   : torch.LongTensor,
              decoder_position_ids : torch.LongTensor,
              # actor_action_logits  : torch.LongTensor = None,
              ) -> typing.Dict[str, torch.Tensor]:
    """Action type forward function."""
    ## Run BERT-Encoder in target feature vector.
    encoder_out = self.feature_encoder(
      input_ids      = encoder_feature_ids,
      input_mask     = encoder_feature_mask,
      position_ids   = encoder_position_ids,
      input_features = None,
    )
    encoder_memory = encoder_out['hidden_states']
    ## Run source code over pre-trained BERT decoder.
    decoder_out = self.source_decoder(
      input_ids             = decoder_input_ids,
      input_mask            = decoder_input_mask,
      position_ids          = decoder_position_ids,
      encoder_hidden_states = encoder_memory,
      input_features        = None,
    )
    decoded_source = decoder_out['hidden_states']
    ## Predict action type logits.
    action_logits = self.action_head(decoded_source)
    action_probs  = self.softmax(action_logits)
    return {
      'action_logits' : action_logits,
      'action_probs'  : action_probs,
    }

class ActionLanguageModelQV(torch.nn.Module):
  """Deep Q-Values for Token type prediction."""
  def __init__(self,
               language_model : language_models.Model,
               config         : config.QValuesConfig,
               is_critic      : bool = False,
               ):
    super(ActionLanguageModelQV, self).__init__()
    ## Feature-Encoder.
    self.encoder = language_model.backend.GetEncoderModule(
      vocab_size                   = config.feature_vocab_size,
      hidden_size                  = config.hidden_size,
      num_hidden_layers            = config.num_hidden_layers,
      num_attention_heads          = config.num_attention_heads,
      intermediate_size            = config.intermediate_size,
      hidden_act                   = config.hidden_act,
      hidden_dropout_prob          = config.hidden_dropout_prob,
      attention_probs_dropout_prob = config.attention_probs_dropout_prob,
      max_position_embeddings      = config.feature_sequence_length,
      type_vocab_size              = config.type_vocab_size,
      initializer_range            = config.initializer_range,
      layer_norm_eps               = config.layer_norm_eps,
      pad_token_id                 = config.feature_pad_idx,
      with_checkpoint              = False,
    )
    ## Decoder for token prediction, given features memory and source code.
    if is_critic:
      output_dim = 1
      self.language_model = language_model.backend.GetDecoderModule(
        with_checkpoint = True,
        without_label_head = True,
      )
      self.decoder = TokenHead(config, output_dim)
    else:
      output_dim = config.vocab_size
      self.language_model = language_model.backend.GetDecoderModule(
        with_checkpoint = True,
      )
    self.softmax   = torch.nn.Softmax(dim = -1)
    self.is_critic = is_critic
    return

  def forward(self,
              encoder_feature_ids  : torch.LongTensor,
              encoder_feature_mask : torch.LongTensor,
              encoder_position_ids : torch.LongTensor,
              decoder_input_ids    : torch.LongTensor,
              decoder_input_mask   : torch.LongTensor,
              decoder_position_ids : torch.LongTensor,
              encoder_input_features = None,
              ):
    encoder_out = self.encoder(
      input_ids      = encoder_feature_ids,
      input_mask     = encoder_feature_mask,
      position_ids   = encoder_position_ids,
      input_features = encoder_input_features,
    )
    encoder_memory = encoder_out['hidden_states']
    decoder_out = self.language_model(
      input_ids             = decoder_input_ids,
      input_mask            = decoder_input_mask,
      position_ids          = decoder_position_ids,
      encoder_hidden_states = encoder_memory,
    )
    if self.is_critic:
      decoded_source = decoder_out['hidden_states']
      token_logits = self.decoder(decoded_source)
    else:
      token_logits = decoder_out['prediction_logits']
    token_probs  = self.softmax(token_logits)
    return {
      'token_logits' : token_logits,
      'token_probs'  : token_probs,
    }

class QValuesModel(object):
  """
  Handler of Deep-QNMs for program synthesis.
  """
  @property
  def action_parameters(self) -> torch.Tensor:
    """
    Return all gradient parameters for model involved in action decision.
    """
    if self.model:
      if isinstance(self.model.action, torch.nn.DataParallel):
        module = self.model.action.module
      else:
        module = self.model.action
      return (
        [x for x in module.feature_encoder.parameters()] +
        [x for x in module.source_decoder.parameters()] +
        [x for x in module.action_head.parameters()]
      )
    else:
      return None

  @property
  def index_parameters(self) -> torch.Tensor:
    """
    Return all gradient parameters for model involved in action decision.
    """
    if self.model:
      if isinstance(self.model.action, torch.nn.DataParallel):
        module = self.model.action.module
      else:
        module = self.model.action
      return (
        [x for x in module.feature_encoder.parameters()] +
        [x for x in module.source_decoder.parameters()] +
        [x for x in module.index_head.parameters()]
      )
    else:
      return None

  @property
  def token_parameters(self) -> torch.Tensor:
    """
    Return all gradient parameters for model involved in action decision.
    """
    if self.model:
      if isinstance(self.model.token, torch.nn.DataParallel):
        module = self.model.token.module
      else:
        module = self.model.token
      return (
        [x for x in module.encoder.parameters()] +
        [x for x in module.language_model.parameters()]
      )
    else:
      return None
