"""
Modeling for reinforcement learning program synthesis.
"""
import pathlib
import typing
import math

from deeplearning.clgen.reinforcement_learning import interactions
from deeplearning.clgen.models.torch_bert import model
from deeplearning.clgen.util import environment
from deeplearning.clgen.util import pytorch

torch = pytorch.torch

class FeatureEncoder(torch.nn.Module):
  """Transformer-Encoder architecture for encoding states."""
  def __init__(self, config):
    super().__init__()
    ## Dimensional variables.
    self.encoder_embedding_size = config.feature_vocab_size
    ## Feature vector encoder.
    self.encoder_embedding = torch.nn.Embedding(
      num_embeddings = config.feature_vocab_size,
      embedding_dim  = config.hidden_size,
      padding_idx    = config.feature_pad_idx
    )
    self.encoder_pos_encoder = model.FeaturePositionalEncoding(config)
    encoder_layers = torch.nn.TransformerEncoderLayer(
      d_model         = config.hidden_size,
      nhead           = config.num_attention_heads,
      dim_feedforward = config.intermediate_size,
      dropout         = config.hidden_dropout_prob,
      batch_first     = True
    )
    encoder_norm = torch.nn.LayerNorm(
      config.hidden_size,
      eps = config.layer_norm_eps
    )
    self.encoder_transformer = torch.nn.TransformerEncoder(
      encoder_layer = encoder_layers,
      num_layers    = config.num_hidden_layers,
      norm          = encoder_norm,
    )
    return

  def forward(self,
              input_features                  : torch.LongTensor,
              input_features_mask             : torch.ByteTensor,
              input_features_key_padding_mask : torch.ByteTensor,
              ) -> torch.Tensor:
    ## Run the encoder transformer over the features.
    enc_embed = self.encoder_embedding(input_features) * math.sqrt(self.encoder_embedding_size)
    pos_enc_embed = self.encoder_pos_encoder(enc_embed)
    encoded = self.encoder_transformer(
      pos_enc_embed,
      mask = input_features_mask,
      src_key_padding_mask = input_features_key_padding_mask,
    )
    return encoded

class SourceDecoder(torch.nn.Module):
  """Source code decoder for action prediction."""
  def __init__(self, config):
    super().__init__()
    ## Source code decoder.
    self.decoder_embedding = model.BertEmbeddings(config)
    decoder_layers = torch.nn.TransformerDecoderLayer(
      d_model         = config.hidden_size,
      nhead           = config.num_attention_heads,
      dim_feedforward = config.intermediate_size,
      dropout         = config.hidden_dropout_prob,
      batch_first     = True,
    )
    decoder_norm = torch.nn.LayerNorm(
      config.hidden_size,
      eps = config.layer_norm_eps,
    )
    self.decoder_transformer = torch.nn.TransformerDecoder(
      decoder_layer = decoder_layers,
      num_layers    = config.num_hidden_layers,
      norm          = decoder_norm,
    )
    return

  def forward(self,
              input_ids                  : torch.LongTensor,
              encoded_features           : torch.FloatTensor,
              input_ids_mask             : torch.ByteTensor,
              input_ids_key_padding_mask : torch.ByteTensor,
              ) -> torch.Tensor:
    ## Run the decoder over the source code.
    dec_embed = self.decoder_embedding(input_ids)
    decoded = self.decoder_transformer(
      dec_embed,
      memory = encoded_features,
      tgt_mask = input_ids_mask,
      # memory_mask = None, ??
      tgt_key_padding_mask = input_ids_key_padding_mask,
      # memory_key_padding_mask = None, ??
    )
    return decoded

class ActionHead(torch.nn.Module):
  """Classification head for action prediction."""
  def __init__(self, config):
    super().__init__()
    self.transform = model.BertPredictionHeadTransform(config)
    self.decoder   = torch.nn.Linear(config.hidden_size, len(interactions.ACTION_TYPE_SPACE), bias = False)
    self.bias      = torch.nn.Parameter(torch.zeros(len(interactions.ACTION_TYPE_SPACE)))
    self.decoder.bias = self.bias
    return

  def forward(self, decoder_out: torch.FloatTensor) -> torch.FloatTensor:
    transformed = self.transform(decoder_out)
    action_logits = self.decoder(transformed)
    return action_logits

class IndexHead(torch.nn.Module):
  """Classification head for token index prediction."""
  def __init__(self, config):
    super().__init__()
    return

  def forward(self, decoder_out, action_logits):
    return

class ActionQV(torch.nn.Module):
  """Deep Q-Values for Action type prediction."""
  def __init__(self, config):
    self.feature_encoder = FeatureEncoder(config)
    self.source_decoder  = SourceDecoder(config)
    self.action_head     = ActionHead(config)
    self.index_head      = IndexHead(config)
    return

  def forward(self,
              input_ids                        : torch.LongTensor,
              target_features                  : torch.LongTensor,
              input_ids_mask                   : torch.ByteTensor,
              input_ids_key_padding_mask       : torch.ByteTensor,
              target_features_mask             : torch.ByteTensor,
              target_features_key_padding_mask : torch.ByteTensor,
              ) -> typing.Dict[str, torch.Tensor]:
    """Action type forward function."""
    encoded_features = self.feature_encoder(
      target_features, target_features_mask, target_features_key_padding_mask
    )
    decoded_source = self.source_decoder(
      input_ids, encoded_features, input_ids_mask, input_ids_key_padding_mask,
    )
    action_logits = self.action_head(decoded_source)
    index_logits  = self.index_head(decoded_source, action_logits)
    return action_logits, index_logits

class ActionLanguageModelQV(object):
  """Deep Q-Values for Token type prediction."""
  def __init__(self, language_model, config):
    super(ActionLanguageModelQV, self).__init__()
    self.language_model = language_model
    self.config = config
    return
  
  def TrainBatch(self, input_ids: typing.Dict[str, torch.Tensor]):
    self.language_model.Trainbatch(input_ids)
    return

  def SampleBatch(self, input_ids: typing.Dict[str, torch.Tensor]):
    self.language_model.SampleBatch(input_ids)
    return

class QValuesModel(object):
  """
  Handler of Deep-QNMs for program synthesis.
  """
  def __init__(self, language_model, config, cache_path: pathlib.Path) -> None:
    self.cache_path = cache_path / "DQ_model"
    if environment.WORLD_RANK == 0:
      self.cache_path.mkdir(exists_ok = True, parents = True)

    self.config = config
    self.action_type_qv = ActionQV(config)
    self.token_type_qv  = ActionLanguageModelQV(language_model, config)
    self.loadCheckpoint()
    return

  def Train(self, input_ids: typing.Dict[str, torch.Tensor]) -> None:
    """Update the Q-Networks with some memories."""
    raise NotImplementedError
    return

  def SampleActionType(self, input_ids: typing.Dict[str, torch.Tensor]) -> typing.Dict[str, torch.Tensor]:
    """Predict the next action given an input state."""
    return self.action_type_qv(input_ids)
  
  def SampleActionIndex(self, input_ids: typing.Dict[str, torch.Tensor]) -> typing.Dict[str, torch.Tensor]:
    """Predict Action index"""
    return self.action_index_qv(input_ids)
  
  def SampleTokenType(self, input_ids: typing.Dict[str, torch.Tensor]) -> typing.Dict[str, torch.Tensor]:
    """Predict token type"""
    return self.token_type_qv(input_ids)

  def saveCheckpoint(self) -> None:
    """Checkpoint Deep Q-Nets."""
    raise NotImplementedError
    return

  def loadCheckpoint(self) -> None:
    """Load Deep Q-Nets."""
    raise NotImplementedError
    return
