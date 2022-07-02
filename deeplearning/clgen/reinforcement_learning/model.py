"""
Modeling for reinforcement learning program synthesis.
"""
import pathlib
import typing
import math

from deeplearning.clgen.reinforcement_learning import interactions
from deeplearning.clgen.reinforcement_learning import data_generator
from deeplearning.clgen.reinforcement_learning import config
from deeplearning.clgen.models.torch_bert import model
from deeplearning.clgen.models import language_models
from deeplearning.clgen.corpuses import tokenizers
from deeplearning.clgen.util import environment
from deeplearning.clgen.util import pytorch
from deeplearning.clgen.util import logging as l

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
              input_features_key_padding_mask : torch.ByteTensor,
              ) -> torch.Tensor:
    ## Run the encoder transformer over the features.
    enc_embed = self.encoder_embedding(input_features) * math.sqrt(self.encoder_embedding_size)
    pos_enc_embed = self.encoder_pos_encoder(enc_embed)
    encoded = self.encoder_transformer(
      pos_enc_embed,
      src_key_padding_mask = input_features_key_padding_mask,
    )
    return encoded

class PredictionHeadTransform(torch.nn.Module):
  def __init__(self, config, dense_size):
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
              input_ids_key_padding_mask : torch.ByteTensor,
              ) -> torch.Tensor:
    ## Run the decoder over the source code.
    dec_embed = self.decoder_embedding(input_ids)
    decoded = self.decoder_transformer(
      dec_embed,
      memory = encoded_features,
      # tgt_mask = input_ids_mask,
      # memory_mask = None, ??
      tgt_key_padding_mask = input_ids_key_padding_mask,
      # memory_key_padding_mask = None, ??
    )
    return decoded

class ActionHead(torch.nn.Module):
  """Classification head for action prediction."""
  def __init__(self, config):
    super().__init__()
    self.transform = PredictionHeadTransform(config, dense_size = config.hidden_size)
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
    self.transform = PredictionHeadTransform(config, dense_size = config.hidden_size + len(interactions.ACTION_TYPE_SPACE))
    self.decoder   = torch.nn.Linear(config.hidden_size, config.max_position_embeddings, bias = False)
    self.bias      = torch.nn.Parameter(torch.zeros(config.max_position_embeddings))
    self.decoder.bias = self.bias
    return

  def forward(self, decoder_out, action_logits):
    decoded_with_action = torch.cat((decoder_out, action_logits), -1)
    transformed = self.transform(decoded_with_action)
    action_logits = self.decoder(transformed)
    return action_logits

class TokenHead(torch.nn.Module):
  """Feature-extended token prediction head."""
  def __init__(self, config):
    super().__init__()
    self.transform = PredictionHeadTransform(config, dense_size = config.vocab_size + config.feature_sequence_length)
    self.decoder   = torch.nn.Linear(config.vocab_size + config.feature_sequence_length, config.vocab_size, bias = False)
    self.bias      = torch.nn.Parameter(torch.zeros(config.vocab_size))
    self.decoder.bias = self.bias
    return

  def forward(self, decoder_out: torch.FloatTensor) -> torch.FloatTensor:
    transformed = self.transform(decoder_out)
    token_logits = self.decoder(transformed)
    return token_logits

class ActionQV(torch.nn.Module):
  """Deep Q-Values for Action type prediction."""
  def __init__(self, config):
    super().__init__()
    self.feature_encoder = FeatureEncoder(config)
    self.source_decoder  = SourceDecoder(config)
    self.action_head     = ActionHead(config)
    self.index_head      = IndexHead(config)


    """
    ## Pre-trained Encoder LM.
    self.feature_encoder = language_model.backend.GetEncoderModule(
      #### Add new parameters here.
    )
    ## Decoder for token prediction, given features and source code encoded memory.
    self.source_decoder = language_model.backend.GetDecoderModule(with_checkpoint = True)
    """
    return

  def forward(self,
              input_ids                        : torch.LongTensor,
              target_features                  : torch.LongTensor,
              input_ids_key_padding_mask       : torch.ByteTensor,
              target_features_key_padding_mask : torch.ByteTensor,

              ) -> typing.Dict[str, torch.Tensor]:
    """Action type forward function."""
    ## Encode feature vector.
    encoded_features = self.feature_encoder(
      target_features, target_features_key_padding_mask
    )
    ## Run source code over transformer decoder and insert Transformer encoder's memory.
    decoded_source = self.source_decoder(
      input_ids, encoded_features, input_ids_key_padding_mask,
    )
    ## Predict the action logits.
    action_logits = self.action_head(decoded_source)
    ## Predict the index logits.
    index_logits  = self.index_head(decoded_source, action_logits)

    """
    ## Run BERT-Encoder in target feature vector.
    encoded_features = self.feature_encoder(
      input_ids      = encoder_feature_ids,
      input_mask     = encoder_feature_mask,
      position_ids   = encoder_position_ids,
      input_features = None,
    )
    ## Run source code over pre-trained BERT decoder.
    decoded_source = self.source_decoder(
      input_ids      = decoder_input_ids,
      input_mask     = decoder_input_mask,
      position_ids   = decoder_position_ids,
      input_features = None,
    )
    ## Predict action type logits.
    action_logits = self.action_head(decoded_source)
    ## Predict index logits | action type logits.
    index_logits = self.index_head(decoded_source, action_logits)
    """
    return action_logits, index_logits

class ActionLanguageModelQV(torch.nn.Module):
  """Deep Q-Values for Token type prediction."""
  def __init__(self, language_model: language_models.Model, config: config.QValuesConfig):
    super(ActionLanguageModelQV, self).__init__()
    ## Pre-trained Encoder LM.
    self.language_model = language_model.backend.GetEncoderModule(with_checkpoint = True)
    ## Decoder for token prediction, given features and source code encoded memory.
    self.decoder = language_model.backend.GetDecoderModule(
      config.feature_sequence_length, config.vocab_size, config.feature_pad_idx
    )
    return

  def forward(self,
              encoder_input_ids,
              encoder_input_mask,
              encoder_position_ids,
              decoder_feature_ids,
              decoder_feature_mask,
              decoder_position_ids,
              encoder_input_features = None,
              ):
    encoder_out = self.language_model(
      input_ids      = encoder_input_ids,
      input_mask     = encoder_input_mask,
      position_ids   = encoder_position_ids,
      input_features = encoder_input_features,
    )
    encoder_memory = encoder_out['hidden_states']
    decoder_out = self.decoder(
      input_ids             = decoder_feature_ids,
      input_mask            = decoder_feature_mask,
      position_ids          = decoder_position_ids,
      encoder_hidden_states = encoder_memory,
    )
    return decoder_out

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
  class QValuesEstimator(typing.NamedTuple):
    """Torch model wrapper for Deep Q-Values."""
    action_qv : torch.nn.Module
    token_qv  : torch.nn.Module

  def __init__(self,
               language_model          : language_models.Model,
               feature_tokenizer       : tokenizers.FeatureTokenizer,
               config                  : config.QValuesConfig,
               cache_path              : pathlib.Path
               ) -> None:
    self.cache_path = cache_path / "DQ_model"
    if environment.WORLD_RANK == 0:
      self.cache_path.mkdir(exist_ok = True, parents = True)

    self.config                  = config
    self.language_model          = language_model
    self.tokenizer               = language_model.tokenizer
    self.feature_tokenizer       = feature_tokenizer
    self.feature_sequence_length = self.config.feature_sequence_length

    self.train_qvalues  = None
    self.sample_qvalues = None
    return

  def _ConfigModelParams(self) -> QValuesEstimator:
    """Initialize model parameters."""
    actm = ActionQV(self.config).to(pytorch.offset_device)
    tokm = ActionLanguageModelQV(self.language_model, self.config).to(pytorch.offset_device)

    if pytorch.num_nodes > 1:
      actm = torch.nn.parallel.DistributedDataParallel(
        actm,
        device_ids = [pytorch.offset_device],
        output_device = pytorch.offset_device,
        find_unused_parameters = True,
      )
      tokm = torch.nn.parallel.DistributedDataParallel(
        tokm,
        device_ids = [pytorch.offset_device],
        output_device = pytorch.offset_device,
        find_unused_parameters = True,
      )
    elif pytorch.num_gpus > 1:
      actm = torch.nn.DataParallel(actm)
      tokm = torch.nn.DataParallel(tokm)

    return QValuesModel.QValuesEstimator(
      action_qv = actm,
      token_qv  = tokm,
    )

  def _ConfigTrainParams(self) -> None:
    """Initialize Training parameters for model."""
    if not self.train_qvalues:
      self.train_qvalues = self._ConfigModelParams()
    return
  
  def _ConfigSampleParams(self) -> None:
    """Initialize sampling parameters for model."""
    if not self.sample_qvalues:
      self.sample_qvalues = self._ConfigModelParams()
    return

  def Train(self, input_ids: typing.Dict[str, torch.Tensor]) -> None:
    """Update the Q-Networks with some memories."""
    self._ConfigTrainParams()
    self.loadCheckpoint()
    raise NotImplementedError
    return

  def SampleAction(self, state: interactions.State) -> typing.Dict[str, torch.Tensor]:
    """Predict the next action given an input state."""
    self._ConfigSampleParams()
    inputs = data_generator.StateToActionTensor(
      state, self.tokenizer.padToken, self.feature_tokenizer.padToken
    )
    return self.sample_qvalues.action_qv(
      **{k: v.to(pytorch.device) for k, v in inputs.items()}
    )
  
  def SampleToken(self,
                  state          : interactions.State,
                  mask_idx       : int,
                  tokenizer      : tokenizers.TokenizerBase,
                  feat_tokenizer : tokenizers.FeatureTokenizer,
                  ) -> typing.Dict[str, torch.Tensor]:
    """Predict token type"""
    self._ConfigSampleParams()
    inputs = data_generator.StateToTokenTensor(
      state, mask_idx, tokenizer.holeToken, tokenizer.padToken, feat_tokenizer.padToken
    )
    return self.sample_qvalues.token_qv(
      **{k: v.to(pytorch.device) for k, v in inputs.items()}
    )

  def saveCheckpoint(self) -> None:
    """Checkpoint Deep Q-Nets."""
    l.logger().error("Save checkpoint for QV Model has not been implemented.")
    return

  def loadCheckpoint(self) -> None:
    """Load Deep Q-Nets."""
    l.logger().error("Load checkpoint for QV Model has not been implemented.")
    return
