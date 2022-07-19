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
    return {
      'action_logits': action_logits,
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
      hidden_size                  = config.token_hidden_size,
      num_hidden_layers            = config.token_num_hidden_layers,
      num_attention_heads          = config.token_num_attention_heads,
      intermediate_size            = config.token_intermediate_size,
      hidden_act                   = config.token_hidden_act,
      hidden_dropout_prob          = config.token_hidden_dropout_prob,
      attention_probs_dropout_prob = config.token_attention_probs_dropout_prob,
      max_position_embeddings      = config.feature_sequence_length,
      type_vocab_size              = config.token_type_vocab_size,
      initializer_range            = config.token_initializer_range,
      layer_norm_eps               = config.token_layer_norm_eps,
      pad_token_id                 = config.feature_pad_idx,
      with_checkpoint              = False,
    )
    ## Decoder for token prediction, given features memory and source code.
    self.language_model = language_model.backend.GetDecoderModule(
      with_checkpoint = True
    )
    if is_critic:
      output_dim = 1
    else:
      output_dim = config.vocab_size
    self.decoder = TokenHead(config, output_dim)
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
    encoder_out = self.language_model(
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
    outputs = self.decoder(decoder_out)
    return outputs

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

  class QValuesEstimator(typing.NamedTuple):
    """Torch model wrapper for Deep Q-Values."""
    action : torch.nn.Module
    token  : torch.nn.Module

  def __init__(self,
               language_model    : language_models.Model,
               feature_tokenizer : tokenizers.FeatureTokenizer,
               config            : config.QValuesConfig,
               cache_path        : pathlib.Path,
               is_critic         : bool,
               ) -> None:
    self.cache_path = cache_path / "DQ_model"
    self.ckpt_path  = cache_path / "checkpoint"
    self.log_path   = cache_path / "logs"
    if environment.WORLD_RANK == 0:
      self.cache_path.mkdir(exist_ok = True, parents = True)
      self.ckpt_path.mkdir (exist_ok = True, parents = True)
      self.log_path.mkdir  (exist_ok = True, parents = True)
    self.ckpt_step = 0

    self.config                  = config
    self.language_model          = language_model
    self.tokenizer               = language_model.tokenizer
    self.feature_tokenizer       = feature_tokenizer
    self.feature_sequence_length = self.config.feature_sequence_length
    self.is_critic               = is_critic
    self.batch_size              = self.config.batch_size

    self.model = None
    self._ConfigModelParams()
    return

  def _ConfigModelParams(self) -> QValuesEstimator:
    """Initialize model parameters."""
    if not self.model:
      actm    = ActionQV(self.language_model, self.config, self.is_critic).to(pytorch.offset_device)
      tokm    = ActionLanguageModelQV(self.language_model, self.config, self.is_critic).to(pytorch.offset_device)
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
        actm    = torch.nn.DataParallel(actm)
        tokm    = torch.nn.DataParallel(tokm)

        self.model = QValuesModel.QValuesEstimator(
          action = actm,
          token  = tokm,
        )
    return

  def Train(self, input_ids: typing.Dict[str, torch.Tensor]) -> None:
    """Update the Q-Networks with some memories."""
    self._ConfigModelParams()
    self.loadCheckpoint()
    raise NotImplementedError
    return

  def SampleAction(self,
                   state               : interactions.State,
                   ) -> typing.Dict[str, torch.Tensor]:
    """Predict the next action given an input state."""
    self._ConfigModelParams()
    inputs = data_generator.StateToActionTensor(
      state, self.tokenizer.padToken, self.feature_tokenizer.padToken, self.batch_size
    )
    with torch.no_grad():
      inputs = {k: v.to(pytorch.device) for k, v in inputs.items()}
      outputs = self.model.action(
        **inputs
      )
    return outputs

  def SampleToken(self,
                  state          : interactions.State,
                  mask_idx       : int,
                  tokenizer      : tokenizers.TokenizerBase,
                  feat_tokenizer : tokenizers.FeatureTokenizer,
                  replace_token  : bool = False,
                  ) -> typing.Dict[str, torch.Tensor]:
    """Predict token type"""
    self._ConfigModelParams()
    inputs = data_generator.StateToTokenTensor(
      state,
      mask_idx,
      tokenizer.holeToken,
      tokenizer.padToken,
      feat_tokenizer.padToken,
      self.batch_size,
      replace_token = replace_token,
    )
    with torch.no_grad():
      inputs = {k: v.to(pytorch.device) for k, v in inputs.items()}
      outputs = self.model.token(
        **inputs
      )
    return outputs

  def saveCheckpoint(self, prefix = "") -> None:
    """Checkpoint Deep Q-Nets."""
    if self.is_world_process_zero():
      ckpt_comp = lambda x: self.ckpt_path / "{}{}_model-{}.pt".format(prefix, x, self.ckpt_step)
      if self.torch_tpu_available:
        if self.pytorch.torch_xla_model.rendezvous("saving_checkpoint"):
          self.pytorch.torch_xla_model.save(self.model.action, ckpt_comp("action"))
          self.pytorch.torch_xla_model.save(self.model.token, ckpt_comp("token"))
        self.pytorch.torch_xla.rendezvous("saving_optimizer_states")
      else:
        if isinstance(self.model.action, self.torch.nn.DataParallel):
          self.torch.save(self.model.action.module.state_dict(), ckpt_comp("action"))
        else:
          self.torch.save(self.model.action.state_dict(), ckpt_comp("action"))
        if isinstance(self.model.token, self.torch.nn.DataParallel):
          self.torch.save(self.model.token.module.state_dict(), ckpt_comp("token"))
        else:
          self.torch.save(self.model.token.state_dict(), ckpt_comp("token"))
      with open(self.ckpt_path / "{}checkpoint.meta".format(prefix), 'a') as mf:
        mf.write("train_step: {}\n".format(self.ckpt_step))
    self.ckpt_step += 1
    torch.distributed.barrier()
    return

  def loadCheckpoint(self, prefix = "") -> None:
    """Load Deep Q-Nets."""
    if not (self.ckpt_path / "{}checkpoint.meta".format(prefix)).exists():
      return -1
    with open(self.ckpt_path / "{}checkpoint.meta".format(prefix), 'w') as mf:
      get_step = lambda x: int(x.replace("\n", "").replace("train_step: ", ""))
      lines = mf.readlines()
      entries = set({get_step(x) for x in lines})

    ckpt_step = max(entries)
    ckpt_comp = lambda x: self.ckpt_path / "{}{}_model-{}.pt".format(prefix, x, ckpt_step)

    if isinstance(self.model.action, torch.nn.DataParallel):
      try:
        self.model.action.module.load_state_dict(torch.load(ckpt_comp("action")))
      except RuntimeError:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in torch.load(ckpt_comp("action")).items():
          if k[:7] == "module.":
            name = k[7:]
          else:
            name = "module." + k
          new_state_dict[name] = k
        self.model.action.module.load_state_dict(new_state_dict)
    else:
      try:
        self.model.action.module.load_state_dict(torch.load(ckpt_comp("action")))
      except RuntimeError:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in self.torch.load(ckpt_comp("action")).items():
          if k[:7] == 'module.':
            name = k[7:] # remove `module.`
          else:
            name = 'module.' + k # Add 'module.'
          new_state_dict[name] = v
        self.model.action.load_state_dict(new_state_dict)

    if isinstance(self.model.token, torch.nn.DataParallel):
      try:
        self.model.token.module.load_state_dict(torch.load(ckpt_comp("token")))
      except RuntimeError:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in torch.load(ckpt_comp("token")).items():
          if k[:7] == "module.":
            name = k[7:]
          else:
            name = "module." + k
          new_state_dict[name] = k
        self.model.token.module.load_state_dict(new_state_dict)
    else:
      try:
        self.model.token.module.load_state_dict(torch.load(ckpt_comp("token")))
      except RuntimeError:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in self.torch.load(ckpt_comp("token")).items():
          if k[:7] == 'module.':
            name = k[7:] # remove `module.`
          else:
            name = 'module.' + k # Add 'module.'
          new_state_dict[name] = v
        self.model.token.load_state_dict(new_state_dict)
    self.model.action.eval()
    self.model.token.eval()
    return ckpt_step

  def is_world_process_zero(self) -> bool:
    """
    Whether or not this process is the global main process (when training in a distributed fashion on
    several machines, this is only going to be :obj:`True` for one process).
    """
    if torch_tpu_available:
      return pytorch.torch_xla_model.is_master_ordinal(local=False)
    elif pytorch.num_nodes > 1:
      return torch.distributed.get_rank() == 0
    else:
      return True
