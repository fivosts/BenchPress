"""
Array of NN models used for Active Learning Query-By-Committee.

This module handles
a) the passive training of the committee,
b) the confidence level of the committee for a datapoint (using entropy)
"""
import typing
import pathlib

from deeplearning.clgen.models import backends
from deeplearning.clgen.util.pytorch import torch

class ActiveCommittee(backends.BackendBase):

  def __init__(self, *args, **kwargs):

    super(torchBert, self).__init__(*args, **kwargs)
    
    from deeplearning.clgen.util import pytorch
    if not pytorch.initialized:
      pytorch.initPytorch()

    self.pytorch             = pytorch
    self.torch               = pytorch.torch
    self.torch_tpu_available = pytorch.torch_tpu_available

    self.torch.manual_seed(self.config.training.random_seed)
    self.torch.cuda.manual_seed_all(self.config.training.random_seed)

    self.ckpt_path         = self.cache.path / "checkpoints"
    self.sample_path       = self.cache.path / "samples"

    self.logfile_path      = self.cache.path / "logs"

    self.is_validated      = False
    self.trained           = False
    l.logger().info("Active Committee config initialized in {}".format(self.cache.path))
    return

  def _ConfigModelParams(self, is_sampling):
    """General model hyperparameters initialization."""
    raise NotImplementedError
    self.bertAttrs = {
          "vocab_size"                   : self.tokenizer.vocab_size,
          "hidden_size"                  : self.config.architecture.hidden_size,
          "num_hidden_layers"            : self.config.architecture.num_hidden_layers,
          "num_attention_heads"          : self.config.architecture.num_attention_heads,
          "intermediate_size"            : self.config.architecture.intermediate_size,
          "hidden_act"                   : self.config.architecture.hidden_act,
          "hidden_dropout_prob"          : self.config.architecture.hidden_dropout_prob,
          "attention_probs_dropout_prob" : self.config.architecture.attention_probs_dropout_prob,
          "max_position_embeddings"      : self.config.architecture.max_position_embeddings,
          "type_vocab_size"              : self.config.architecture.type_vocab_size,
          "initializer_range"            : self.config.architecture.initializer_range,
          "layer_norm_eps"               : self.config.architecture.layer_norm_eps,
          "pad_token_id"                 : self.tokenizer.padToken,
    }
    if self.feature_encoder:
      self.featureAttrs = {
        "feature_encoder"                 : self.feature_encoder,
        "feature_sequence_length"         : self.feature_sequence_length,
        "feature_embedding_size"          : self.config.architecture.feature_embedding_size,
        "feature_pad_idx"                 : self.feature_tokenizer.padToken,
        "feature_dropout_prob"            : self.config.architecture.feature_dropout_prob,
        "feature_vocab_size"              : len(self.feature_tokenizer),
        "feature_num_attention_heads"     : self.config.architecture.feature_num_attention_heads,
        "feature_transformer_feedforward" : self.config.architecture.feature_transformer_feedforward,
        "feature_layer_norm_eps"          : self.config.architecture.feature_layer_norm_eps,
        "feature_num_hidden_layers"       : self.config.architecture.feature_num_hidden_layers,
      }
    self.bert_config = config.BertConfig.from_dict(
      self.bertAttrs,
      **self.featureAttrs,
      xla_device         = self.torch_tpu_available,
      reward_compilation = FLAGS.reward_compilation,
      is_sampling        = is_sampling,
    )
    return

  def Train(self, corpus, **kwargs) -> None:
    """
    Training point of active learning committee.
    """
    raise NotImplementedError
    return