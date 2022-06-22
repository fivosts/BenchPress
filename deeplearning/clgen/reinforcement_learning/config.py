"""Modeling configuration for Deep_Q Network"""
from deeplearning.clgen.corpuses import tokenizers
from deeplearning.clgen.proto import reinforcement_learning_pb2

class QValuesConfig(object):
  @classmethod
  def from_config(cls,
                  config            : reinforcement_learning_pb2.RLModel,
                  tokenizer         : tokenizers.TokenizerBase,
                  feature_tokenizer : tokenizers.FeatureTokenizer,
                  ) -> 'QValuesConfig':
    dict = {
      'vocab_size'              : tokenizer.vocab_size,
      'feature_vocab_size'      : feature_tokenizer.vocab_size,
      'feature_pad_idx'         : feature_tokenizer.padToken,
      'pad_idx'                 : tokenizer.pad_idx,
      'hidden_dropout_prob'     : config.deep_qv.hidden_dropout_prob,
      'hidden_size'             : config.deep_qv.hidden_size,
      'num_attention_heads'     : config.deep_qv.num_attention_heads,
      'intermediate_size'       : config.deep_qv.intermediate_size,
      'num_hidden_layers'       : config.deep_qv.num_hidden_layers,
      'layer_norm_eps'          : config.deep_qv.layer_norm_eps,
      'max_position_embeddings' : config.deep_qv.max_position_embeddings,

    }
    return QValuesConfig()

  def __init__(self, config):
    return