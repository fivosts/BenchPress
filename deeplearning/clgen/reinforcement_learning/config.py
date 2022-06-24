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
      'pad_idx'                 : tokenizer.padToken,
      'hidden_dropout_prob'     : config.agent.action_qv.hidden_dropout_prob,
      'feature_dropout_prob'    : config.agent.action_qv.hidden_dropout_prob,
      'hidden_size'             : config.agent.action_qv.hidden_size,
      'feature_sequence_length' : config.agent.action_qv.feature_sequence_length,
      'feature_embedding_size'  : config.agent.action_qv.hidden_size,
      'num_attention_heads'     : config.agent.action_qv.num_attention_heads,
      'intermediate_size'       : config.agent.action_qv.intermediate_size,
      'num_hidden_layers'       : config.agent.action_qv.num_hidden_layers,
      'layer_norm_eps'          : config.agent.action_qv.layer_norm_eps,
      'max_position_embeddings' : config.agent.action_qv.max_position_embeddings,

    }
    return QValuesConfig(**dict)

  def __init__(self, **attrs):
    self.__dict__.update(attrs)
    return
