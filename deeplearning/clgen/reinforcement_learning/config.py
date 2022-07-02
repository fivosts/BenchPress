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
      'vocab_size'                     : tokenizer.vocab_size,
      'feature_vocab_size'             : feature_tokenizer.vocab_size,
      'feature_pad_idx'                : feature_tokenizer.padToken,
      'pad_token_id'                   : tokenizer.padToken,
      'feature_sequence_length'        : config.feature_tokenizer.feature_sequence_length,
      'action_hidden_dropout_prob'     : config.agent.action_qv.hidden_dropout_prob,
      'action_feature_dropout_prob'    : config.agent.action_qv.hidden_dropout_prob,
      'action_hidden_size'             : config.agent.action_qv.hidden_size,
      'action_feature_embedding_size'  : config.agent.action_qv.hidden_size,
      'action_num_attention_heads'     : config.agent.action_qv.num_attention_heads,
      'action_intermediate_size'       : config.agent.action_qv.intermediate_size,
      'action_num_hidden_layers'       : config.agent.action_qv.num_hidden_layers,
      'action_layer_norm_eps'          : config.agent.action_qv.layer_norm_eps,
      'action_hidden_act'              : config.agent.action_qv.hidden_act,
      'action_attention_dropout_prob'  : config.agent.action_qv.attention_dropout_prob,
      'action_type_vocab_size'         : config.agent.action_qv.type_vocab_size,
      'action_initializer_range'       : config.agent.action_qv.initializer_range,
      'token_hidden_dropout_prob'      : config.agent.action_lm.hidden_dropout_prob,
      'token_feature_dropout_prob'     : config.agent.action_lm.hidden_dropout_prob,
      'token_hidden_size'              : config.agent.action_lm.hidden_size,
      'token_feature_embedding_size'   : config.agent.action_lm.hidden_size,
      'token_num_attention_heads'      : config.agent.action_lm.num_attention_heads,
      'token_intermediate_size'        : config.agent.action_lm.intermediate_size,
      'token_num_hidden_layers'        : config.agent.action_lm.num_hidden_layers,
      'token_layer_norm_eps'           : config.agent.action_lm.layer_norm_eps,
      'token_hidden_act'               : config.agent.action_lm.hidden_act,
      'token_attention_dropout_prob'   : config.agent.action_lm.attention_dropout_prob,
      'token_type_vocab_size'          : config.agent.action_lm.type_vocab_size,
      'token_initializer_range'        : config.agent.action_lm.initializer_range,
      'temperature'                    : config.agent.action_lm.temperature_micros / 1e6,
      'feature_encoder'                : False,
    }
    return QValuesConfig(**dict)

  def __init__(self, **attrs):
    self.__dict__.update(attrs)
    return
