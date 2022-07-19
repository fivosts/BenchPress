"""Modeling configuration for Deep_Q Network"""
from deeplearning.clgen.corpuses import tokenizers
from deeplearning.clgen.models import language_models
from deeplearning.clgen.proto import reinforcement_learning_pb2

class QValuesConfig(object):
  @classmethod
  def from_config(cls,
                  config                  : reinforcement_learning_pb2.RLModel,
                  max_position_embeddings : int,
                  tokenizer               : tokenizers.TokenizerBase,
                  feature_tokenizer       : tokenizers.FeatureTokenizer,
                  language_model          : language_models.Model,
                  ) -> 'QValuesConfig':
    dict = {
      'vocab_size'                   : tokenizer.vocab_size,
      'feature_vocab_size'           : feature_tokenizer.vocab_size,
      'feature_pad_idx'              : feature_tokenizer.padToken,
      'pad_token_id'                 : tokenizer.padToken,
      'max_position_embeddings'      : max_position_embeddings,
      'feature_sequence_length'      : config.agent.feature_tokenizer.feature_sequence_length,
      'hidden_dropout_prob'          : language_model.backend.bert_config.hidden_dropout_prob,
      'feature_dropout_prob'         : language_model.backend.bert_config.hidden_dropout_prob,
      'hidden_size'                  : language_model.backend.bert_config.hidden_size,
      'feature_embedding_size'       : language_model.backend.bert_config.hidden_size,
      'num_attention_heads'          : language_model.backend.bert_config.num_attention_heads,
      'intermediate_size'            : language_model.backend.bert_config.intermediate_size,
      'num_hidden_layers'            : language_model.backend.bert_config.num_hidden_layers,
      'layer_norm_eps'               : language_model.backend.bert_config.layer_norm_eps,
      'hidden_act'                   : language_model.backend.bert_config.hidden_act,
      'attention_probs_dropout_prob' : language_model.backend.bert_config.attention_probs_dropout_prob,
      'type_vocab_size'              : language_model.backend.bert_config.type_vocab_size,
      'initializer_range'            : language_model.backend.bert_config.initializer_range,
      'temperature'                  : config.agent.action_temperature_micros / 10e6,
      'token_temperature'            : config.agent.token_temperature_micros / 10e6,
      'feature_encoder'              : False,
      'batch_size'                   : config.agent.batch_size
    }
    return QValuesConfig(**dict)

  def __init__(self, **attrs):
    self.__dict__.update(attrs)
    return
