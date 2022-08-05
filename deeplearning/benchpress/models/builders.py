# coding=utf-8
# Copyright 2022 Foivos Tsimpourlas and Chris Cummins.
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
"""This file builds Keras models from BenchPress Model config protos."""

from deeplearning.benchpress.proto import model_pb2
from absl import flags
from deeplearning.benchpress.util import pbutil
from deeplearning.benchpress.models import lm_data_generator

FLAGS = flags.FLAGS


def AssertIsBuildable(config: model_pb2.Model) -> model_pb2.Model:
  """Assert that a model configuration is buildable.

  Args:
    config: A model proto.

  Returns:
    The input model proto, unmodified.

  Raises:
    UserError: If the model is not buildable.
    InternalError: If the value of the training.optimizer field is not
      understood.
  """
  # Any change to the Model proto schema will require a change to this function.
  try:
    pbutil.AssertFieldIsSet(config, "corpus")
    pbutil.AssertFieldIsSet(config, "architecture")
    pbutil.AssertFieldIsSet(config, "training")
    pbutil.AssertFieldIsSet(config.architecture, "backend")
    if config.architecture.backend == model_pb2.NetworkArchitecture.KERAS_SEQ:
      pbutil.AssertFieldIsSet(config.architecture, "neuron_type")
      pbutil.AssertFieldConstraint(
        config.architecture,
        "embedding_size",
        lambda x: 0 < x,
        "NetworkArchitecture.embedding_size must be > 0",
      )
    elif config.architecture.backend == model_pb2.NetworkArchitecture.TENSORFLOW_SEQ:
      pbutil.AssertFieldIsSet(config.architecture, "neuron_type")
      pbutil.AssertFieldConstraint(
        config.architecture,
        "neurons_per_layer",
        lambda x: 0 < x,
        "NetworkArchitecture.neurons_per_layer must be > 0",
      )
      pbutil.AssertFieldConstraint(
        config.architecture,
        "num_layers",
        lambda x: 0 < x,
        "NetworkArchitecture.num_layers must be > 0",
      )
      pbutil.AssertFieldConstraint(
        config.architecture,
        "post_layer_dropout_micros",
        lambda x: 0 <= x <= 1000000,
        "NetworkArchitecture.post_layer_dropout_micros "
        "must be >= 0 and <= 1000000",
      )
      pbutil.AssertFieldConstraint(
        config.training,
        "num_epochs",
        lambda x: 0 < x,
        "TrainingOptions.num_epochs must be > 0",
      )
    elif config.architecture.backend == model_pb2.NetworkArchitecture.TENSORFLOW_BERT\
      or config.architecture.backend == model_pb2.NetworkArchitecture.TORCH_BERT\
      or config.architecture.backend == model_pb2.NetworkArchitecture.INCODER_1B\
      or config.architecture.backend == model_pb2.NetworkArchitecture.INCODER_6B:
      # Data generator is needed when using bert.
      pbutil.AssertFieldIsSet(config.training, "data_generator")
      # Parse data_generator params.
      _ = lm_data_generator.AssertConfigIsValid(config.training.data_generator)
      if config.architecture.backend != model_pb2.NetworkArchitecture.INCODER_1B and config.architecture.backend != model_pb2.NetworkArchitecture.INCODER_6B:
        ## .architecture params
        pbutil.AssertFieldIsSet(
          config.architecture,
          "hidden_size",
        )
        pbutil.AssertFieldIsSet(
          config.architecture,
          "num_hidden_layers",
        )
        pbutil.AssertFieldIsSet(
          config.architecture,
          "num_attention_heads",
        )
        pbutil.AssertFieldIsSet(
          config.architecture,
          "intermediate_size",
        )
        pbutil.AssertFieldConstraint(
          config.architecture,
          "hidden_size",
          lambda x: x % config.architecture.num_attention_heads == 0,
          "The hidden size is not a multiple of the number of attention "
          "heads."
        )
        pbutil.AssertFieldIsSet(
          config.architecture,
          "hidden_act",
        )
        pbutil.AssertFieldIsSet(
          config.architecture,
          "hidden_dropout_prob",
        )
        pbutil.AssertFieldIsSet(
          config.architecture,
          "attention_probs_dropout_prob",
        )
        pbutil.AssertFieldIsSet(
          config.architecture,
          "type_vocab_size",
        )
        pbutil.AssertFieldIsSet(
          config.architecture,
          "initializer_range",
        )
        pbutil.AssertFieldIsSet(
          config.architecture,
          "layer_norm_eps",
        )
        ## Optional feature encoder attributes
        if config.architecture.HasField("feature_encoder") and config.architecture.feature_encoder == True:
          pbutil.AssertFieldIsSet(
            config.architecture,
            "feature_sequence_length"
          )
          pbutil.AssertFieldIsSet(
            config.architecture,
            "feature_embedding_size"
          )
          pbutil.AssertFieldIsSet(
            config.architecture,
            "feature_dropout_prob"
          )
          pbutil.AssertFieldIsSet(
            config.architecture,
            "feature_singular_token_thr"
          )
          pbutil.AssertFieldIsSet(
            config.architecture,
            "feature_max_value_token"
          )
          pbutil.AssertFieldIsSet(
            config.architecture,
            "feature_token_range"
          )
          pbutil.AssertFieldIsSet(
            config.architecture,
            "feature_num_attention_heads"
          )
          pbutil.AssertFieldIsSet(
            config.architecture,
            "feature_transformer_feedforward"
          )
          pbutil.AssertFieldIsSet(
            config.architecture,
            "feature_layer_norm_eps"
          )
          pbutil.AssertFieldIsSet(
            config.architecture,
            "feature_num_hidden_layers"
          )
      ## .training params
      pbutil.AssertFieldIsSet(
        config.training,
        "max_predictions_per_seq",
      )
      pbutil.AssertFieldIsSet(
        config.training,
        "num_train_steps",
      )
      pbutil.AssertFieldIsSet(
        config.training,
        "num_warmup_steps",
      )
      if config.HasField("pre_train_corpus"):
        pbutil.AssertFieldIsSet(
          config.training,
          "num_pretrain_steps",
        )
        pbutil.AssertFieldIsSet(
          config.training,
          "num_prewarmup_steps",
        )
      pbutil.AssertFieldIsSet(
        config.training,
        "dupe_factor",
      )
      pbutil.AssertFieldIsSet(
        config.training,
        "masked_lm_prob",
      )
      pbutil.AssertFieldConstraint(
        config.training,
        "random_seed",
        lambda x: 0 <= x,
        "TrainingOptions.random_seed must be >= 0",
      )

    pbutil.AssertFieldConstraint(
      config.training,
      "sequence_length",
      lambda x: 1 <= x,
      "TrainingOptions.sequence_length must be >= 1",
    )
    pbutil.AssertFieldIsSet(
      config.training, "shuffle_corpus_contentfiles_between_epochs"
    )
    pbutil.AssertFieldConstraint(
      config.training,
      "batch_size",
      lambda x: 0 < x,
      "TrainingOptions.batch_size must be > 0",
    )
    pbutil.AssertFieldIsSet(config.training, "optimizer")
    if config.training.HasField("adam_optimizer"):
      pbutil.AssertFieldConstraint(
        config.training.adam_optimizer,
        "initial_learning_rate_micros",
        lambda x: 0 <= x,
        "AdamOptimizer.initial_learning_rate_micros must be >= 0",
      )
      if config.architecture.backend == model_pb2.NetworkArchitecture.KERAS_SEQ or \
         config.architecture.backend == model_pb2.NetworkArchitecture.TENSORFLOW_SEQ:
        pbutil.AssertFieldConstraint(
          config.training.adam_optimizer,
          "learning_rate_decay_per_epoch_micros",
          lambda x: 0 <= x,
          "AdamOptimizer.learning_rate_decay_per_epoch_micros must be >= 0",
        )
        pbutil.AssertFieldConstraint(
          config.training.adam_optimizer,
          "beta_1_micros",
          lambda x: 0 <= x <= 1000000,
          "AdamOptimizer.beta_1_micros must be >= 0 and <= 1000000",
        )
        pbutil.AssertFieldConstraint(
          config.training.adam_optimizer,
          "beta_2_micros",
          lambda x: 0 <= x <= 1000000,
          "AdamOptimizer.beta_2_micros must be >= 0 and <= 1000000",
        )
        pbutil.AssertFieldConstraint(
          config.training.adam_optimizer,
          "normalized_gradient_clip_micros",
          lambda x: 0 <= x,
          "AdamOptimizer.normalized_gradient_clip_micros must be >= 0",
        )
    elif config.training.HasField("rmsprop_optimizer"):
      pbutil.AssertFieldConstraint(
        config.training.rmsprop_optimizer,
        "initial_learning_rate_micros",
        lambda x: 0 <= x,
        "RmsPropOptimizer.initial_learning_rate_micros must be >= 0",
      )
      pbutil.AssertFieldConstraint(
        config.training.rmsprop_optimizer,
        "learning_rate_decay_per_epoch_micros",
        lambda x: 0 <= x,
        "RmsPropOptimizer.learning_rate_decay_per_epoch_micros must be >= 0",
      )
    else:
      raise SystemError(
        "Unrecognized value: 'TrainingOptions.optimizer'"
      )
  except Exception as e:
    raise e
  return config


def BuildOptimizer(config: model_pb2.Model) -> "keras.optimizers.Optimizer":
  """Construct the training optimizer from config.

  Args:
    config: A Model config proto.

  Raises:
    InternalError: If the value of the optimizer field is not understood.
  """
  # Deferred importing of Keras so that we don't have to activate the
  # TensorFlow backend every time we import this module.
  import keras

  # We do not use *any* default values for arguments, in case for whatever
  # reason the Keras API changes a default arg.
  if config.training.HasField("adam_optimizer"):
    opts = {}
    opt = config.training.adam_optimizer
    if opt.normalized_gradient_clip_micros:
      opts["clipnorm"] = opt.normalized_gradient_clip_micros / 1e6
    return keras.optimizers.Adam(
      lr=opt.initial_learning_rate_micros / 1e6,
      beta_1=opt.beta_1_micros / 1e6,
      beta_2=opt.beta_2_micros / 1e6,
      epsilon=None,
      decay=opt.learning_rate_decay_per_epoch_micros / 1e6,
      amsgrad=False,
      **opts,
    )
  elif config.training.HasField("rmsprop_optimizer"):
    opt = config.training.rmsprop_optimizer
    return keras.optimizers.RMSprop(
      lr=opt.initial_learning_rate_micros / 1e6,
      decay=opt.initial_learning_rate_micros / 1e6,
      rho=0.9,
      epsilon=None,
    )
  else:
    raise SystemError(
      "Unrecognized value: 'TrainingOptions.optimizer'"
    )


def BuildKerasModel(
  config: model_pb2.Model, vocabulary_size: int
) -> "keras.models.Sequential":
  """Build a Keras model from a Model proto.

  Args:
    config: A Model proto instance.
    vocabulary_size: The number of tokens in the vocabulary.

  Returns:
    A Sequential model instance.
  """
  # Deferred importing of Keras so that we don't have to activate the
  # TensorFlow backend every time we import this module.
  import keras

  dropout = (config.architecture.post_layer_dropout_micros or 0) / 1e6
  model = keras.models.Sequential()
  layer = {
    model_pb2.NetworkArchitecture.LSTM: keras.layers.LSTM,
    model_pb2.NetworkArchitecture.RNN: keras.layers.RNN,
    model_pb2.NetworkArchitecture.GRU: keras.layers.GRU,
  }[config.architecture.neuron_type]

  # The input layer.
  model.add(
    keras.layers.Embedding(
      vocabulary_size,
      config.architecture.embedding_size,
      batch_input_shape=(
        config.training.batch_size,
        config.training.sequence_length,
      ),
    )
  )
  model.add(keras.layers.Dropout(dropout))
  # The recurrent network layers.
  for _ in range(config.architecture.num_layers):
    model.add(
      layer(
        config.architecture.neurons_per_layer,
        return_sequences=True,
        stateful=True,
      )
    )
    model.add(keras.layers.Dropout(dropout))
  # The output layer.
  model.add(
    keras.layers.TimeDistributed(
      keras.layers.Dense(vocabulary_size, activation="softmax")
    )
  )
  return model
