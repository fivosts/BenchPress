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
RL Environment for the task of targeted benchmark generation.
"""
import pathlib
import os
import time
import typing

from deeplearning.benchpress.corpuses import tokenizers
from deeplearning.benchpress.corpuses import corpuses
from deeplearning.benchpress.samplers import samplers
from deeplearning.benchpress.models import backends
from deeplearning.benchpress.models import language_models
from deeplearning.benchpress.proto import reinforcement_learning_pb2
from deeplearning.benchpress.proto import internal_pb2
from deeplearning.benchpress.util import logging as l
from deeplearning.benchpress.util import distrib
from deeplearning.benchpress.util import commit
from deeplearning.benchpress.util import environment
from deeplearning.benchpress.util import pbutil
from deeplearning.benchpress.util import crypto
from deeplearning.benchpress.util import cache
from deeplearning.benchpress.reinforcement_learning import env
from deeplearning.benchpress.reinforcement_learning import agent
from deeplearning.benchpress.reinforcement_learning import memory
from deeplearning.benchpress.models.torch_bert import model as bert_model

from absl import flags

FLAGS = flags.FLAGS

from deeplearning.benchpress.util import cache

def AssertConfigIsValid(config: reinforcement_learning_pb2.RLModel) -> reinforcement_learning_pb2.RLModel:
  """
  Check validity of RL Model config.
  """
  ## Just check if language_model exists, later the language_models class will check the pbtxt.
  pbutil.AssertFieldIsSet(config, "language_model")
  ## Now check the specialized agent attributes.
  pbutil.AssertFieldIsSet(config, "target_features")
  pbutil.AssertFieldIsSet(config, "agent")
  ## Parse FeatureTokenizer fields.
  pbutil.AssertFieldIsSet(config.agent, "feature_tokenizer")
  pbutil.AssertFieldIsSet(config.agent, "batch_size")
  pbutil.AssertFieldIsSet(config.agent, "action_temperature_micros")
  pbutil.AssertFieldIsSet(config.agent, "token_temperature_micros")
  pbutil.AssertFieldIsSet(config.agent, "num_epochs")
  pbutil.AssertFieldIsSet(config.agent, "num_episodes")
  pbutil.AssertFieldIsSet(config.agent, "steps_per_episode")
  pbutil.AssertFieldIsSet(config.agent, "num_updates")
  pbutil.AssertFieldIsSet(config.agent, "gamma")
  pbutil.AssertFieldIsSet(config.agent, "lam")
  pbutil.AssertFieldIsSet(config.agent, "epsilon")
  pbutil.AssertFieldIsSet(config.agent, "learning_rate_micros")
  pbutil.AssertFieldIsSet(config.agent, "value_loss_coefficient")
  pbutil.AssertFieldIsSet(config.agent, "entropy_coefficient")
  pbutil.AssertFieldIsSet(config.agent.feature_tokenizer, "feature_max_value_token")
  pbutil.AssertFieldIsSet(config.agent.feature_tokenizer, "feature_singular_token_thr")
  pbutil.AssertFieldIsSet(config.agent.feature_tokenizer, "feature_token_range")
  pbutil.AssertFieldIsSet(config.agent.feature_tokenizer, "feature_sequence_length")
  return config

class RLModel(object):
  """
  Manager class of Reinforcement Learning pipeline for benchmark generation.
  """
  @property
  def tokenizer(self) -> tokenizers.TokenizerBase:
    return self.language_model.tokenizer

  @property
  def corpus(self) -> corpuses.Corpus:
    return self.language_model.corpus
  
  @property
  def pre_train_corpus(self) -> corpuses.Corpus:
    return self.language_model.pre_train_corpus

  @staticmethod
  def _ComputeHash(language_model: language_models.Model, config: reinforcement_learning_pb2.RLModel) -> str:
    """
    Compute unique hash of model specifications.
    """
    lm_hash = language_model.hash
    config_to_hash = reinforcement_learning_pb2.RLModel()
    config_to_hash.CopyFrom(config)
    config_to_hash.ClearField("language_model")
    return crypto.sha1_list([lm_hash, config_to_hash.SerializeToString()])

  def __init__(self, config: reinforcement_learning_pb2.RLModel):
    """
    A Reinforcement Learning model, wrapping a Language Model backend.
    """
    # Error early, so that a cache isn't created.
    if not isinstance(config, reinforcement_learning_pb2.RLModel):
      t = type(config).__name__
      raise TypeError(f"Config must be an RLModel proto. Received: '{t}'")

    self.config = reinforcement_learning_pb2.RLModel()
    self.config.CopyFrom(AssertConfigIsValid(config))

    # Initialize the LM-backend for token sampling.
    self.language_model = language_models.Model(self.config.language_model)
 
    self.hash = self._ComputeHash(self.language_model, self.config)
    self._created = False

    if environment.WORLD_RANK == 0:
      self.cache = cache.mkcache("rl_model", self.hash)
      self.cache.path.mkdir(exist_ok = True, parents = True)
    else:
      while not cache.cachepath("rl_model", self.hash).exists():
        time.sleep(0.5)
      self.cache = cache.mkcache("rl_model", self.hash)

    if environment.WORLD_RANK == 0:
      # Create the necessary cache directories.
      (self.cache.path / "feature_sampler").mkdir(exist_ok = True)
      (self.cache.path / "samples").mkdir(exist_ok = True)
      # Create symlink to language model.
      symlink = self.cache.path / "language_model"
      if not symlink.is_symlink():
        os.symlink(
          os.path.relpath(
            pathlib.Path(self.language_model.cache.path),
            self.cache.path
          ),
          symlink
        )
      # Setup META.pbtxt
      if self.cache.get("META.pbtxt"):
        cached_meta = pbutil.FromFile(
          pathlib.Path(self.cache["META.pbtxt"]), internal_pb2.RLModelMeta()
        )
        # Exclude num_epochs and corpus location from metadata comparison.
        config_to_compare = reinforcement_learning_pb2.RLModel()
        config_to_compare.CopyFrom(self.config)
        config_to_compare.language_model.corpus.ClearField("contentfiles")
        if config_to_compare.language_model.HasField("pre_train_corpus"):
          config_to_compare.language_model.pre_train_corpus.ClearField("contentfiles")
        config_to_compare.language_model.training.ClearField("num_epochs")
        config_to_compare.language_model.training.ClearField("num_train_steps")
        if config_to_compare.language_model.HasField("pre_train_corpus"):
          config_to_compare.language_model.training.ClearField("num_pretrain_steps")
        config_to_compare.language_model.training.ClearField("batch_size")
        if config_to_compare.language_model.training.HasField("data_generator"):
          config_to_compare.language_model.training.data_generator.ClearField("steps_per_epoch")
          config_to_compare.language_model.training.data_generator.ClearField("validation_set")
        # These fields should have already been cleared, but we'll do it again
        # so that metadata comparisons don't fail when the cached meta schema
        # is updated.
        cached_to_compare = reinforcement_learning_pb2.RLModel()
        cached_to_compare.CopyFrom(cached_meta.config)
        cached_to_compare.language_model.corpus.ClearField("contentfiles")
        if cached_to_compare.language_model.HasField("pre_train_corpus"):
          cached_to_compare.language_model.pre_train_corpus.ClearField("contentfiles")
        cached_to_compare.language_model.training.ClearField("num_epochs")
        cached_to_compare.language_model.training.ClearField("num_train_steps")
        if cached_to_compare.language_model.HasField("pre_train_corpus"):
          cached_to_compare.language_model.training.ClearField("num_pretrain_steps")
        cached_to_compare.language_model.training.ClearField("batch_size")
        if cached_to_compare.language_model.training.HasField("data_generator"):
          cached_to_compare.language_model.training.data_generator.ClearField("steps_per_epoch")
          cached_to_compare.language_model.training.data_generator.ClearField("validation_set")
        if cached_to_compare.language_model.training.sequence_length != config_to_compare.language_model.training.sequence_length:
          l.logger().warning("Mismatch between pre-trained and current config sequence_length!\
            This can only be intended in BERT model!")
        cached_to_compare.language_model.training.ClearField("sequence_length")
        config_to_compare.language_model.training.ClearField("sequence_length")
        if config_to_compare != cached_to_compare:
          raise SystemError("Metadata mismatch: {} \n\n {}".format(config_to_compare, cached_to_compare))
        self.meta = cached_meta
      else:
        self.meta = internal_pb2.RLModelMeta()
        self.meta.config.CopyFrom(self.config)
        self._WriteMetafile()

      ## Store current commit
      commit.saveCommit(self.cache.path)
    l.logger().info("Initialized RL Pipeline in {}".format(self.cache.path))

    """
    How do you target features during training ?
    1) Active learner - downstream task <- Sampler
    2) Random feasible vectors (collected from OpenCL corpus ?) <- Sampler ?
    3) Got from benchmark suites ? <- Sampler
    """
    return

  def Create(self, **kwargs) -> bool:
    """
    Create the LM and RL environment.
    """
    _ = self.language_model.Create()
    if self.language_model.pre_train_corpus:
      self.language_model.PreTrain(**kwargs)
    self.language_model.Train(**kwargs)

    self.feature_tokenizer = tokenizers.FeatureTokenizer.FromArgs(
      self.config.agent.feature_tokenizer.feature_singular_token_thr,
      self.config.agent.feature_tokenizer.feature_max_value_token,
      self.config.agent.feature_tokenizer.feature_token_range
    )
    if self._created:
      return False
    FLAGS.sample_indices_limit = 1 # Force BERT-LM on one prediction per hole.
    self._created = True
    self.env = env.Environment(
      self.config,
      self.language_model.backend.config.architecture.max_position_embeddings,
      self.language_model.corpus,
      self.tokenizer,
      self.feature_tokenizer,
      self.cache.path,
    )
    self.agent  = agent.Agent(
      self.config, self.language_model, self.tokenizer, self.feature_tokenizer, self.cache.path
    )
    self.memory = memory.Memory(self.cache.path)
    return True

  def PreTrain(self, **kwargs) -> 'RLModel':
    """
    Pre-train wrapper for Language model.
    No-pretraining is supported for RL model.
    """
    self.Create(**kwargs)
    return self

  def Train(self, **kwargs) -> None:
    """
    Train the RL-Agent.
    """
    self.Create(**kwargs)
    ## First, train the Language model backend.

    num_epochs        = self.config.agent.num_epochs
    num_episodes      = self.config.agent.num_episodes
    steps_per_episode = self.config.agent.steps_per_episode
    num_updates       = self.config.agent.num_updates
    gamma             = self.config.agent.gamma
    lam               = self.config.agent.lam
    epsilon           = self.config.agent.epsilon
    lr                = self.config.agent.learning_rate_micros / 10e6
    value_loss_coeff  = self.config.agent.value_loss_coefficient
    entropy_coeff     = self.config.agent.entropy_coefficient

    self.agent.Train(
      env               = self.env,
      num_epochs        = num_epochs,
      num_episodes      = num_episodes,
      steps_per_episode = steps_per_episode,
      num_updates       = num_updates,
      gamma             = gamma,
      lr                = lr,
      lam               = lam,
      epsilon           = epsilon,
      value_loss_coeff  = value_loss_coeff,
      entropy_coeff     = entropy_coeff,
    )
    return

  def Sample(self, sampler: samplers.Sampler) -> None:
    """
    Instead of calling Model's sample, this sample will be called, acting as a backend (BERT) wrapper.
    """
    raise NotImplementedError("Here you must sample your RL-Model.")
    return

  def SamplerCache(self, sampler: samplers.Sampler) -> pathlib.Path:
    """Get the path to a sampler cache.

    Args:
      sampler: A Sampler instance.

    Returns:
      A path to a directory. Note that this directory may not exist - it is
      created only after a call to Sample().
    """
    return self.cache.path / "samples" / sampler.hash

  def _WriteMetafile(self) -> None:
    pbutil.ToFile(self.meta, pathlib.Path(self.cache.keypath("META.pbtxt")))

  def saveCheckpoint(self) -> None:
    """
    Save current state of RL pipeline.
    """
    self.feature_loader.saveCheckpoint()
    self.env.saveCheckpoint()
    self.agent.saveCheckpoint()
    self.memory.saveCheckpoint()
    return
  
  def loadCheckpoint(self) -> None:
    """
    Load RL pipeline checkpoint.
    """
    self.feature_loader.loadCheckpoint()
    self.env.loadCheckpoint()
    self.agent.loadCheckpoint()
    self.memory.loadCheckpoint()
    return
