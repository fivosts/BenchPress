"""
RL Environment for the task of targeted benchmark generation.
"""
import pathlib
import os
import time
import typing

from deeplearning.clgen.corpuses import tokenizers
from deeplearning.clgen.corpuses import corpuses
from deeplearning.clgen.samplers import samplers
from deeplearning.clgen.models import backends
from deeplearning.clgen.models import language_models
from deeplearning.clgen.proto import reinforcement_learning_pb2
from deeplearning.clgen.proto import internal_pb2
from deeplearning.clgen.util import logging as l
from deeplearning.clgen.util import distrib
from deeplearning.clgen.util import commit
from deeplearning.clgen.util import environment
from deeplearning.clgen.util import pbutil
from deeplearning.clgen.util import crypto
from deeplearning.clgen.util import cache
from deeplearning.clgen.reinforcement_learning import env
from deeplearning.clgen.reinforcement_learning import agent
from deeplearning.clgen.reinforcement_learning import memory
from deeplearning.clgen.models.torch_bert import model as bert_model

from absl import flags

from deeplearning.clgen.util import cache

def AssertConfigIsValid(config: reinforcement_learning_pb2.RLModel) -> reinforcement_learning_pb2.RLModel:
  """
  Check validity of RL Model config.
  """
  ## Just check if language_model exists, later the language_models class will check the pbtxt.
  pbutil.AssertFieldIsSet(config, "language_model")
  ## Now check the specialized agent attributes.
  pbutil.AssertFieldIsSet(config, "target_features")
  pbutil.AssertFieldIsSet(config, "agent")
  pbutil.AssertFieldIsSet(config.agent, "action_qv")
  pbutil.AssertFieldIsSet(config.agent, "action_lm")
  ## Parse FeatureTokenizer fields.
  pbutil.AssertFieldIsSet(config.agent, "feature_tokenizer")
  pbutil.AssertFieldIsSet(config.agent.feature_tokenizer, "feature_max_value_token")
  pbutil.AssertFieldIsSet(config.agent.feature_tokenizer, "feature_singular_token_thr")
  pbutil.AssertFieldIsSet(config.agent.feature_tokenizer, "feature_token_range")
  pbutil.AssertFieldIsSet(config.agent.feature_tokenizer, "feature_sequence_length")
  ## Parse ActionQV fields.
  pbutil.AssertFieldIsSet(config.agent.action_qv, "hidden_size")
  pbutil.AssertFieldConstraint(
    config.agent.action_qv,
    "num_attention_heads",
    lambda x: x > 0,
    "#Attention heads must be at least 1."
  )
  pbutil.AssertFieldIsSet(config.agent.action_qv, "intermediate_size")
  pbutil.AssertFieldConstraint(
    config.agent.action_qv,
    "num_hidden_layers",
    lambda x: x > 0,
    "#hidden layers must be at least 1."
  )
  pbutil.AssertFieldIsSet(config.agent.action_qv, "layer_norm_eps")
  pbutil.AssertFieldConstraint(
    config.agent.action_qv,
    "hidden_dropout_prob",
    lambda x: 0.0 < x < 1.0,
    "Dropout probability must be between 0.0 and 1.0."
  )
  pbutil.AssertFieldConstraint(
    config.agent.action_qv,
    "attention_probs_dropout_prob",
    lambda x: 0.0 < x < 1.0,
    "Attention Dropout probability must be between 0.0 and 1.0."
  )
  pbutil.AssertFieldIsSet(config.agent.action_qv, 'type_vocab_size')
  pbutil.AssertFieldConstraint(
    config.agent.initializer_range,
    "initializer_range",
    lambda x: x >= 0.0,
    "Initializer range must be non-negative float."
  )
  pbutil.AssertFieldConstraint(
    config.agent.action_qv,
    "hidden_act",
    lambda x: x in set(bert_model.ACT2FN.keys()),
    "Invalid choice for hidden_act"
  )
  pbutil.AssertFieldIsSet(config.agent.action_lm, "hidden_size")
  pbutil.AssertFieldConstraint(
    config.agent.action_lm,
    "num_attention_heads",
    lambda x: x > 0,
    "#Attention heads must be at least 1."
  )
  pbutil.AssertFieldIsSet(config.agent.action_lm, "intermediate_size")
  pbutil.AssertFieldConstraint(
    config.agent.action_lm,
    "num_hidden_layers",
    lambda x: x > 0,
    "#hidden layers must be at least 1."
  )
  pbutil.AssertFieldIsSet(config.agent.action_lm, "layer_norm_eps")
  pbutil.AssertFieldConstraint(
    config.agent.action_lm,
    "hidden_dropout_prob",
    lambda x: 0.0 < x < 1.0,
    "Dropout probability must be between 0.0 and 1.0."
  )
  pbutil.AssertFieldConstraint(
    config.agent.action_lm,
    "attention_probs_dropout_prob",
    lambda x: 0.0 < x < 1.0,
    "Attention Dropout probability must be between 0.0 and 1.0."
  )
  pbutil.AssertFieldIsSet(config.agent.action_lm, 'type_vocab_size')
  pbutil.AssertFieldConstraint(
    config.agent.initializer_range,
    "initializer_range",
    lambda x: x >= 0.0,
    "Initializer range must be non-negative float."
  )
  pbutil.AssertFieldConstraint(
    config.agent.action_lm,
    "hidden_act",
    lambda x: x in set(bert_model.ACT2FN.keys()),
    "Invalid choice for hidden_act"
  )
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
      (self.cache.path / "checkpoints").mkdir(exist_ok = True)
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
    self._created = True
    self.env    = env.Environment(
      self.config, self.language_model.corpus, self.tokenizer, self.feature_tokenizer, self.cache.path
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

    num_episodes = 10
    ## Start the RL training pipeline.
    for ep in range(num_episodes):
      self.env.reset()
      is_term = False
      while not is_term:
        state  = self.env.get_state()           # Get current state.
        action = self.agent.make_action(state)  # Predict the action given the state.
        reward = self.env.step(action)          # Step the action into the environment and face the consequences.
        self.memory.add(state, action, reward)  # Add to replay buffer the episode.
      self.agent.update(self.memory.sample()) # Train the agent on a pool of memories.
      self.saveCheckpoint()
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
