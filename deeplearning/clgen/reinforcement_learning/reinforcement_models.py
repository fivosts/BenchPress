"""
RL Environment for the task of targeted benchmark generation.
"""
import pathlib

from deeplearning.clgen.corpuses import tokenizers
from deeplearning.clgen.models import backends
from deeplearning.clgen.models import language_models
from deeplearning.clgen.proto import reinforcement_learning_pb2
from deeplearning.clgen.proto import internal_pb2
from deeplearning.clgen.util import logging as l
from deeplearning.clgen.util import distrib
from deeplearning.clgen.util import commit
from deeplearning.clgen.util import environment
from deeplearning.clgen.util import pbutil

from absl import flags

from deeplearning.clgen.util.cache import mkcache

def AssertConfigIsValid(config: reinforcement_learning_pb2.RLModel) -> reinforcement_learning_pb2.RLModel:
  """
  Check validity of RL Model config.
  """
  ## Just check if language_model exists, later the language_models class will check the pbtxt.
  pbutil.AssertFieldIsSet(config, "language_model")

  if not (pbutil.HasField("train_set") or pbutil.HasField("random")):
    raise ValueError("You haven't specified the target features dataloader for RL-training:\n{}".format(str(config)))
  return config

class RLModel(object):
  """
  Manager class of Reinforcement Learning pipeline for benchmark generation.
  """
  @property
  def tokenizer(self) -> tokenizers.TokenizerBase:
    return self.language_model.tokenizer

  def __init__(self, config: reinforcement_learning_pb2.RLModel, cache_path: pathlib.Path):
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

    distrib.lock()
    self.cache = cache.mkcache("rl_model", self.hash)
    distrib.unlock()

    if environment.WORLD_RANK == 0:
      # Create the necessary cache directories.
      (self.cache.path / "checkpoints").mkdir(exist_ok = True)
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
        if cached_to_compare.language_model.training.sequence_length != config_to_compare.training.sequence_length:
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

    raise NotImplementedError
    corpus, pre_train_corpus, SamplerCache(), hash

    if environment.WORLD_RANK == 0:
      ## Store current commit
      commit.saveCommit(self.cache_path)

    """
    How do you target features during training ?
    1) Active learner - downstream task <- Sampler
    2) Random feasible vectors (collected from OpenCL corpus ?) <- Sampler ?
    3) Got from benchmark suites ? <- Sampler
    """

    self.env = env.Environment()
    self.agent = agent.Agent()
    return

  def Create(self):
    raise NotImplementedError

  def PreTrain(self):
    raise NotImplementedError

  def Train(self) -> None:
    """
    Train the RL-Agent.
    """
    for ep in range(num_episodes):
      self.env.reset()
      target_features = self.feature_sampler.sample()
      self.env.init_state(target_features)
      is_term = False
      while not is_term:
        state  = self.env.get_state()
        action = self.agent.make_action(state)
        reward = self.env.step(action)
        self.memory.add(state, action, reward)
        self.agent.update(self.memory.sample())
    return
  
  def Sample(self, backend: backends.BackendBase) -> None:
    """
    Instead of calling Model's sample, this sample will be called, acting as a backend (BERT) wrapper.
    """
    return

  def _WriteMetafile(self) -> None:
    pbutil.ToFile(self.meta, pathlib.Path(self.cache.keypath("META.pbtxt")))
