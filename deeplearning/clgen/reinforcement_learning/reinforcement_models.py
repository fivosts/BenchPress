"""
RL Environment for the task of targeted benchmark generation.
"""
import pathlib

from deeplearning.clgen.models import backends
from deeplearning.clgen.proto import reinforcement_learning_pb2
from deeplearning.clgen.util import logging as l
from deeplearning.clgen.util import distrib
from deeplearning.clgen.util import commit
from deeplearning.clgen.util import environment

from absl import flags

def AssertConfigIsValid(config: reinforcement_learning_pb2.RLModel) -> reinforcement_learning_pb2.RLModel:
  """
  Check validity of RL Model config.
  """
  raise NotImplementedError("TODO")
  return config

class Models(object):
  """
  Manager class of Reinforcement Learning pipeline
  for benchmark generation
  """
  def __init__(self, config: reinforcement_learning_pb2.RLModel, cache_path: pathlib.Path):
    """
    Initialize RL manager.
    """
    # Error early, so that a cache isn't created.
    if not isinstance(config, reinforcement_learning_pb2.RLModel):
      t = type(config).__name__
      raise TypeError(f"Config must be an RLModel proto. Received: '{t}'")

    self.config = AssertConfigIsValid(config)
    self.cache_path = cache_path / "reinforcement_learning"

    if environment.WORLD_RANK == 0:
      self.cache_path.mkdir(exist_ok = True, parents = True)
    distrib.barrier()

    if environment.WORLD_RANK == 0:
      ## Store current commit
      commit.saveCommit(self.cache_path)
    return

  def Train(self) -> None:
    """
    Train the RL-Agent.
    """
    raise NotImplementedError("This should happen after LM pretraining-fine-tuning, and needs to be bound to the target features")
    return
  
  def Sample(self, backend: backends.BackendBase) -> None:
    """
    Instead of callid Model's sample, this sample will be called, acting as a backend (BERT) wrapper.
    """
    return
