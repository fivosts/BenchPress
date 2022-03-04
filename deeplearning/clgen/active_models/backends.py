"""Neural network backends for active learning models."""
import typing
import numpy as np

from deeplearning.clgen.util import cache

class BackendBase(object):
  """
  The base class for an active learning model backend.
  """
  def __init__(
    self,
    config: active_learning_pb2.ActiveLearner,
    fs_cache: cache.FSCache,
    hash: str,
  ):
    self.config = config
    self.cache = fs_cache
    self.hash = hash

  def Train(self, corpus: "Corpus", **extra_kwargs) -> None:
    """Train the backend."""
    raise NotImplementedError

  def Sample(self, sampler: 'samplers.Sampler', seed: typing.Optional[int] = None) -> None:
    """
    Sampling regime for backend.
    """
    raise NotImplementedError
