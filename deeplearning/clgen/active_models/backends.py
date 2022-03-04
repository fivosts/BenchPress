"""Neural network backends for active learning models."""
import typing
import numpy as np

from deeplearning.clgen.active_models import downstream_tasks
from deeplearning.clgen.proto import active_learning_pb2
from deeplearning.clgen.util import cache

class BackendBase(object):
  """
  The base class for an active learning model backend.
  """
  def __init__(
    self,
    config          : active_learning_pb2.ActiveLearner,
    fs_cache        : cache.FSCache,
    downstream_task : downstream_tasks.DownstreamTask
  ):
    self.config          = config
    self.cache           = fs_cache
    self.downstream_task = downstream_task
    return

  def Train(self, corpus: "Corpus", **extra_kwargs) -> None:
    """Train the backend."""
    raise NotImplementedError

  def Sample(self, sampler: 'samplers.Sampler', seed: typing.Optional[int] = None) -> None:
    """
    Sampling regime for backend.
    """
    raise NotImplementedError
