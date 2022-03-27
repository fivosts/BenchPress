"""Neural network backends for active learning models."""
import typing
import pathlib
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
    cache_path      : pathlib.Path,
    downstream_task : downstream_tasks.DownstreamTask
  ):
    self.config          = config
    self.cache_path      = cache_path
    self.downstream_task = downstream_task
    self.downstream_task.setup_dataset(num_train_steps = self.config.num_train_steps)
    return

  def Train(self, **extra_kwargs) -> None:
    """Train the backend."""
    raise NotImplementedError

  def Sample(self, sampler: 'samplers.Sampler', seed: typing.Optional[int] = None) -> None:
    """
    Sampling regime for backend.
    """
    raise NotImplementedError
