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
"""Neural network backends for active learning models."""
import typing
import pathlib
import numpy as np

from deeplearning.benchpress.active_models import downstream_tasks
from deeplearning.benchpress.proto import active_learning_pb2
from deeplearning.benchpress.util import cache

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
