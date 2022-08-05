# Copyright (c) 2016-2020 Chris Cummins.
#
# clgen is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# clgen is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with clgen.  If not, see <https://www.gnu.org/licenses/>.
"""The CLgen language model."""
import os
import time
import socket
import shutil
import getpass
import pathlib
import typing
import datetime
import humanize

import numpy as np

from deeplearning.benchpress.util import pbutil
from deeplearning.benchpress.util import cache
from deeplearning.benchpress.util import crypto
from deeplearning.benchpress.util import commit
from deeplearning.benchpress.util import sqlutil
from deeplearning.benchpress.util import environment
from deeplearning.benchpress.util import distrib
from deeplearning.benchpress.active_models import downstream_tasks
from deeplearning.benchpress.active_models.committee import active_committee
from deeplearning.benchpress.active_models.committee import config as com_config
from deeplearning.benchpress.samplers import sample_observers as sample_observers_lib
from deeplearning.benchpress.proto import active_learning_pb2
from absl import flags

from deeplearning.benchpress.util import logging as l

FLAGS = flags.FLAGS

flags.DEFINE_boolean(
  "disable_active_learning",
  False,
  "Set True to disable active learner from learning feature space."
  "All candidate feature vectors have equal likelihood of being important"
)

def AssertConfigIsValid(config: active_learning_pb2.ActiveLearner) -> active_learning_pb2.ActiveLearner:
  """
  Parse proto description and check for validity.
  """
  pbutil.AssertFieldConstraint(
    config,
    "downstream_task",
    lambda x: x in downstream_tasks.TASKS,
    "Downstream task has to be one of {}".format(', '.join([str(x) for x in downstream_tasks.TASKS]))
  )
  if config.HasField("committee"):
    com_config.AssertConfigIsValid(config)
  else:
    raise NotImplementedError(config)
  return config

class Model(object):
  """Predictive models for active learning.

  Please note model instances should be treated as immutable. Upon
  instantiation, a model's properties are used to determine its hash. If you
  modify a property after instantiation, the hash will be out of date, which
  can lead to bad things happening.
  """

  def __init__(self, config: active_learning_pb2.ActiveLearner, cache_path: pathlib.Path):
    """Instantiate a model.

    Args:
      config: A Model message.

    Raises:
      TypeError: If the config argument is not a Model proto.
      UserError: In case on an invalid config.
    """
    # Error early, so that a cache isn't created.
    if not isinstance(config, active_learning_pb2.ActiveLearner):
      t = type(config).__name__
      raise TypeError(f"Config must be an ActiveLearner proto. Received: '{t}'")

    self.config = active_learning_pb2.ActiveLearner()
    # Validate config options.
    self.config.CopyFrom(AssertConfigIsValid(config))

    self.cache_path = cache_path / "active_model"
    if environment.WORLD_RANK == 0:
      self.cache_path.mkdir(exist_ok = True, parents = True)
      (self.cache_path / "samples").mkdir(exist_ok = True)
    distrib.barrier()

    (self.cache_path / "downstream_task").mkdir(exist_ok = True, parents = True)
    self.downstream_task = downstream_tasks.DownstreamTask.FromTask(
      self.config.downstream_task,
      pathlib.Path(self.config.training_corpus).resolve(),
      self.cache_path / "downstream_task",
      self.config.random_seed,
    )

    if environment.WORLD_RANK == 0:
      ## Store current commit
      commit.saveCommit(self.cache_path)
    self.backend = active_committee.QueryByCommittee(self.config, self.cache_path, self.downstream_task)
    l.logger().info("Initialized {} in {}".format(self.backend, self.cache_path))
    return

  def Train(self, **kwargs) -> "Model":
    """Train the model.

    Returns:
      The model instance.

    Raises:
      UnableToAcquireLockError: If the model is locked (i.e. there is another
        process currently modifying the model).
    """
    if FLAGS.disable_active_learning:
      l.logger().warn("Active learning has been disabled. Skip training.")
    else:
      self.backend.Train(**kwargs)
    return self

  def UpdateLearn(self, update_dataloader: 'torch.utils.data.Dataset') -> None:
    """
    Train-update active learner with new generated datapoints.
    """
    if FLAGS.disable_active_learning:
      l.logger().warn("Active learning has been disabled. Skip update training.")
    else:
      self.Train(update_dataloader = update_dataloader)
    return

  def Sample(self, num_samples: int = 512) -> typing.List[typing.Dict[str, float]]:
    """
    Sample the active learner.
    Knowing a downstream task, the active learning model samples
    and returns the datapoints that are deemed valuable.
    """
    sample_set = self.downstream_task.sample_space(num_samples = num_samples)
    if FLAGS.disable_active_learning:
      l.logger().warn("Active learning has been disabled. Skip update training.")
      return [
        {
          'idx'             : int(x['idx']),
          'static_features' : self.downstream_task.VecToStaticFeatDict(x['static_features'].numpy()),
          'runtime_features': self.downstream_task.VecToRuntimeFeatDict(x['runtime_features'].numpy()),
          'input_features'  : self.downstream_task.VecToInputFeatDict(x['input_ids'].numpy()),
        } for x in
        sample_set.get_random_subset(num = len(sample_set), seed = self.config.random_seed).dataset
      ]
    else:
      return self.backend.Sample(sample_set = sample_set)

  def SamplerCache(self, sampler: 'samplers.Sampler') -> pathlib.Path:
    """Get the path to a sampler cache.

    Args:
      sampler: A Sampler instance.

    Returns:
      A path to a directory. Note that this directory may not exist - it is
      created only after a call to Sample().
    """
    return self.cache_path / "samples" / sampler.hash

  @property
  def is_trained(self) -> bool:
    return self.backend.is_trained
