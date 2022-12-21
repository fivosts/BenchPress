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
"""Active Learning feature space models."""
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
from deeplearning.benchpress.active_models.expected_error_reduction import eer
from deeplearning.benchpress.active_models.expected_error_reduction import config as eer_config
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

flags.DEFINE_integer(
  "num_active_samples",
  256,
  "Select number of points you want to sample with active learner."
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
  pbutil.AssertFieldIsSet(config, "training_corpus")
  pbutil.AssertFieldIsSet(config, "num_train_steps")
  pbutil.AssertFieldIsSet(config, "random_seed")
  p = pathlib.Path(config.training_corpus).resolve()
  if not p.exists() and config.num_train_steps > 0:
    raise FileNotFoundError(p)

  if config.HasField("query_by_committee"):
    com_config.AssertConfigIsValid(config.query_by_committee)
  elif config.HasField("expected_error_reduction"):
    eer_config.AssertConfigIsValid(config.expected_error_reduction)
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

  def __init__(self,
               config            : active_learning_pb2.ActiveLearner,
               cache_path        : pathlib.Path,
               hidden_state_size : int = None,
               ):
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
      hidden_state_size = hidden_state_size,
      test_db = pathlib.Path(self.config.test_db).resolve() if self.config.HasField("test_db") else None
    )

    if environment.WORLD_RANK == 0:
      ## Store current commit
      commit.saveCommit(self.cache_path)
    if self.config.HasField("query_by_committee"):
      self.backend = active_committee.QueryByCommittee(self.config, self.cache_path, self.downstream_task)
    elif self.config.HasField("expected_error_reduction"):
      self.backend = eer.ExpectedErrorReduction(self.config, self.cache_path, self.downstream_task)
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

  def Sample(self, num_samples: int = None) -> typing.List[typing.Dict[str, float]]:
    """
    Sample the active learner.
    Knowing a downstream task, the active learning model samples
    and returns the datapoints that are deemed valuable.
    """
    sample_set = self.downstream_task.sample_space(num_samples = FLAGS.num_active_samples if num_samples is None else num_samples)
    if FLAGS.disable_active_learning:
      l.logger().warn("Active learning has been disabled. Skip update training.")
      l.logger().warn("This is passive learning mode to illustrate AL's significance.")
      l.logger().warn("Instead of querying, a random datapoint is returned.")
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
