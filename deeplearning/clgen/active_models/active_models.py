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

from deeplearning.clgen.util import pbutil
from deeplearning.clgen.util import cache
from deeplearning.clgen.util import crypto
from deeplearning.clgen.util import commit
from deeplearning.clgen.util import sqlutil
from deeplearning.clgen.util import environment
from deeplearning.clgen.util import distrib
from deeplearning.clgen.active_models import downstream_tasks
from deeplearning.clgen.active_models.committee import active_committee
from deeplearning.clgen.active_models.committee import config as com_config
from deeplearning.clgen.samplers import sample_observers as sample_observers_lib
from deeplearning.clgen.proto import active_learning_pb2
from absl import flags

from deeplearning.clgen.util import logging as l

FLAGS = flags.FLAGS

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

  def __init__(self, config: active_learning_pb2.ActiveLearner):
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

    distrib.lock()
    self.cache = cache.mkcache("active_model")
    distrib.unlock()

    self.downstream_task = downstream_tasks.DownstreamTask.FromTask(
      self.config.downstream_task, self.config.training_corpus
    )

    if environment.WORLD_RANK == 0:
      ## Store current commit
      commit.saveCommit(self.cache.path)
    self.backend = active_committee.ActiveCommittee(self.config, self.cache, self.downstream_task)
    l.logger().info("Initialized {} in {}".format(self.backend, self.cache.path))
    return

  def Train(self, **kwargs) -> "Model":
    """Train the model.

    Returns:
      The model instance.

    Raises:
      UnableToAcquireLockError: If the model is locked (i.e. there is another
        process currently modifying the model).
    """
    self.backend.Train(**kwargs)
    l.logger().info(
      "Trained model for {} {} in {} ms. " "Training loss: {}."
        .format(
          telemetry_logs[-1].epoch_num,
          "steps",
          humanize.intcomma(sum(t.epoch_wall_time_ms for t in telemetry_logs)),
          telemetry_logs[-1].loss,
          )
    )
    return self

  def Sample(
    self,
    sampler: 'samplers.Sampler',
    sample_observers: typing.List[sample_observers_lib.SampleObserver],
    seed: int = None,
  ) -> None:
    """Sample a model.

    This method uses the observer model, returning nothing. To access the
    samples produced, implement a SampleObserver and pass it in as an argument.
    Sampling continues indefinitely until one of the sample observers returns
    False when notified of a new sample.

    If the model is not already trained, calling Sample() first trains the
    model. Thus a call to Sample() is equivalent to calling Train() then
    Sample().

    Args:
      sampler: The sampler to sample using.
      sample_observers: A list of SampleObserver objects that are notified of
        new generated samples.
      seed: A numeric value to seed the RNG with. If not present, the RNG is
        seeded randomly.

    Raises:
      UserError: If called with no sample observers.
      UnableToAcquireLockError: If the model is locked (i.e. there is another
        process currently modifying the model).
      InvalidStartText: If the sampler start text cannot be encoded.
      InvalidSymtokTokens: If the sampler symmetrical depth tokens cannot be
        encoded.
    """
    if not sample_observers:
      raise ValueError("Cannot sample without any observers")

    self.Create()
    epoch = self.backend.telemetry.EpochTelemetry()[-1].epoch_num
    sample_start_time = datetime.datetime.utcnow()    

    if environment.WORLD_RANK == 0:
      (self.cache.path / "samples" / sampler.hash).mkdir(exist_ok = True)
    tokenizer = self.corpus.tokenizer
    if sampler.isFixedStr and not sampler.is_active:
      sampler.Specialize(tokenizer)
    elif sampler.is_live:
      start_text = [str(input("Live Feed: "))]
      while True:
        try:
          start_text.append(str(input()))
        except EOFError:
          break
      sampler.start_text = '\n'.join(start_text)
      sampler.Specialize(tokenizer)

    self.backend.InitSampling(sampler, seed, self.corpus)
    [obs.Specialize(self, sampler) for obs in sample_observers]

    if isinstance(self.backend, tf_bert.tfBert) or isinstance(self.backend, torch_bert.torchBert):
      sample_batch = lambda : self._SampleLMBatch(sampler, tokenizer, sample_observers, epoch)
    elif isinstance(self.backend, tf_sequential.tfSequential) or isinstance(self.backend, keras_sequential.kerasSequential):
      sample_batch = lambda : self._SampleSeqBatch(sampler, tokenizer, sample_observers, epoch)
    else:
      raise ValueError("Unrecognized backend.")

    try:
      seq_count, cont = 0, True
      while cont:
        cont, seq_count = sample_batch()
        if sampler.is_live:
          start_text = [str(input("Live Feed: "))]
          while True:
            try:
              start_text.append(str(input()))
            except EOFError:
              break
          sampler.start_text = '\n'.join(start_text)
          sampler.Specialize(tokenizer)
    except KeyboardInterrupt:
      l.logger().info("Wrapping up sampling...")
    except Exception as e:
      raise e

    for obs in sample_observers:
      obs.endSample()
    if isinstance(self.backend, torch_bert.torchBert) and sampler.is_active:
      self.backend.sample.data_generator.samples_cache_obs.endSample()

    time_now = datetime.datetime.utcnow()
    l.logger().info( "Produced {} samples at a rate of {} ms / sample."
                        .format(
                          humanize.intcomma(seq_count),
                          humanize.intcomma(int(1000 * ((time_now - sample_start_time) / max(seq_count, 1)).total_seconds()))
                        )
    )

  def SamplerCache(self, sampler: 'samplers.Sampler') -> pathlib.Path:
    """Get the path to a sampler cache.

    Args:
      sampler: A Sampler instance.

    Returns:
      A path to a directory. Note that this directory may not exist - it is
      created only after a call to Sample().
    """
    return self.cache.path / "samples" / sampler.hash

  @property
  def is_trained(self) -> bool:
    return self.backend.is_trained
