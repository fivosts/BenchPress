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
import pathlib
import typing

import numpy as np

from deeplearning.clgen import cache

from deeplearning.clgen import sample_observers as sample_observers_lib
from deeplearning.clgen import samplers
from deeplearning.clgen import telemetry
from deeplearning.clgen.corpuses import atomizers
from deeplearning.clgen.corpuses import corpuses
from deeplearning.clgen.dashboard import dashboard_db
from deeplearning.clgen.models import builders
from deeplearning.clgen.models.keras_sequential import keras_sequential
from deeplearning.clgen.models.tf_sequential import tf_sequential
from deeplearning.clgen.models.tf_bert import tf_bert
from deeplearning.clgen.proto import internal_pb2
from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen.proto import telemetry_pb2
from labm8.py import app
from labm8.py import crypto
from labm8.py import humanize
from labm8.py import labdate
from labm8.py import lockfile
from labm8.py import logutil
from deeplearning.clgen import pbutil
from labm8.py import system

from eupy.native import logger as l

FLAGS = flags.FLAGS


class Model(object):
  """A CLgen language model.

  Please note model instances should be treated as immutable. Upon
  instantiation, a model's properties are used to determine its hash. If you
  modify a property after instantiation, the hash will be out of date, which
  can lead to bad things happening.
  """

  def __init__(self, config: model_pb2.Model):
    """Instantiate a model.

    Args:
      config: A Model message.

    Raises:
      TypeError: If the config argument is not a Model proto.
      UserError: In case on an invalid config.
    """
    l.getLogger().debug("deeplearning.clgen.models.Model.__init__()")
    # Error early, so that a cache isn't created.
    if not isinstance(config, model_pb2.Model):
      t = type(config).__name__
      raise TypeError(f"Config must be a Model proto. Received: '{t}'")

    self.config = model_pb2.Model()
    # Validate config options.
    self.config.CopyFrom(builders.AssertIsBuildable(config))
    self.corpus = corpuses.Corpus(config.corpus)
    self.hash = self._ComputeHash(self.corpus, self.config)
    self.cache = cache.mkcache("model", self.hash)
    # Create the necessary cache directories.
    (self.cache.path / "checkpoints").mkdir(exist_ok=True)
    (self.cache.path / "samples").mkdir(exist_ok=True)
    (self.cache.path / "logs").mkdir(exist_ok=True)

    self._created = False
    self.dashboard_db = dashboard_db.GetDatabase()
    self._dashboard_db_id: typing.Optional[int] = None

    # Create symlink to encoded corpus.
    symlink = self.cache.path / "corpus"
    if not symlink.is_symlink():
      os.symlink(
        os.path.relpath(
          pathlib.Path(self.corpus.encoded.url[len("sqlite:///") :]).parent,
          self.cache.path,
        ),
        symlink,
      )

    # Create symlink to the atomizer.
    symlink = self.cache.path / "atomizer"
    if not symlink.is_symlink():
      os.symlink(
        os.path.relpath(self.corpus.atomizer_path, self.cache.path), symlink
      )

    # Validate metadata against cache.
    if self.cache.get("META.pbtxt"):
      cached_meta = pbutil.FromFile(
        pathlib.Path(self.cache["META.pbtxt"]), internal_pb2.ModelMeta()
      )
      # Exclude num_epochs and corpus location from metadata comparison.
      config_to_compare = model_pb2.Model()
      config_to_compare.CopyFrom(self.config)
      config_to_compare.corpus.ClearField("contentfiles")
      config_to_compare.training.ClearField("num_epochs")
      # These fields should have already been cleared, but we'll do it again
      # so that metadata comparisons don't fail when the cached meta schema
      # is updated.
      cached_to_compare = model_pb2.Model()
      cached_to_compare.CopyFrom(cached_meta.config)
      cached_to_compare.corpus.ClearField("contentfiles")
      cached_to_compare.training.ClearField("num_epochs")
      if cached_to_compare.training.sequence_length != config_to_compare.training.sequence_length:
        l.getLogger().warning("Mismatch between pre-trained and current config sequence_length!\
          This can only be intended in tfBert model!")
      cached_to_compare.training.ClearField("sequence_length")
      config_to_compare.training.ClearField("sequence_length")
      if config_to_compare != cached_to_compare:
        raise SystemError("Metadata mismatch: {} \n\n {}".format(config_to_compare, cached_to_compare))
      self.meta = cached_meta
    else:
      self.meta = internal_pb2.ModelMeta()
      self.meta.config.CopyFrom(self.config)
      self._WriteMetafile()

    self.backend = {
      model_pb2.NetworkArchitecture.TENSORFLOW_SEQ: tf_sequential.tfSequential,
      model_pb2.NetworkArchitecture.KERAS_SEQ: keras_sequential.kerasSequential,
      model_pb2.NetworkArchitecture.TENSORFLOW_BERT: tf_bert.tfBert,
    }[config.architecture.backend](self.config, self.cache)

  def GetShortSummary(self) -> str:
    l.getLogger().debug("deeplearning.clgen.models.Model.GetShortSummary()")
    return self.backend.GetShortSummary()

  @staticmethod
  def _ComputeHash(corpus_: corpuses.Corpus, config: model_pb2.Model) -> str:
    """Compute model hash.

    The hash is computed from the ID of the corpus and the serialized
    representation of the config proto. The number of epochs that the model is
    trained for does not affect the hash, since we can share checkpoints
    between different models if the only variable is the epoch count. E.g.
    we have a model trained for 10 epochs, we can use the checkpoint as the
    starting point for a training a model for 20 epochs.

    Args:
      corpus: A corpus instance.
      config: A Model config proto.

    Returns:
      The unique model ID.
    """
    l.getLogger().debug("deeplearning.clgen.models.Model._ComputeHash()")
    config_to_hash = model_pb2.Model()
    config_to_hash.CopyFrom(config)
    config_to_hash.ClearField("corpus")
    config_to_hash.training.ClearField("num_epochs")
    return crypto.sha1_list(corpus_.hash, config_to_hash.SerializeToString())

  def Create(self) -> bool:
    l.getLogger().debug("deeplearning.clgen.models.Model.Create()")
    if self._created:
      return False
    self._created = True
    self.corpus.Create()
    self.backend.Create(atomizer = self.corpus.atomizer)

    # Add entry to dashboard database
    with self.dashboard_db.Session(commit=True) as session:
      config_to_store = model_pb2.Model()
      config_to_store.CopyFrom(self.config)
      config_to_store.ClearField("corpus")
      config_to_store.training.ClearField("num_epochs")
      corpus = session.GetOrAdd(
        dashboard_db.Model,
        corpus_id=self.corpus.dashboard_db_id,
        config_proto_sha1=crypto.sha1(config_to_store.SerializeToString()),
        config_proto=str(config_to_store),
        cache_path=(
          f"ssh://{system.USERNAME}@{system.HOSTNAME}" f"/{self.cache.path}"
        ),
        summary=self.GetShortSummary(),
      )
      session.flush()
      self._dashboard_db_id = corpus.id
      self.backend.dashboard_model_id = self.dashboard_db_id
      self.backend.dashboard_db = self.dashboard_db

  @property
  def dashboard_db_id(self) -> int:
    l.getLogger().debug("deeplearning.clgen.models.Model.dashboard_db_id()")
    if not self._created:
      raise TypeError("Cannot access dashboard_db_id before Create() called")
    return self._dashboard_db_id

  def Train(self, **kwargs) -> "Model":
    """Train the model.

    Returns:
      The model instance.

    Raises:
      UnableToAcquireLockError: If the model is locked (i.e. there is another
        process currently modifying the model).
    """
    l.getLogger().debug("deeplearning.clgen.models.Model.Train()")
    self.Create()
    with self.training_lock.acquire():
      self.backend.Train(self.corpus, **kwargs)
    # telemetry_logs = self.TrainingTelemetry()[: self.backend.num_epochs]
    telemetry_logs = self.backend.telemetry.EpochTelemetry()

    if len(telemetry_logs) != self.backend.num_epochs:
      raise ValueError("Epoch telemetry logs contain {} epoch entries, but model has {} epochs!"
                              .format(
                                  len(telemetry_logs),
                                  self.backend.num_epochs,
                                )
                              )
    final_loss = telemetry_logs[-1].loss
    total_time_ms = sum(t.epoch_wall_time_ms for t in telemetry_logs)
    l.getLogger().info(
      "Trained model for {} epochs in {} ms. " "Training loss: {}."
        .format(
          self.backend.num_epochs,
          humanize.Commas(total_time_ms),
          final_loss,
          )
    )
    return self

  def Sample(
    self,
    sampler: samplers.Sampler,
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
    l.getLogger().debug("deeplearning.clgen.models.Model.Sample()")
    if not sample_observers:
      raise ValueError("Cannot sample without any observers")

    sample_start_time = labdate.MillisecondsTimestamp()

    self.Train()

    with logutil.TeeLogsToFile(
      f"sampler_{sampler.hash}", self.cache.path / "logs"
    ):
      l.getLogger().info("Sampling: '{}'".format(sampler.start_text))

      atomizer = self.corpus.atomizer
      sampler.Specialize(atomizer)
      self.backend.InitSampling(sampler, seed)
      [obs.Specialize(self, sampler) for obs in sample_observers]

      batch_count = 1
      while self._SampleBatch(sampler, atomizer, sample_observers):
        batch_count += 1

      time_now = labdate.MillisecondsTimestamp()
      l.getLogger().info( "Produced {} sample batches at a rate of {} ms / batch."
                          .format(
                            humanize.Commas(batch_count),
                            humanize.Commas(int((time_now - sample_start_time) / max(batch_count, 1)))
                          )
      )

  def _SampleBatch(
    self,
    sampler: samplers.Sampler,
    atomizer: atomizers.AtomizerBase,
    sample_observers: typing.List[sample_observers_lib.SampleObserver],
  ) -> bool:
    """Run a single iteration of the batched sample inner-loop."""
    l.getLogger().debug("deeplearning.clgen.models.Model._SampleBatch()")
    samples_in_progress = [
      sampler.tokenized_start_text.copy() for _ in range(sampler.batch_size)
    ]
    done = np.zeros(sampler.batch_size, dtype=np.bool)
    start_time = labdate.MillisecondsTimestamp()
    wall_time_start = start_time

    self.backend.InitSampleBatch(sampler)

    # The return value of this method. If any of the sample_observers return
    # False, this value is set to False.
    continue_sampling = True

    # Sampling loop. Continues until all samples in the batch are done.
    while not done.all():
      indices = self.backend.SampleNextIndices(sampler, done)

      # Iterate over all samples in batch to determine whether they're
      # done.
      for i in range(sampler.batch_size):
        if done[i]:
          continue

        for index in indices[i]:
          samples_in_progress[i].append(atomizer.decoder[index])
          if sampler.SampleIsComplete(samples_in_progress[i]):
            end_time = labdate.MillisecondsTimestamp()
            done[i] = 1
            sample = model_pb2.Sample(
              text="".join(samples_in_progress[i]),
              sample_start_epoch_ms_utc=start_time,
              sample_time_ms=end_time - start_time,
              wall_time_ms=end_time - wall_time_start,
              num_tokens=len(samples_in_progress[i]),
            )
            # Notify sample observers.
            continue_sampling &= all(
              [obs.OnSample(sample) for obs in sample_observers]
            )

            # Wall sample time is the difference between the end of the previous
            # sample and the end of the current sample.
            wall_time_start = labdate.MillisecondsTimestamp()
            break

    return continue_sampling

  def SamplerCache(self, sampler: samplers.Sampler) -> pathlib.Path:
    """Get the path to a sampler cache.

    Args:
      sampler: A Sampler instance.

    Returns:
      A path to a directory. Note that this directory may not exist - it is
      created only after a call to Sample().
    """
    l.getLogger().debug("deeplearning.clgen.models.Model.SamplerCache()")
    return self.cache.path / "samples" / sampler.hash

  def _WriteMetafile(self) -> None:
    l.getLogger().debug("deeplearning.clgen.models.Model._WriteMetafile()")
    pbutil.ToFile(self.meta, pathlib.Path(self.cache.keypath("META.pbtxt")))

  def TrainingTelemetry(self) -> typing.List[telemetry_pb2.ModelEpochTelemetry]:
    """Get the training telemetry data."""
    l.getLogger().debug("deeplearning.clgen.models.Model.TrainingTelemetry()")
    return telemetry.TrainingLogger(self.cache.path / "logs").EpochTelemetry()

  def InferenceManifest(self) -> typing.List[pathlib.Path]:
    """Return the list of files which are required for model inference.

    Returns:
      A list of absolute paths.
    """
    l.getLogger().debug("deeplearning.clgen.models.Model.InferenceManifest()")
    return sorted(
      [self.cache.path / "atomizer", self.cache.path / "META.pbtxt",]
      + self.backend.InferenceManifest()
    )

  @property
  def atomizer(self) -> atomizers.AtomizerBase:
    l.getLogger().debug("deeplearning.clgen.models.Model.atomizer()")
    return self.corpus.atomizer

  @property
  def training_lock(self) -> lockfile.LockFile:
    l.getLogger().debug("deeplearning.clgen.models.Model.training_lock()")
    """A lockfile for exclusive training."""
    return lockfile.LockFile(self.cache.keypath("LOCK"))

  @property
  def is_trained(self) -> bool:
    l.getLogger().debug("deeplearning.clgen.models.Model.is_trained()")
    return self.backend.is_trained

  def __repr__(self) -> str:
    """String representation."""
    l.getLogger().debug("deeplearning.clgen.models.Model.__repr__()")
    return f"model[{self.hash}]"

  def __eq__(self, rhs) -> bool:
    l.getLogger().debug("deeplearning.clgen.models.Model.__eq__()")
    if not isinstance(rhs, Model):
      return False
    return rhs.hash == self.hash

  def __ne__(self, rhs) -> bool:
    l.getLogger().debug("deeplearning.clgen.models.Model.__ne__()")
    return not self.__eq__(rhs)
