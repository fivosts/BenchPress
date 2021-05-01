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
import socket
import getpass
import pathlib
import typing
import datetime
import humanize

import numpy as np

from deeplearning.clgen.samplers import sample_observers as sample_observers_lib
from deeplearning.clgen.samplers import samplers
from deeplearning.clgen.util import pbutil
from deeplearning.clgen.util import cache
from deeplearning.clgen.util import crypto
from deeplearning.clgen.util import commit
from deeplearning.clgen.features import extractor
from deeplearning.clgen.corpuses import tokenizers
from deeplearning.clgen.corpuses import corpuses
from deeplearning.clgen.dashboard import dashboard_db
from deeplearning.clgen.models import builders
from deeplearning.clgen.models import telemetry
from deeplearning.clgen.models.keras_sequential import keras_sequential
from deeplearning.clgen.models.tf_sequential import tf_sequential
from deeplearning.clgen.models.tf_bert import tf_bert
from deeplearning.clgen.models.torch_bert import torch_bert
from deeplearning.clgen.proto import internal_pb2
from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen.proto import telemetry_pb2
from deeplearning.clgen.preprocessors import opencl
from absl import flags
from labm8.py import sqlutil

from eupy.native import logger as l

FLAGS = flags.FLAGS

flags.DEFINE_integer(
  "num_train_steps",
  None,
  "Bypass num_train_steps provided by protobuf file."
)

flags.DEFINE_integer(
  "num_pretrain_steps",
  None,
  "Bypass num_pretrain_steps provided by protobuf file."
)

flags.DEFINE_integer(
  "num_epochs",
  None,
  "Bypass num_epochs provided by protobuf file."
)

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
    # Error early, so that a cache isn't created.
    if not isinstance(config, model_pb2.Model):
      t = type(config).__name__
      raise TypeError(f"Config must be a Model proto. Received: '{t}'")

    self.config = model_pb2.Model()
    # Validate config options.
    self.config.CopyFrom(builders.AssertIsBuildable(config))
    if FLAGS.num_train_steps:
      self.config.training.num_train_steps = FLAGS.num_train_steps
    if FLAGS.num_pretrain_steps:
      self.config.training.num_pretrain_steps = FLAGS.num_pretrain_steps
    if FLAGS.num_epochs:
      self.config.training.num_epochs = FLAGS.num_epochs
      
    self.corpus           = corpuses.Corpus(config.corpus)
    self.pre_train_corpus = None
    if config.HasField("pre_train_corpus"):
      self.pre_train_corpus = corpuses.Corpus(config.pre_train_corpus)
    self.hash = self._ComputeHash(self.pre_train_corpus, self.corpus, self.config)
    self.cache = cache.mkcache("model", self.hash)
    # Create the necessary cache directories.
    (self.cache.path / "checkpoints").mkdir(exist_ok=True)
    (self.cache.path / "samples").mkdir(exist_ok=True)

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
    if self.pre_train_corpus:
      symlink = self.cache.path / "pre_train_corpus"
      if not symlink.is_symlink():
        os.symlink(
          os.path.relpath(
            pathlib.Path(self.pre_train_corpus.encoded.url[len("sqlite:///") :]).parent,
            self.cache.path,
          ),
          symlink,
        )

    # Create symlink to the tokenizer.
    symlink = self.cache.path / "tokenizer"
    if not symlink.is_symlink():
      os.symlink(
        os.path.relpath(self.corpus.tokenizer_path, self.cache.path), symlink
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
      if config_to_compare.HasField("pre_train_corpus"):
        config_to_compare.pre_train_corpus.ClearField("contentfiles")
      config_to_compare.training.ClearField("num_epochs")
      config_to_compare.training.ClearField("num_train_steps")
      if config_to_compare.model.HasField("pre_train_corpus"):
        config_to_compare.training.ClearField("num_pretrain_steps")
      config_to_compare.training.ClearField("batch_size")
      if config_to_compare.training.HasField("data_generator"):
        config_to_compare.training.data_generator.ClearField("steps_per_epoch")
        config_to_compare.training.data_generator.ClearField("validation_set")
      # These fields should have already been cleared, but we'll do it again
      # so that metadata comparisons don't fail when the cached meta schema
      # is updated.
      cached_to_compare = model_pb2.Model()
      cached_to_compare.CopyFrom(cached_meta.config)
      cached_to_compare.corpus.ClearField("contentfiles")
      if cached_to_compare.HasField("pre_train_corpus"):
        cached_to_compare.pre_train_corpus.ClearField("contentfiles")
      cached_to_compare.training.ClearField("num_epochs")
      cached_to_compare.training.ClearField("num_train_steps")
      if cached_to_compare.model.HasField("pre_train_corpus"):
        cached_to_compare.training.ClearField("num_pretrain_steps")
      cached_to_compare.training.ClearField("batch_size")
      if cached_to_compare.training.HasField("data_generator"):
        cached_to_compare.training.data_generator.ClearField("steps_per_epoch")
        cached_to_compare.training.data_generator.ClearField("validation_set")
      if cached_to_compare.training.sequence_length != config_to_compare.training.sequence_length:
        l.getLogger().warning("Mismatch between pre-trained and current config sequence_length!\
          This can only be intended in BERT model!")
      cached_to_compare.training.ClearField("sequence_length")
      config_to_compare.training.ClearField("sequence_length")
      if config_to_compare != cached_to_compare:
        raise SystemError("Metadata mismatch: {} \n\n {}".format(config_to_compare, cached_to_compare))
      self.meta = cached_meta
    else:
      self.meta = internal_pb2.ModelMeta()
      self.meta.config.CopyFrom(self.config)
      self._WriteMetafile()

    ## Store current commit
    commit.saveCommit(self.cache.path)

    self.backend = {
      model_pb2.NetworkArchitecture.TENSORFLOW_SEQ: tf_sequential.tfSequential,
      model_pb2.NetworkArchitecture.KERAS_SEQ: keras_sequential.kerasSequential,
      model_pb2.NetworkArchitecture.TENSORFLOW_BERT: tf_bert.tfBert,
      model_pb2.NetworkArchitecture.TORCH_BERT: torch_bert.torchBert,
    }[config.architecture.backend](self.config, self.cache, self.hash)

  def GetShortSummary(self) -> str:
    return self.backend.GetShortSummary()

  @staticmethod
  def _ComputeHash(pre_train_corpus_ : corpuses.Corpus,
                   corpus_           : corpuses.Corpus,
                   config            : model_pb2.Model,
                   ) -> str:
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
    config_to_hash = model_pb2.Model()
    config_to_hash.CopyFrom(config)
    config_to_hash.ClearField("pre_train_corpus")
    config_to_hash.ClearField("corpus")
    config_to_hash.training.ClearField("num_epochs")
    config_to_hash.training.ClearField("num_train_steps")
    config_to_hash.training.ClearField("batch_size")
    if config_to_hash.training.HasField("data_generator"):
      config_to_hash.training.data_generator.ClearField("steps_per_epoch")
      config_to_hash.training.data_generator.ClearField("validation_set")
    if pre_train_corpus_:
      hash_list = [pre_train_corpus_.hash, corpus_.hash, config_to_hash.SerializeToString()]
    else:
      hash_list = [corpus_.hash, config_to_hash.SerializeToString()]
    return crypto.sha1_list(hash_list)

  def Create(self) -> bool:
    if self._created:
      return False
    self._created = True
    self.corpus.Create()
    if self.pre_train_corpus:
      self.pre_train_corpus.Create(self.corpus.tokenizer)
    self.backend.Create(tokenizer = self.corpus.tokenizer)

    # Add entry to dashboard database
    with self.dashboard_db.Session(commit=True) as session:
      config_to_store = model_pb2.Model()
      config_to_store.CopyFrom(self.config)
      config_to_store.ClearField("corpus")
      config_to_store.ClearField("pre_train_corpus")
      config_to_store.training.ClearField("num_epochs")
      corpus = session.GetOrAdd(
        dashboard_db.Model,
        corpus_id         = self.corpus.dashboard_db_id,
        config_proto_sha1 = crypto.sha1(config_to_store.SerializeToString()),
        config_proto      = str(config_to_store),
        cache_path        = (f"ssh://{socket.gethostname()}@{getpass.getuser()}" f"/{self.cache.path}"),
        summary           = self.GetShortSummary(),
      )
      session.flush()
      self._dashboard_db_id           = corpus.id
      self.backend.dashboard_model_id = self.dashboard_db_id
      self.backend.dashboard_db       = self.dashboard_db

  @property
  def dashboard_db_id(self) -> int:
    if not self._created:
      raise TypeError("Cannot access dashboard_db_id before Create() called")
    return self._dashboard_db_id

  def PreTrain(self, **kwargs) -> "Model":
    """
    Pre-Train the model. Only supported for PyTorch BERT.

    Returns:
      The model instance.

    Raises:
      UnableToAcquireLockError: If the model is locked (i.e. there is another
        process currently modifying the model).
    """
    self.Create()

    self.backend.PreTrain(self.pre_train_corpus, **kwargs)
    pre_telemetry_logs = self.backend.pre_telemetry.EpochTelemetry()

    l.getLogger().info(
      "Pre-trained model for {} {} in {} ms. " "Training loss: {}."
        .format(
          pre_telemetry_logs[-1].epoch_num,
          "steps" if isinstance(self.backend, tf_bert.tfBert) or isinstance(self.backend, torch_bert.torchBert) else "epochs",
          humanize.intcomma(sum(t.epoch_wall_time_ms for t in pre_telemetry_logs)),
          pre_telemetry_logs[-1].loss,
          )
    )
    return self

  def Train(self, **kwargs) -> "Model":
    """Train the model.

    Returns:
      The model instance.

    Raises:
      UnableToAcquireLockError: If the model is locked (i.e. there is another
        process currently modifying the model).
    """
    self.Create()

    self.backend.Train(self.corpus, **kwargs)
    telemetry_logs = self.backend.telemetry.EpochTelemetry()

    l.getLogger().info(
      "Trained model for {} {} in {} ms. " "Training loss: {}."
        .format(
          telemetry_logs[-1].epoch_num,
          "steps" if isinstance(self.backend, tf_bert.tfBert) or isinstance(self.backend, torch_bert.torchBert) else "epochs",
          humanize.intcomma(sum(t.epoch_wall_time_ms for t in telemetry_logs)),
          telemetry_logs[-1].loss,
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
    if not sample_observers:
      raise ValueError("Cannot sample without any observers")

    self.Create()
    epoch = self.backend.telemetry.EpochTelemetry()[-1].epoch_num
    sample_start_time = datetime.datetime.utcnow()    

    (self.cache.path / "samples" / sampler.hash).mkdir(exist_ok = True)
    tokenizer = self.corpus.tokenizer
    if sampler.isFixedStr:
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

    self.backend.InitSampling(sampler, seed)
    [obs.Specialize(self, sampler) for obs in sample_observers]

    batch_count = 0
    if isinstance(self.backend, tf_bert.tfBert) or isinstance(self.backend, torch_bert.torchBert):
      sample_batch = lambda : self._SampleLMBatch(sampler, tokenizer, sample_observers, epoch)
    elif isinstance(self.backend, tf_sequential.tfSequential) or isinstance(self.backend, keras_sequential.kerasSequential):
      sample_batch = lambda : self._SampleSeqBatch(sampler, tokenizer, sample_observers, epoch)
    else:
      raise ValueError("Unrecognized backend.")

    try:
      while sample_batch():
        batch_count += 1
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
      l.getLogger().info("Wrapping up sampling...")

    for obs in sample_observers:
      obs.endSample()

    time_now = datetime.datetime.utcnow()
    l.getLogger().info( "Produced {} sample batches at a rate of {} ms / batch."
                        .format(
                          humanize.intcomma(batch_count),
                          humanize.intcomma(int(1000 * ((time_now - sample_start_time) / max(batch_count, 1)).total_seconds()))
                        )
    )

  def _SampleLMBatch(self,
                     sampler: samplers.Sampler,
                     tokenizer: tokenizers.TokenizerBase,
                     sample_observers: typing.List[sample_observers_lib.SampleObserver],
                     epoch: int,
                     ) -> bool:
    """
    Run a sampling iteration over BERT models.
    """
    start_time = datetime.datetime.utcnow()
    self.backend.InitSampleBatch(sampler)
    org_inputs, input_ids, samples, indices = self.backend.SampleNextIndices(sampler)

    if not indices:
      # Return empty means model has not produced something that can be stored.
      # This if accommodates active sampling, which is very selective.
      return True

    continue_sampling = True
    for org, inp, sample, idxs in zip(org_inputs, input_ids, samples, indices):

      src = self.tokenizer.ArrayToCode(sample, with_formatting = True)
      features = extractor.DictKernelFeatures(src)
      try:
        stdout = opencl.Compile(src)
        compile_flag = True
      except ValueError:
        compile_flag = False

      end_time = datetime.datetime.utcnow()
      sample = model_pb2.Sample(
        train_step                = epoch,
        text                      = src,
        sample_indices            = '\n'.join([','.join([self.tokenizer.decoder[idx] for idx in hole_idxs]).replace('\n', '\\n') for hole_idxs in idxs]),
        encoded_sample_indices    = '\n'.join([','.join([str(idx) for idx in hole_idxs]) for hole_idxs in idxs]),
        original_input            = self.tokenizer.tokensToString(org, with_formatting = True, ignore_token = self.tokenizer.padToken),
        sample_feed               = self.tokenizer.tokensToString(inp, with_formatting = False, ignore_token = self.tokenizer.padToken),
        encoded_text              = ",".join([str(x) for x in sample]),
        sample_start_epoch_ms_utc = int(start_time.strftime("%s%f")),
        sample_time_ms            = int(round(1000 * ((end_time - start_time) / len(samples)).total_seconds())),
        wall_time_ms              = int(round(1000 * ((end_time - start_time) / len(samples)).total_seconds())),
        feature_vector            = "\n".join(["{}:{}".format(k, v) for (k, v) in features.items()]),
        num_tokens                = np.where(sample == self.tokenizer.padToken)[0][0] if self.tokenizer.padToken in sample else len(sample),
        compile_status            = compile_flag,
        categorical_sampling      = self.backend.samplesWithCategorical(),
        date_added                = datetime.datetime.utcnow().strftime("%m/%d/%Y, %H:%M:%S"),
      )
      # Notify sample observers.
      continue_sampling &= all(
        [obs.OnSample(sample) for obs in sample_observers]
      )

    return continue_sampling

  def _SampleSeqBatch(
    self,
    sampler: samplers.Sampler,
    tokenizer: tokenizers.TokenizerBase,
    sample_observers: typing.List[sample_observers_lib.SampleObserver],
    epoch: int,
  ) -> bool:
    """
    Run a single iteration of the batched sample inner-loop for sequential models.
    """

    self.backend.InitSampleBatch(sampler)
    samples_in_progress = [
      sampler.tokenized_start_text.copy() for _ in range(sampler.batch_size)
    ]
    done = np.zeros(sampler.batch_size, dtype=np.bool)
    start_time = datetime.datetime.utcnow()
    wall_time_start = start_time

    # The return value of this method. If any of the sample_observers return
    # False, this value is set to False.
    continue_sampling = True

    # Sampling loop. Continues until all samples in the batch are done.
    while not done.all():
      indices = self.backend.SampleNextIndices(sampler, done)
      # Iterate over all samples in batch to determine whether they're
      # done.

      for i in range(len(indices)):
        if done[i]:
          continue

        for index in indices[i]:
          samples_in_progress[i].append(tokenizer.decoder[index])
          step_ind             = ""
          encoded_step_indices = ""

          if sampler.SampleIsComplete(samples_in_progress[i]):
            end_time       = datetime.datetime.utcnow()
            sample_kernel  = [x for x in samples_in_progress[i]]
            feature_vector = extractor.DictKernelFeatures(''.join(samples_in_progress[i]))
            done[i]        = 1
            try:
              stdout = opencl.Compile(''.join(samples_in_progress[i]))
              compile_flag = True
            except ValueError:
              compile_flag = False

            sample = model_pb2.Sample(
              train_step                = epoch,
              text                      = samples_in_progress[i],
              sample_indices            = "",
              encoded_sample_indices    = "",
              sample_feed               = sampler.start_text,
              encoded_text              = ",".join([str(tokenizer.vocab[x]) for x in sample_kernel]),
              sample_start_epoch_ms_utc = int(start_time.strftime("%s%f")),
              sample_time_ms            = int(round(1000 * ((end_time - start_time) / sampler.batch_size).total_seconds())),
              wall_time_ms              = int(round(1000 * ((end_time - start_time) / sampler.batch_size).total_seconds())),
              feature_vector            = "\n".join(["{}:{}".format(k, v) for (k, v) in feature_vector.items()]),
              num_tokens                = len(samples_in_progress[i]),
              compile_status            = compile_flag,
              categorical_sampling      = self.backend.samplesWithCategorical(),
              date_added                = datetime.datetime.utcnow().strftime("%m/%d/%Y, %H:%M:%S"),
            )
            # Notify sample observers.
            continue_sampling &= all(
              [obs.OnSample(sample) for obs in sample_observers]
            )
            # Wall sample time is the difference between the end of the previous
            # sample and the end of the current sample.
            wall_time_start = datetime.datetime.utcnow()
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
    return self.cache.path / "samples" / sampler.hash

  def _WriteMetafile(self) -> None:
    pbutil.ToFile(self.meta, pathlib.Path(self.cache.keypath("META.pbtxt")))

  def InferenceManifest(self) -> typing.List[pathlib.Path]:
    """Return the list of files which are required for model inference.

    Returns:
      A list of absolute paths.
    """
    return sorted(
      [self.cache.path / "tokenizer", self.cache.path / "META.pbtxt",]
      + self.backend.InferenceManifest()
    )

  @property
  def tokenizer(self) -> tokenizers.TokenizerBase:
    return self.corpus.tokenizer

  @property
  def is_trained(self) -> bool:
    return self.backend.is_trained

  def __repr__(self) -> str:
    """String representation."""
    return f"model[{self.hash}]"

  def __eq__(self, rhs) -> bool:
    if not isinstance(rhs, Model):
      return False
    return rhs.hash == self.hash

  def __ne__(self, rhs) -> bool:
    return not self.__eq__(rhs)
