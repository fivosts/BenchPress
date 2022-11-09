# coding=utf-8
# Copyright 2022 Foivos Tsimpourlas and Chris Cummins.
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
"""The BenchPress language model."""
import os
import time
import shutil
import pathlib
import typing
import datetime
import humanize

import numpy as np

from deeplearning.benchpress.samplers import sample_observers as sample_observers_lib
from deeplearning.benchpress.util import pbutil
from deeplearning.benchpress.util import cache
from deeplearning.benchpress.util import crypto
from deeplearning.benchpress.util import commit
from deeplearning.benchpress.util import environment
from deeplearning.benchpress.util import distrib
from deeplearning.benchpress.features import extractor
from deeplearning.benchpress.features import hidden_state
from deeplearning.benchpress.corpuses import tokenizers
from deeplearning.benchpress.corpuses import corpuses
from deeplearning.benchpress.models import builders
from deeplearning.benchpress.models.keras_sequential import keras_sequential
from deeplearning.benchpress.models.tf_sequential import tf_sequential
from deeplearning.benchpress.models.tf_bert import tf_bert
from deeplearning.benchpress.models.torch_bert import torch_bert
from deeplearning.benchpress.models.incoder import incoder
from deeplearning.benchpress.proto import internal_pb2
from deeplearning.benchpress.proto import model_pb2
from deeplearning.benchpress.preprocessors import opencl
from absl import flags

from deeplearning.benchpress.util import logging as l

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

flags.DEFINE_integer(
  "sample_workload_size",
  2048,
  "Select size of workload samples for single sample step, per node."
)

class Model(object):
  """A BenchPress language model.

  Please note model instances should be treated as immutable. Upon
  instantiation, a model's properties are used to determine its hash. If you
  modify a property after instantiation, the hash will be out of date, which
  can lead to bad things happening.
  """
  @property
  def tokenizer(self) -> tokenizers.TokenizerBase:
    return self.corpus.tokenizer

  @property
  def is_trained(self) -> bool:
    return self.backend.is_trained

  @property
  def hidden_state_size(self) -> int:
    return self.backend.hidden_state_size

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

    # Initialize corpuses
    self.corpus           = corpuses.Corpus(config.corpus)
    self.pre_train_corpus = None
    if config.HasField("pre_train_corpus"):
      self.pre_train_corpus = corpuses.Corpus(config.pre_train_corpus)

    self.hash = self._ComputeHash(self.pre_train_corpus, self.corpus, self.config)
    self._created = False

    if environment.WORLD_RANK == 0:
      self.cache = cache.mkcache("model", self.hash)
      self.cache.path.mkdir(exist_ok = True, parents = True)
    else:
      while not cache.cachepath("model", self.hash).exists():
        time.sleep(0.5)
      self.cache = cache.mkcache("model", self.hash)

    if environment.WORLD_RANK == 0:
      # Create the necessary cache directories.
      (self.cache.path / "checkpoints").mkdir(exist_ok=True)
      (self.cache.path / "samples").mkdir(exist_ok=True)
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

      # Create symlink to the tokenizer and create a backup inside checkpoints.
      symlink = self.cache.path / "tokenizer"
      if not symlink.is_symlink():
        os.symlink(
          os.path.relpath(self.corpus.tokenizer_path, self.cache.path), symlink
        )
      if (self.cache.path / "checkpoints" / "backup_tokenizer.pkl").exists():
        shutil.copyfile(self.cache.path / "checkpoints" / "backup_tokenizer.pkl", self.corpus.tokenizer_path)

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
        if config_to_compare.HasField("pre_train_corpus"):
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
        if cached_to_compare.HasField("pre_train_corpus"):
          cached_to_compare.training.ClearField("num_pretrain_steps")
        cached_to_compare.training.ClearField("batch_size")
        if cached_to_compare.training.HasField("data_generator"):
          cached_to_compare.training.data_generator.ClearField("steps_per_epoch")
          cached_to_compare.training.data_generator.ClearField("validation_set")
        if cached_to_compare.training.sequence_length != config_to_compare.training.sequence_length:
          l.logger().warning("Mismatch between pre-trained and current config sequence_length!\
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
      model_pb2.NetworkArchitecture.INCODER_1B: incoder.Incoder1B,
      model_pb2.NetworkArchitecture.INCODER_6B: incoder.Incoder6B,
    }[config.architecture.backend](self.config, self.cache, self.hash)
    l.logger().info("Initialized {} in {}".format(self.backend, self.cache.path))
    return

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
    if FLAGS.custom_incoder_ckpt is not None:
      hash_list.append(FLAGS.custom_incoder_ckpt)
    return crypto.sha1_list(hash_list)

  def Create(self) -> bool:
    if self._created:
      return False
    self._created = True
    self.corpus.Create()
    if self.pre_train_corpus:
      self.pre_train_corpus.Create(self.corpus.tokenizer)

    if not (self.cache.path / "checkpoints" / "backup_tokenizer.pkl").exists():
      shutil.copyfile(self.corpus.tokenizer_path, self.cache.path / "checkpoints" / "backup_tokenizer.pkl")

    self.backend.Create(tokenizer = self.corpus.tokenizer)
    return

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

    l.logger().info(
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
    l.logger().info(
      "Trained model for {} {} in {} ms. " "Training loss: {}."
        .format(
          telemetry_logs[-1].epoch_num if FLAGS.select_checkpoint_step == -1 else telemetry_logs[FLAGS.select_checkpoint_step-1].epoch_num,
          "steps" if isinstance(self.backend, tf_bert.tfBert) or isinstance(self.backend, torch_bert.torchBert) else "epochs",
          humanize.intcomma(sum(t.epoch_wall_time_ms for t in telemetry_logs)),
          telemetry_logs[-1].loss if FLAGS.select_checkpoint_step == -1 else telemetry_logs[FLAGS.select_checkpoint_step-1].loss,
          )
    )
    hidden_state.setup_lm(self.backend)
    return self

  def Sample(
    self,
    sampler: 'samplers.Sampler',
    sample_observers: typing.List[sample_observers_lib.SampleObserver],
    seed: int = None,
    num_batches: int = None,
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
    sampler.Create()

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

    if isinstance(self.backend, tf_bert.tfBert) or isinstance(self.backend, torch_bert.torchBert) or isinstance(self.backend, incoder.Incoder):
      sample_batch = lambda : self._SampleLMBatch(sampler, tokenizer, sample_observers, epoch)
    elif isinstance(self.backend, tf_sequential.tfSequential) or isinstance(self.backend, keras_sequential.kerasSequential):
      sample_batch = lambda : self._SampleSeqBatch(sampler, tokenizer, sample_observers, epoch)
    else:
      raise ValueError("Unrecognized backend.")

    try:
      seq_count, cont, compiled = 0, True, 0
      nb = 0
      while cont:
        if num_batches and nb >= num_batches:
          break
        nb+=1
        cont, s, c = sample_batch()
        seq_count += s
        compiled += c
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

    if environment.WORLD_RANK == 0:
      for obs in sample_observers:
        obs.endSample()
    if isinstance(self.backend, torch_bert.torchBert) and sampler.is_active:
      self.backend.sample.data_generator.samples_cache_obs.endSample()

    time_now = datetime.datetime.utcnow()
    l.logger().info( "Produced {} samples at a rate of {} ms / sample. Session's compilation rate was {}%"
                        .format(
                          humanize.intcomma(seq_count),
                          humanize.intcomma(int(1000 * ((time_now - sample_start_time) / max(seq_count, 1)).total_seconds())),
                          round(100 * ((compiled / seq_count if seq_count > 0 else 0)), 3),
                        )
    )
    return

  def _SampleLMBatch(self,
                     sampler: 'samplers.Sampler',
                     tokenizer: tokenizers.TokenizerBase,
                     sample_observers: typing.List[sample_observers_lib.SampleObserver],
                     epoch: int,
                     ) -> bool:
    """
    Run a sampling iteration over BERT models.
    """
    start_time = datetime.datetime.utcnow()
    seq_count  = 0
    compiled   = 0
    self.backend.InitSampleBatch(sampler, workload_size = FLAGS.sample_workload_size // environment.WORLD_SIZE)
    try:
      org_inputs, input_ids, samples, indices = self.backend.SampleNextIndices(sampler)
    except StopIteration:
      return False, seq_count, compiled

    if not samples:
      # Return empty means model has not produced something that can be stored.
      # This 'if' accommodates active sampling, which is very selective.
      return True, seq_count, compiled

    continue_sampling = True

    if environment.WORLD_RANK == 0:
      assert len(org_inputs) == len(input_ids) == len(samples) == len(indices), "Length mismatch, {}-{}-{}-{}".format(len(org_inputs), len(input_ids), len(samples), len(indices))
      for org, inp, sample, idxs in zip(org_inputs, input_ids, samples, indices):

        src = self.tokenizer.ArrayToCode(sample, with_formatting = True)
        try:
          stdout = opencl.Compile(src)
          compile_flag = True
          compiled += 1
          features = extractor.ExtractRawFeatures(src)
        except ValueError:
          compile_flag = False
          features     = ""

        end_time = datetime.datetime.utcnow()
        sample = model_pb2.Sample(
          train_step                = epoch,
          text                      = src,
          sample_indices            = ','.join([self.tokenizer.decoder[idx].replace('\n', '\\n') for idx in idxs]).replace('\n', '\\n'),
          encoded_sample_indices    = ','.join([str(idx) for idx in idxs]),
          original_input            = self.tokenizer.tokensToString(org, with_formatting = False, ignore_token = self.tokenizer.padToken),
          sample_feed               = self.tokenizer.tokensToString(inp, with_formatting = False, ignore_token = self.tokenizer.padToken),
          encoded_text              = ",".join([str(x) for x in sample]),
          sample_start_epoch_ms_utc = int(start_time.strftime("%s%f")),
          sample_time_ms            = int(round(1000 * ((end_time - start_time) / len(samples)).total_seconds())),
          wall_time_ms              = int(round(1000 * ((end_time - start_time) / len(samples)).total_seconds())),
          feature_vector            = features,
          num_tokens                = np.where(sample == self.tokenizer.padToken)[0][0] if self.tokenizer.padToken in sample else len(sample),
          compile_status            = compile_flag,
          categorical_sampling      = self.backend.samplesWithCategorical(),
          date_added                = datetime.datetime.utcnow().strftime("%m/%d/%Y, %H:%M:%S"),
        )
        # Notify sample observers.
        continue_sampling &= all(
          [obs.OnSample(sample) for obs in sample_observers]
        )
        seq_count += 1
      if environment.WORLD_SIZE > 1:
        _ = distrib.broadcast(str(continue_sampling))
    else:
      status = distrib.broadcast()
      if status == "True":
        continue_sampling = True        
      elif status == "False":
        continue_sampling = False
      else:
        raise OSError("Broken distributed message: '{}'".format(status))
    return continue_sampling, seq_count, compiled

  def _SampleSeqBatch(
    self,
    sampler: 'samplers.Sampler',
    tokenizer: tokenizers.TokenizerBase,
    sample_observers: typing.List[sample_observers_lib.SampleObserver],
    epoch: int,
  ) -> bool:
    """
    Run a single iteration of the batched sample inner-loop for sequential models.
    """

    start_time = datetime.datetime.utcnow()

    self.backend.InitSampleBatch(sampler)
    samples_in_progress = [
      sampler.tokenized_start_text.copy() for _ in range(sampler.batch_size)
    ]
    done = np.zeros(sampler.batch_size, dtype=np.bool)
    wall_time_start = start_time
    seq_count  = 0
    compiled   = 0

    # The return value of this method. If any of the sample_observers return
    # False, this value is set to False.
    continue_sampling = True

    # Sampling loop. Continues until all samples in the batch are done.
    while not done.all():
      indices, _ = self.backend.SampleNextIndices(sampler, done)
      # Iterate over all samples in batch to determine whether they're
      # done.

      for i in range(len(indices)):
        if done[i]:
          continue

        for index in indices[i]:
          samples_in_progress[i].append(tokenizer.decoder[index])

          if sampler.SampleIsComplete(samples_in_progress[i]):
            end_time       = datetime.datetime.utcnow()
            sample_kernel  = [x for x in samples_in_progress[i]]
            features       = extractor.ExtractRawFeatures(''.join(samples_in_progress[i]))
            done[i]        = 1
            try:
              stdout = opencl.Compile(''.join(samples_in_progress[i]))
              compile_flag = True
              compiled += 1
            except ValueError:
              compile_flag = False

            sample = model_pb2.Sample(
              train_step                = epoch,
              text                      = ''.join(samples_in_progress[i]),
              sample_indices            = "",
              encoded_sample_indices    = "",
              sample_feed               = sampler.start_text,
              encoded_text              = ",".join([str(tokenizer.vocab[x]) for x in sample_kernel]),
              sample_start_epoch_ms_utc = int(start_time.strftime("%s%f")),
              sample_time_ms            = int(round(1000 * ((end_time - start_time) / sampler.batch_size).total_seconds())),
              wall_time_ms              = int(round(1000 * ((end_time - start_time) / sampler.batch_size).total_seconds())),
              feature_vector            = features,
              num_tokens                = len(samples_in_progress[i]),
              compile_status            = compile_flag,
              categorical_sampling      = self.backend.samplesWithCategorical(),
              date_added                = datetime.datetime.utcnow().strftime("%m/%d/%Y, %H:%M:%S"),
            )
            # Notify sample observers.
            continue_sampling &= all(
              [obs.OnSample(sample) for obs in sample_observers]
            )
            if sampler.is_live and self.backend.feature_encoder:
              print(sample.feature_vector)
            seq_count += 1
            # Wall sample time is the difference between the end of the previous
            # sample and the end of the current sample.
            wall_time_start = datetime.datetime.utcnow()
            break
    return continue_sampling, seq_count, compiled

  def SamplerCache(self, sampler: 'samplers.Sampler') -> pathlib.Path:
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

  def __repr__(self) -> str:
    """String representation."""
    return f"model[{self.hash}]"

  def __eq__(self, rhs) -> bool:
    if not isinstance(rhs, Model):
      return False
    return rhs.hash == self.hash

  def __ne__(self, rhs) -> bool:
    return not self.__eq__(rhs)
