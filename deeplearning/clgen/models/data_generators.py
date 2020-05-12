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
"""This file defines the streaming generators for model training data.

We train models on overlapping one-hot encoded text sequences. For a corpus of
a reasonable size, the full training data may not fit in memory. This modules
provides Python Generator classes for use by a sequential Keras model's
fit_generator() method to stream batches of training data.
"""
import sys
import time
import typing
import random
import progressbar
import collections

import numpy as np

from deeplearning.clgen import cache
from deeplearning.clgen import errors
from deeplearning.clgen.proto import model_pb2
from labm8.py import app
from labm8.py import humanize

from eupy.native import logger as l

FLAGS = app.FLAGS


class DataBatch(typing.NamedTuple):
  """An <X,y> data tuple used for training one batch."""

  X: np.array
  y: np.array

class MaskBatch(typing.NamedTuple):
  input_ids             : np.array
  masked_lm_positions   : np.array
  masked_lm_ids         : np.array

def LogBatchTelemetry(
  batch, steps_per_epoch: int, num_epochs: int
) -> None:
  """Log analytics about the batch."""
  l.getLogger().debug("deeplearning.clgen.models.data_generators.LogBatchTelemetry()")
  sizeof_batch = 0
  if isinstance(batch, DataBatch):
    l.getLogger().info("Step shape: X: {}, y" ": {}.".format(batch.X.shape, batch.y.shape))
    # sys.getsizeof() includes only the memory required for an object, not any
    # objects it refernces, so we must manually sum the X and y arrays.
    sizeof_batch = sys.getsizeof(batch) + batch.X.nbytes + batch.y.nbytes
  elif isinstance(batch, MaskBatch):
    l.getLogger().info("Step shape: Input_ids: {}, masked_lm_positions: {}, masked_lm_ids: {}"
                        .format(
                                batch.input_ids.shape, 
                                batch.masked_lm_positions.shape, 
                                batch.masked_lm_ids.shape))
    sizeof_batch = (sys.getsizeof(batch) +
                    batch.input_ids.nbytes +
                    batch.masked_lm_positions.nbytes +
                    batch.masked_lm_ids.nbytes
                  )
  else:
    raise errors.UserError("Unrecognized Data Batch type: {}".format(type(batch)))
  l.getLogger().info(
    "Memory: {} per batch, {} per epoch, {} total.".format(
            humanize.BinaryPrefix(sizeof_batch, "B"),
            humanize.BinaryPrefix(sizeof_batch * steps_per_epoch, "B"),
            humanize.BinaryPrefix(sizeof_batch * steps_per_epoch * num_epochs, "B"),
        )
  )

class KerasBatchGenerator():

  def AutoGenerator(
    self, corpus: "corpuses.Corpus", training_opts: model_pb2.TrainingOptions
  ) -> typing.Generator[DataBatch, typing.Any, None]:
    """Determine and construct what we believe to be the best data generator.

    The optimum generator will depend on the corpus, the amount of memory
    available, and the vocabulary encoding.

    Args:
      corpus: A Corpus instance.
      training_opts: A TrainingOptions proto.

    Returns:
      A generator suitable for use by a model's fit_generator() method.
    """
    l.getLogger().debug("deeplearning.clgen.models.data_generators.KerasBatchGenerator.AutoGenerator()")
    return self.BatchGenerator(corpus, training_opts)

  def BatchGenerator(
    self, corpus: "corpuses.Corpus", training_opts: model_pb2.TrainingOptions
  ) -> typing.Generator[DataBatch, typing.Any, None]:
    """A batch generator which lazily one-hot encodes the y vectors.

    This reduces the memory overhead by only one-hot encoding the y vectors on a
    per-batch basis. This is of course slower than one-hot encoding the entire
    y corpus, but that requires more memory than is available on many systems for
    a reasonable corpus.

    Args:
      corpus: A Corpus instance.
      training_opts: A TrainingOptions proto.

    Returns:
      A generator suitable for use by a model's fit_generator() method.
    """
    l.getLogger().debug("deeplearning.clgen.models.data_generators.KerasBatchGenerator.BatchGenerator()")
    x, y, steps_per_epoch = self.GetTrainingCorpus(corpus, training_opts)

    # Per-epoch outer loop.
    epoch_num = 0
    while True:
      # Re-shuffle corpus if needed.
      if epoch_num and training_opts.shuffle_corpus_contentfiles_between_epochs:
        x, y, steps_per_epoch = self.GetTrainingCorpus(corpus, training_opts)

      # Roll so that we don't need to reset model states over epochs.
      x_epoch = np.split(np.roll(x, -epoch_num, axis=0), steps_per_epoch, axis=1)
      y_epoch = np.split(np.roll(y, -epoch_num, axis=0), steps_per_epoch, axis=1)
      # Per-batch inner loop.
      for batch_num in range(steps_per_epoch):
        batch = DataBatch(
          X=x_epoch[batch_num],
          # Lazy one-hot encoding.
          y=self.OneHotEncode(y_epoch[batch_num], corpus.vocab_size),
        )
        if not batch_num and not epoch_num:
          LogBatchTelemetry(batch, steps_per_epoch, training_opts.num_epochs)
        yield batch
      epoch_num += 1
    return

  def GetTrainingCorpus(
    self, corpus: "corpuses.Corpus", training_opts: model_pb2.TrainingOptions
  ) -> typing.Tuple[np.ndarray, np.ndarray, int]:
    """Get the corpus to train over.

    Args:
      corpus: A Corpus instance.
      training_opts: A TrainingOptions proto.

    Returns:
      An X, y pair of data for an epoch, and the number of steps in the epoch.

    Raises:
      UserError: If batch_size and sequence_length are too large for the corpus,
        yielding no batches.
    """
    l.getLogger().debug("deeplearning.clgen.models.data_generators.KerasBatchGenerator.GetTrainingCorpus()")
    start_time = time.time()
    encoded_corpus = np.concatenate(corpus.GetTrainingData(
          shuffle=training_opts.shuffle_corpus_contentfiles_between_epochs
        ))
    corpus_length = len(encoded_corpus)
    steps_per_epoch = (corpus_length - 1) // (
      training_opts.batch_size * training_opts.sequence_length
    )
    if not steps_per_epoch:
      raise errors.UserError(
        f"Requested batch size ({training_opts.batch_size}) and "
        f"sequence length ({training_opts.sequence_length}) are too large for "
        f"corpus of size {corpus_length}."
      )

    clipped_corpus_length = (
      steps_per_epoch * training_opts.batch_size * training_opts.sequence_length
    )

    x = np.reshape(
      encoded_corpus[:clipped_corpus_length],
      [training_opts.batch_size, steps_per_epoch * training_opts.sequence_length],
    )
    y = np.reshape(
      encoded_corpus[1 : clipped_corpus_length + 1],
      [training_opts.batch_size, steps_per_epoch * training_opts.sequence_length],
    )

    l.getLogger().info(
      "Encoded corpus of {} tokens (clipped last {} tokens) in {} ms.".format(
              humanize.Commas(clipped_corpus_length),
              humanize.Commas(corpus_length - clipped_corpus_length),
              humanize.Commas(int((time.time() - start_time) * 1000)),
          )
    )
    return x, y, steps_per_epoch


  def OneHotEncode(self, indices: np.ndarray, vocabulary_size: int):
    """One-hot encode an array of vocabulary indices.

      Args:
        indices: A 1D array of vocabulary indices.
        vocabulary_size: The size of the vocabulary.

      Returns:
        A 2D array of one-hot encoded tokens.
      """
    l.getLogger().debug("deeplearning.clgen.models.data_generators.KerasBatchGenerator.OneHotEncode()")
    return np.eye(vocabulary_size)[indices]


class TensorflowBatchGenerator(object):
  def __init__(
    self, corpus: "corpuses.Corpus", training_opts: model_pb2.TrainingOptions
  ):
    l.getLogger().debug("deeplearning.clgen.models.data_generators.TensorflowBatchGenerator.__init__()")
    self.corpus = corpus
    self.training_opts = training_opts

    # Lazily instantiated.
    self.original_encoded_corpus = None
    self.encoded_corpus = None
    self.num_batches = 0
    self.batches = None
    self.CreateBatches()

    LogBatchTelemetry(
      self.batches[0], self.num_batches, self.training_opts.num_epochs
    )
    return

  def CreateBatches(self) -> None:
    l.getLogger().debug("deeplearning.clgen.models.data_generators.TensorflowBatchGenerator.CreateBatches()")
    start_time = time.time()

    self.i = 0
    if self.original_encoded_corpus is None:
      self.original_encoded_corpus = self.corpus.GetTrainingData(
          shuffle=self.training_opts.shuffle_corpus_contentfiles_between_epochs
        )

    self.encoded_corpus = np.concatenate(self.original_encoded_corpus)
    batch_size = self.training_opts.batch_size
    sequence_length = self.training_opts.sequence_length

    # set corpus size and number of batches
    self.num_batches = int(
      len(self.encoded_corpus) / (batch_size * sequence_length)
    )
    if self.num_batches == 0:
      raise errors.UserError(
        "Not enough data. Use a smaller sequence_length and batch_size"
      )

    # split into batches
    clipped_corpus_length = self.num_batches * batch_size * sequence_length
    clipped_corpus = self.encoded_corpus[:clipped_corpus_length]
    xdata = clipped_corpus
    ydata = np.copy(clipped_corpus)

    # Wrap-around.
    ydata[:-1] = xdata[1:]
    ydata[-1] = xdata[0]
    self.batches = [
      DataBatch(x, y)
      for x, y in zip(
        np.split(xdata.reshape(batch_size, -1), self.num_batches, 1),
        np.split(ydata.reshape(batch_size, -1), self.num_batches, 1),
      )
    ]
    l.getLogger().info(
      "Encoded corpus of {} tokens (clipped last {} tokens) in {} ms.".format(
                humanize.Commas(clipped_corpus_length),
                humanize.Commas(len(self.encoded_corpus) - clipped_corpus_length),
                humanize.Commas(int((time.time() - start_time) * 1000)),
            )
    )
    return

  def NextBatch(self) -> DataBatch:
    """Fetch next batch.

    Returns:
      X, Y DataBatch.
    """
    l.getLogger().debug("deeplearning.clgen.models.data_generators.TensorflowBatchGenerator.NextBatch()")
    batch = self.batches[self.i]
    self.i += 1
    assert 0 <= self.i <= self.num_batches
    return batch

class MaskLMBatchGenerator(object):
  def __init__(self,
               corpus: "corpuses.Corpus",
               training_opts: model_pb2.TrainingOptions,
               cache_path,
               tf, 
  ):
    l.getLogger().debug("deeplearning.clgen.models.data_generators.MaskLMBatchGenerator.__init__()")
    self.corpus = corpus
    self.training_opts = training_opts
    self.tfRecord = cache_path / "dataset" / "maskedDataset.tf_record"
    self.tf = tf

    # Lazily instantiated.
    self._masked_corpus = None
    self._original_encoded_corpus = None
    self._encoded_corpus = None
    self.num_batches = 0
    self.sequence_length = self.training_opts.sequence_length

    if self.training_opts.random_seed:
      self.rngen = random.Random(training_opts.random_seed)
    else:
      self.rngen = random.Random()

    if not self.tfRecord.exists():
      self.CreateCorpus()
    else:
      pass ## TODO ?
    return

  def generateTfDataset(self,
                      max_seq_length,
                      is_training,
                      num_cpu_threads,
                      ) -> "tf.Dataset":

    def _decode_record(record, name_to_features):
      """Decodes a record to a TensorFlow example."""
      ## This function assumes record is still a file (expressed as TF dataset)
      ## It decodes this record to tf scalars.
      ## You already have them so this will be skipped
      example = self.tf.io.parse_single_example(record, name_to_features)

      # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
      # So cast all int64 to int32.
      for name in list(example.keys()):
        t = example[name]
        if t.dtype == self.tf.int64:
          t = self.tf.to_int32(t)
        example[name] = t

      return example

    def input_fn(params):

      batch_size = params["batch_size"]
      name_to_features = {
          "input_ids":
              self.tf.FixedLenFeature([max_seq_length], tf.int64),
          "masked_lm_positions":
              self.tf.FixedLenFeature([self.training_opts.max_predictions_per_seq], tf.int64),
          "masked_lm_ids":
              self.tf.FixedLenFeature([self.training_opts.max_predictions_per_seq], tf.int64),
      }

      # For training, we want a lot of parallel reading and shuffling.
      # For eval, we want no shuffling and parallel reading doesn't matter.
      if is_training:
        d = tf.data.Dataset.from_tensor_slices(tf.constant("""Insert here the tfRecord file"""))
        d = d.repeat()
        d = d.shuffle(buffer_size=len("""The input file"""))

        # `cycle_length` is the number of parallel files that get read.
        cycle_length = min(num_cpu_threads, len("""The input file"""))

        # `sloppy` mode means that the interleaving is not exact. This adds
        # even more randomness to the training pipeline.
        d = d.apply(
            tf.data.experimental.parallel_interleave(
                tf.data.TFRecordDataset,
                sloppy=is_training,
                cycle_length=cycle_length))
        d = d.shuffle(buffer_size=100)
      else:
        d = tf.data.TFRecordDataset("""The input file""")
        # Since we evaluate for a fixed number of steps we don't want to encounter
        # out-of-range exceptions.
        d = d.repeat()

      # We must `drop_remainder` on training because the TPU requires fixed
      # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
      # and we *don't* want to drop the remainder, otherwise we wont cover
      # every sample.
      d = d.apply(
          tf.data.experimental.map_and_batch(
              lambda record: _decode_record(record, name_to_features),
              batch_size=batch_size,
              num_parallel_batches=num_cpu_threads,
              drop_remainder=True))
      return d

    return input_fn


  def CreateCorpus(self) -> None:
    l.getLogger().debug("deeplearning.clgen.models.data_generators.MaskLMBatchGenerator.CreateBatches()")
    start_time = time.time()

    # generate a kernel corpus
    self._original_encoded_corpus = self.corpus.GetTrainingData(
      shuffle=self.training_opts.shuffle_corpus_contentfiles_between_epochs
    )

    self._encoded_corpus = np.concatenate(self._original_encoded_corpus)
    batch_size = self.training_opts.batch_size

    # set corpus size and number of batches
    self.num_batches = int(
      len(self._encoded_corpus) / (batch_size * self.sequence_length)
    )
    if self.num_batches == 0:
      raise errors.UserError(
        "Not enough data. Use a smaller sequence_length and batch_size"
      )
    # split into batches
    clipped_corpus_length = self.num_batches * batch_size * self.sequence_length
    clipped_corpus = self._encoded_corpus[:clipped_corpus_length]

    shaped_corpus = np.split(clipped_corpus.reshape(batch_size, -1), self.num_batches, 1)
    
    self._masked_corpus = self.MaskCorpus(shaped_corpus)
    self.saveMaskedCorpus()
    
    self.num_batches = self.num_batches * int(self.training_opts.dupe_factor)

    l.getLogger().info(
      "Masked corpus of {} tokens (clipped last {} tokens) in {} ms.".format(
                humanize.Commas(clipped_corpus_length),
                humanize.Commas(len(self._encoded_corpus) - clipped_corpus_length),
                humanize.Commas(int((time.time() - start_time) * 1000)),
            )
    )
    return

  def MaskCorpus(self, 
                 corpus: np.array
                )-> list:
    l.getLogger().debug("deeplearning.clgen.models.data_generators.MaskLMBatchGenerator.MaskCorpus()")
    l.getLogger().warn("Masking Corpus is a slow process. Assign multiple threads to it")

    masked_corpus = []
    flattened_corpus = []
    for _ in range(self.training_opts.dupe_factor): # This enables multiprocessing
      flattened_corpus.extend(corpus)

    with progressbar.ProgressBar(max_value = len(flattened_corpus)) as bar:
        for idx, batch in enumerate(flattened_corpus):
          masked_corpus.extend(self.maskBatch(batch))
          bar.update(idx)
    return masked_corpus

  def maskBatch(self, batch):
    # training_batch = {
    #                   'input_ids'           : [], 
    #                   'masked_lm_positions' : [],
    #                   'masked_lm_ids'       : [],
    #                   }

    out_batch = []
    for seq in batch:
      x, ypos, ytok = self.maskSequence(seq)
      out_batch.append(MaskBatch(np.asarray(x), np.asarray(ypos), np.asarray(ytok)))
      # training_batch['input_ids'].append(x)
      # training_batch['masked_lm_positions'].append(ypos)
      # training_batch['masked_lm_ids'].append(ytok)

    # batch = MaskBatch(
    #             np.asarray(training_batch['input_ids']),
    #             np.asarray(training_batch['masked_lm_positions']),
    #             np.asarray(training_batch['masked_lm_ids'])
    #           )
                 
    return out_batch

  def maskSequence(self,
                   seq: np.array,
                  ) -> DataBatch:
    l.getLogger().debug("deeplearning.clgen.models.data_generators.MaskLMBatchGenerator.maskSequence()")
    if seq.ndim != 1:
      raise ValueError("Input for masking is not a single dimensional array!")

    cand_indexes = np.arange(len(seq))
    self.rngen.shuffle(cand_indexes)

    masks_to_predict = min(self.training_opts.max_predictions_per_seq,
                            max(1, int(round(len(seq) * self.training_opts.masked_lm_prob))))

    output_tokens = np.copy(seq)
    masked_lms = []

    for pos_index in cand_indexes:
      if len(masked_lms) >= masks_to_predict:
        break

      # 80% of the time, replace with [MASK]
      if self.rngen.random() < 0.8:
        output_tokens[pos_index] = self.corpus.atomizer.maskToken
      # The else block below is debatable for this use case. So comment out for now
      # else:
      #   # 10% of the time, keep original
      #   if self.rngen.random() < 0.5:
      #     pass
      #   # 10% of the time, replace with random word
      #   else:
      #     output_tokens[pos_index] = self.rngen.randint(0, self.corpus.atomizer.vocab_size - 1)

      class MaskedLmInstance(typing.NamedTuple):
        pos_index: int
        token_id: int

      masked_lms.append(MaskedLmInstance(pos_index=pos_index, token_id=seq[pos_index]))

    assert len(masked_lms) <= masks_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x.pos_index)

    masked_lm_positions = []
    masked_lm_ids = []
    for p in masked_lms:
      masked_lm_positions.append(p.pos_index)
      masked_lm_ids.append(p.token_id)
    while len(masked_lm_positions) < self.training_opts.max_predictions_per_seq:
        masked_lm_positions.append(0)
        masked_lm_ids.append(0)

    return (output_tokens, masked_lm_positions, masked_lm_ids)

  def saveMaskedCorpus(self) -> None:
    l.getLogger().debug("deeplearning.clgen.models.data_generators.MaskLMBatchGenerator.saveMaskedCorpus()")
     
    self.tfRecord.parent.mkdir(exist_ok = True, parents = True)
    writer = self.tf.io.TFRecordWriter(str(self.tfRecord))

    total_written = 0
    for (inst_index, instance) in enumerate(self._masked_corpus):
      input_ids = instance.input_ids

      assert len(input_ids) == self.sequence_length

      masked_lm_positions = instance.masked_lm_positions
      masked_lm_ids = instance.masked_lm_ids

      features = collections.OrderedDict()
      features["input_ids"] = self.tf.train.Feature(
                                            int64_list = self.tf.train.Int64List(
                                                                value = list(input_ids)
                                                                )
                                            )
      features["masked_lm_positions"] = self.tf.train.Feature(
                                            int64_list = self.tf.train.Int64List(
                                                                value = list(masked_lm_positions)
                                                                )
                                            )
      features["masked_lm_ids"] = self.tf.train.Feature(
                                            int64_list = self.tf.train.Int64List(
                                                                value = list(masked_lm_ids)
                                                                )
                                            )

      tf_example = self.tf.train.Example(features=self.tf.train.Features(feature=features))
      writer.write(tf_example.SerializeToString())
      total_written += 1

    writer.close()
    ## TODO print num_batches x batch_size (a.k.a. 185 * 64) instead of 11840
    l.getLogger().info("Wrote {} instances to {}".format(total_written, self.tfRecord))
    return