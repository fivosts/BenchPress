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

import numpy as np

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

def LogBatchTelemetry(
  batch: DataBatch, steps_per_epoch: int, num_epochs: int
) -> None:
  """Log analytics about the batch."""
  l.getLogger().debug("deeplearning.clgen.models.data_generators.LogBatchTelemetry()")
  l.getLogger().info("Step shape: X: {}, y" ": {}.".format(batch.X.shape, batch.y.shape))
  # sys.getsizeof() includes only the memory required for an object, not any
  # objects it refernces, so we must manually sum the X and y arrays.
  batch_size = sys.getsizeof(batch) + batch.X.nbytes + batch.y.nbytes
  l.getLogger().info(
    "Memory: {} per batch, {} per epoch, {} total.".format(
            humanize.BinaryPrefix(batch_size, "B"),
            humanize.BinaryPrefix(batch_size * steps_per_epoch, "B"),
            humanize.BinaryPrefix(batch_size * steps_per_epoch * num_epochs, "B"),
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

    # generate a kernel corpus
    self.i = 0
    if (
      self.encoded_corpus is None
      or self.training_opts.shuffle_corpus_contentfiles_between_epochs
    ):
      self.encoded_corpus = np.concatenate(self.corpus.GetTrainingData(
              shuffle=self.training_opts.shuffle_corpus_contentfiles_between_epochs
            ))

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
  def __init__(
    self, corpus: "corpuses.Corpus", training_opts: model_pb2.TrainingOptions
  ):
    l.getLogger().debug("deeplearning.clgen.models.data_generators.MaskLMBatchGenerator.__init__()")
    self.corpus = corpus
    self.training_opts = training_opts

    # Lazily instantiated.
    self.encoded_corpus = None
    self.num_batches = 0
    self.batches = None
    self.CreateBatches()

    LogBatchTelemetry(
      self.batches[0], self.num_batches, self.training_opts.num_epochs
    )
    return


  # // Maximum number of masked LM predictions per sequence.
  # optional int32 max_predictions_per_seq = 3;
  # // Number of times to duplicate the input data (with different masks).
  # optional int32 dupe_factor = 4;
  # // Masked LM probability.
  # optional float masked_lm_prob = 5;

  def CreateBatches(self) -> None:
    l.getLogger().debug("deeplearning.clgen.models.data_generators.MaskLMBatchGenerator.CreateBatches()")
    start_time = time.time()

    # generate a kernel corpus
    self.i = 0
    if (
      self.encoded_corpus is None
      or self.training_opts.shuffle_corpus_contentfiles_between_epochs
    ):
      self.encoded_corpus = self.corpus.GetTrainingData(
        shuffle=self.training_opts.shuffle_corpus_contentfiles_between_epochs
      )

    l.getLogger().warning(self.corpus.atomizer.DeatomizeIndices(self.encoded_corpus[:200]))
    l.getLogger().warning(self.encoded_corpus.shape)

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

    ## 1. dupe_factor:
    ## 2. masked_lm_prob: def createMasks and mask->keep labels etc.
    ## 3. max_predictions_per_seq: 

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
    exit()
    return

  def NextBatch(self) -> DataBatch:
    """Fetch next batch.

    Returns:
      X, Y DataBatch.
    """
    l.getLogger().debug("deeplearning.clgen.models.data_generators.MaskLMBatchGenerator.NextBatch()")
    batch = self.batches[self.i]
    self.i += 1
    assert 0 <= self.i <= self.num_batches
    return batch
