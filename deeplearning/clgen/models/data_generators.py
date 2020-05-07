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

class BatchGenerator(object):
  def __init__(self,
                corpus: "corpuses.Corpus",
                training_opts: model_pb2.TrainingOptions
              ):

  


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

    self._LogBatchTelemetry(
      self.batches[0], self.num_batches, self.training_opts.num_epochs
    )

  def CreateBatches(self) -> None:
    l.getLogger().debug("deeplearning.clgen.models.data_generators.TensorflowBatchGenerator.CreateBatches()")
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

  def _LogBatchTelemetry(
    self, batch: DataBatch, steps_per_epoch: int, num_epochs: int
  ) -> None:
    """Log analytics about the batch."""
    l.getLogger().debug("deeplearning.clgen.models.data_generators._LogBatchTelemetry()")
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
