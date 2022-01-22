"""This file defines the streaming generators for model training data.

We train models on overlapping one-hot encoded text sequences. For a corpus of
a reasonable size, the full training data may not fit in memory. This modules
provides Python Generator classes for use by a sequential Keras model's
fit_generator() method to stream batches of training data.
"""
import sys
import time
import typing
import copy
import humanize

import numpy as np

from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen.util import logging as l

class DataBatch(typing.NamedTuple):
  """An <X,y> data tuple used for training one batch."""
  X: np.array
  y: np.array

  @property
  def sizeof_batch(self):
    return sys.getsizeof(self) + self.X.nbytes + self.y.nbytes
  
  def LogBatchTelemetry(self,
                        steps_per_epoch: int,
                        num_epochs: int,
                        ) -> None:

    """Log analytics about the batch."""
    l.logger().info("Step shape: X: {}, y" ": {}.".format(self.X.shape, self.y.shape))
    l.logger().info(
      "Memory: {} per batch, {} per epoch, {} total.".format(
              humanize.naturalsize(self.sizeof_batch, binary = True),
              humanize.naturalsize(self.sizeof_batch * steps_per_epoch, binary = True),
              humanize.naturalsize(self.sizeof_batch * steps_per_epoch * num_epochs, binary = True),
          )
    )
    return

class TensorflowBatchGenerator(object):
  def __init__(
    self, corpus: "corpuses.Corpus", training_opts: model_pb2.TrainingOptions
  ):
    self.corpus = corpus
    self.training_opts = training_opts

    # Lazily instantiated.
    self.original_encoded_corpus = None
    self.encoded_corpus = None
    self.num_batches = 0
    self.batches = None
    self.CreateBatches()

    self.batches[0].LogBatchTelemetry(self.num_batches, self.training_opts.num_epochs)
    return

  def CreateBatches(self) -> None:
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
      raise ValueError(
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
    l.logger().info(
      "Encoded corpus of {} tokens (clipped last {} tokens) in {} ms.".format(
                humanize.intcomma(clipped_corpus_length),
                humanize.intcomma(len(self.encoded_corpus) - clipped_corpus_length),
                humanize.intcomma(int((time.time() - start_time) * 1000)),
            )
    )
    return

  def NextBatch(self) -> DataBatch:
    """Fetch next batch.

    Returns:
      X, Y DataBatch.
    """
    batch = self.batches[self.i]
    self.i += 1
    assert 0 <= self.i <= self.num_batches
    return batch
