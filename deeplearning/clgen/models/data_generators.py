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
import copy
import humanize

import numpy as np

from deeplearning.clgen import cache
from deeplearning.clgen import pbutil
from deeplearning.clgen.tf import tf
from deeplearning.clgen.proto import model_pb2
from absl import flags
from eupy.native import logger as l

FLAGS = flags.FLAGS

flags.DEFINE_boolean(
  "write_text_dataset", 
  False, 
  "Set True for MaskLM data generator to write dataset in text format, along with the tfRecord."
)

flags.DEFINE_boolean(
  "force_remake_dataset",
  False,
  "Force data generator to re-mask encoded dataset and store tfRecord."
)

flags.DEFINE_integer(
  "steps_per_epoch",
  1000,
  "Set how many train steps consist an epoch. Checkpoints and Loss reports have once every epoch."
)

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
    l.getLogger().info("Step shape: X: {}, y" ": {}.".format(self.X.shape, self.y.shape))
    l.getLogger().info(
      "Memory: {} per batch, {} per epoch, {} total.".format(
              humanize.naturalsize(self.sizeof_batch, binary = True),
              humanize.naturalsize(self.sizeof_batch * steps_per_epoch, binary = True),
              humanize.naturalsize(self.sizeof_batch * steps_per_epoch * num_epochs, binary = True),
          )
    )
    return

class MaskSequence(typing.NamedTuple):
  """
  Tuple representation of a single MaskLM Instance. 
  This is not batch! generateTfDataset applies native batching,
  so this class represents a single instance!
  """

  input_ids            : np.array
  input_mask           : np.array
  masked_lm_positions  : np.array
  masked_lm_ids        : np.array
  masked_lm_weights    : np.array
  next_sentence_label  : np.int32

  @staticmethod
  def tfTypes():
    return (tf.int32, tf.int32, tf.int32, tf.int32, tf.float32, tf.int32)

  @staticmethod
  def npTypes():
    return (np.int32, np.int32, np.int32, np.int32, np.float32, np.int32)

  @staticmethod
  def tfShapes(batch_size, sequence_length, max_position_embeddings = None):
    return (tf.TensorShape([batch_size, sequence_length]),
            tf.TensorShape([batch_size, sequence_length]),
            tf.TensorShape([batch_size, max_position_embeddings]),
            tf.TensorShape([batch_size, max_position_embeddings]),
            tf.TensorShape([batch_size, max_position_embeddings]),
            tf.TensorShape([batch_size, 1]),
           )

  @property
  def sizeof_sequence(self):
    return (sys.getsizeof(self) + self.input_ids.nbytes + self.input_mask.nbytes +
           self.masked_lm_positions.nbytes + self.masked_lm_ids.nbytes +
           self.masked_lm_weights.nbytes + self.next_sentence_label.nbytes
           )

  def shapeSeqToBatch(self, inp, batch_size):
    return "(" + str(batch_size) + ", " + ", ".join([str(s) for s in inp.shape]) + ")"

  def LogBatchTelemetry(self,
                        batch_size: int,
                        steps_per_epoch: int,
                        num_epochs: int,
                        ) -> None:
    """Log analytics about the batch."""
    l.getLogger().info("Step shape: Input_ids: {}, input_mask: {}, masked_lm_positions: {}, masked_lm_ids: {}, masked_lm_weights: {}, next_sentence_label: {}"
                        .format(self.shapeSeqToBatch(self.input_ids,            batch_size),
                                self.shapeSeqToBatch(self.input_mask,           batch_size),
                                self.shapeSeqToBatch(self.masked_lm_positions,  batch_size),
                                self.shapeSeqToBatch(self.masked_lm_ids,        batch_size),
                                self.shapeSeqToBatch(self.masked_lm_weights,    batch_size),
                                self.shapeSeqToBatch(self.next_sentence_label,  batch_size),
                          )
                        )
    l.getLogger().info(
      "Memory: {} per batch, {} per epoch, {} total.".format(
              humanize.naturalsize(self.sizeof_sequence * batch_size, binary = True),
              humanize.naturalsize(self.sizeof_sequence * batch_size * steps_per_epoch, binary = True),
              humanize.naturalsize(self.sizeof_sequence * batch_size * steps_per_epoch * num_epochs, binary = True),
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
          batch.LogBatchTelemetry(steps_per_epoch, training_opts.num_epochs)
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
      raise ValueError(
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
              humanize.intcomma(clipped_corpus_length),
              humanize.intcomma(corpus_length - clipped_corpus_length),
              humanize.intcomma(int((time.time() - start_time) * 1000)),
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

    self.batches[0].LogBatchTelemetry(self.num_batches, self.training_opts.num_epochs)
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
    l.getLogger().info(
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
    l.getLogger().debug("deeplearning.clgen.models.data_generators.TensorflowBatchGenerator.NextBatch()")
    batch = self.batches[self.i]
    self.i += 1
    assert 0 <= self.i <= self.num_batches
    return batch

class MaskLMBatchGenerator(object):
  def __init__(self):
    l.getLogger().debug("deeplearning.clgen.models.data_generators.MaskLMBatchGenerator.__init__()")

    self.corpus                  = None
    self.atomizer                = None
    self.config                  = None
    self.cache                   = None
    self.shaped_corpus           = None
    self.masked_corpus           = None

    self.training_opts           = None
    self.steps_per_epoch         = None

    self.max_position_embeddings = None

    self.tfRecord                = None
    self.txtRecord               = None
    self.sampleBatch             = None

    self.sampler                 = None
    self.rngen                   = None
    return

  @classmethod
  def TrainMaskLMBatchGenerator(cls,
                               corpus: "corpuses.Corpus",
                               training_opts: model_pb2.TrainingOptions,
                               cache_path,
                               ) -> "data_generators.MaskLMBatchGenerator":
    """Initializes data generator for training."""
    d                     = MaskLMBatchGenerator()
    d.corpus              = corpus
    d.atomizer            = corpus.atomizer

    d.config              = training_opts.data_generator
    d.cache               = cache.mkcache(cache_path, "dataset")

    d.training_opts       = training_opts
    d.tfRecord            = d.cache.path / "Dataset.tf_record"
    d.txtRecord           = d.cache.path / "Dataset.txt"
    d.rngen               = random.Random(training_opts.random_seed)

    d.tfRecord.parent.mkdir(exist_ok = True, parents = True)
    d.CreateCorpus()
    if not d.tfRecord.exists() or FLAGS.force_remake_dataset:
      d._MaskCorpus(d.shaped_corpus)
      d._saveCorpusTfRecord()

    return d

  @classmethod
  def SampleMaskLMBatchGenerator(cls,
                                sampler,
                                atomizer,
                                seed: int,
                                max_position_embeddings: int,
                                ) -> "data_generators.MaskLMBatchGenerator":
    """Initializes data generator for inference."""
    d                         = MaskLMBatchGenerator()
    d.sampler                 = sampler
    d.atomizer                = atomizer
    d.rngen                   = random.Random(seed)
    d.max_position_embeddings = max_position_embeddings
    return d

  def generateTfDataset(self,
                      sequence_length,
                      is_training,
                      num_cpu_threads,
                      use_tpu = False,
                      ) -> "tf.Dataset":
    """Wrapper function that constructs a tf.Dataset used for training BERT."""
    def _decode_record(record, name_to_features):
      """Decodes a record to a TensorFlow example."""
      ## This function assumes record is still a file (expressed as TF dataset)
      ## It decodes this record to tf scalars.
      example = tf.io.parse_single_example(record, name_to_features)
      # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
      # So cast all int64 to int32.
      for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
          t = tf.cast(t, dtype = tf.int32)
        example[name] = t
      return example

    def input_fn(params):
      """
      function used by tf.estimator to generate inputs for training.
      """
      batch_size = params["batch_size"]
      name_to_features = {
          "input_ids"               : tf.io.FixedLenFeature([sequence_length], tf.int64),
          "input_mask"              : tf.io.FixedLenFeature([sequence_length], tf.int64),
          "masked_lm_positions"     : tf.io.FixedLenFeature([self.training_opts.max_predictions_per_seq], tf.int64),
          "masked_lm_ids"           : tf.io.FixedLenFeature([self.training_opts.max_predictions_per_seq], tf.int64),
          "masked_lm_weights"       : tf.io.FixedLenFeature([self.training_opts.max_predictions_per_seq], tf.float32),
          "next_sentence_labels"    : tf.io.FixedLenFeature([1], tf.int64),
      }

      dataset = tf.io.gfile.glob(str(self.tfRecord))
      # For training, we want a lot of parallel reading and shuffling.
      # For eval, we want no shuffling and parallel reading doesn't matter.
      if is_training:
        d = tf.data.Dataset.from_tensor_slices(tf.constant(dataset))
        d = d.repeat()
        if self.training_opts.shuffle_corpus_contentfiles_between_epochs:
          d = d.shuffle(buffer_size=len(dataset))

        # `cycle_length` is the number of parallel files that get read.
        cycle_length = min(num_cpu_threads, len(dataset))

        # `sloppy` mode means that the interleaving is not exact. This adds
        # even more randomness to the training pipeline.
        d = d.apply(
            tf.data.experimental.parallel_interleave(
                tf.data.TFRecordDataset,
                sloppy=is_training,
                cycle_length=cycle_length))
        if self.training_opts.shuffle_corpus_contentfiles_between_epochs:
          d = d.shuffle(buffer_size=100)
      else:
        d = tf.data.TFRecordDataset(dataset)
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
              drop_remainder=use_tpu))
      return d
    return input_fn

  def generateTfSamples(self):
    """
    Contains input_fn closure function for estimator
    
    Returns:
      input_fn callable.
    """

    def sample_gen(batch_size: int):
      """
      This generator yields iteratively the inference input blob for each step.
      In the first iteration, it yields sampler.encoded_start_text and then for each step,
      self.sampleBatch is updated with the model's current output through self.updateVocabulary.
      The generator stops when the model has filled all mask or hole tokens with predictions.

      Arguments:
        batch_size: The batch size used during inference.

      Yields:
        Current step's inference input for model.

      Returns:
        None
      """
      assert batch_size == len(self.sampleBatch), "{}, {}".format(batch_size, len(self.sampleBatch))
      while True:

        (input_ids, input_mask, masked_lm_positions, 
        masked_lm_ids, masked_lm_weights) = [], [], [], [], []

        max_mask_len = max(
        [len(np.where(np.in1d(np.asarray(x), [self.atomizer.maskToken, self.atomizer.holeToken]))[0]) for x in self.sampleBatch]
        )
        if max_mask_len == 0:
          return
        for sample in self.sampleBatch:
          sample_masks = np.where(np.in1d(sample, [self.atomizer.maskToken, self.atomizer.holeToken]))[0]
          actual_mask_len = len(sample_masks)
          len_offset     = max_mask_len - actual_mask_len
          pad_idx      = np.where(sample == self.atomizer.padToken)[0]
          inp_mask     = np.ones(len(sample), dtype = np.int32)
          if len(pad_idx) > 0:
            inp_mask[pad_idx[0]:] = 0

          input_ids.append(list(sample))
          input_mask.append(list(inp_mask))
          masked_lm_positions.append(list(sample_masks) + [0] * len_offset)
          masked_lm_ids.append([self.atomizer.maskToken] * actual_mask_len + [self.atomizer.padToken] * len_offset)
          masked_lm_weights.append([0.0] * (actual_mask_len + len_offset))

        yield (input_ids, input_mask, 
          masked_lm_positions, masked_lm_ids, 
          masked_lm_weights, np.zeros([batch_size, 1]))

    def input_fn(params):
      """
      function used by tf.estimator to generate inputs for inference.
      """
      batch_size = params['batch_size']

      sample = tf.data.Dataset.from_generator(
                lambda: sample_gen(batch_size), 
                output_types = MaskSequence.tfTypes(),
                output_shapes = MaskSequence.tfShapes(batch_size, self.sampler.sequence_length)
                )

      it = tf.compat.v1.data.make_one_shot_iterator(sample)
      (input_ids, input_mask,
        masked_lm_positions, masked_lm_ids,
        masked_lm_weights, next_sentence_labels) = it.get_next()

      return {
          'input_ids'             : input_ids,
          'input_mask'            : input_mask,
          'masked_lm_positions'   : masked_lm_positions,
          'masked_lm_ids'         : masked_lm_ids,
          'masked_lm_weights'     : masked_lm_weights,
          'next_sentence_labels'  : next_sentence_labels,
      }
    return input_fn

  def CreateCorpus(self) -> None:
    """
    Constructs training corpus in text format, stores it in
    self.shaped_corpus

    Each corpus datapoint is either a single kernel or a random
    sequence of size sequence_length (legacy).
    """
    start_time = time.time()

    # Set corpus dimension parameters
    sequence_length             = self.training_opts.sequence_length
    batch_size             = self.training_opts.batch_size
    dupe_factor                 = self.training_opts.dupe_factor
    shuffle                     = self.training_opts.shuffle_corpus_contentfiles_between_epochs
    pad                         = [self.atomizer.padToken   ]
    start                       = [self.atomizer.startToken ]
    end                         = [self.atomizer.endToken   ]

    # generate a kernel corpus
    encoded_corpus       = self.corpus.GetTrainingData()

    if self.config.datapoint_type == "kernel":

      # Reject larger than sequence length
      initial_length       = copy.deepcopy(len(encoded_corpus))
      encoded_corpus       = [list(x) for x in encoded_corpus if 
                             len(x) <= sequence_length - (2 if self.config.use_start_end else 0)] # Account for start and end token
      reduced_length       = copy.deepcopy(len(encoded_corpus))
      # Add start/end tokens
      if self.config.use_start_end:
        encoded_corpus     = [self._addStartEndToken(kf) for kf in encoded_corpus]
      # pad sequences to sequence length
      encoded_corpus       = np.array([x + pad * (sequence_length - len(x)) for x in encoded_corpus])
      # Clone datapoints dupe_factor times
      self.shaped_corpus   = np.repeat(encoded_corpus, dupe_factor, axis = 0)
      # Shuffle
      if shuffle:
        self.rngen.shuffle(self.shaped_corpus)
      assert len(self.shaped_corpus) != 0, "Not enought data. All kernels have been rejected."

      # Set corpus epoch parameters
      self.steps_per_epoch = min(self.training_opts.num_train_steps, FLAGS.steps_per_epoch)
      self.num_epochs      = int(self.training_opts.num_train_steps / self.steps_per_epoch)

      assert self.shaped_corpus.ndim     == 2, "corpus dim: {}".format(self.shaped_corpus.shape)
      assert self.shaped_corpus.shape[1] == sequence_length, "Dim 1 shape mismatch: {}, target: {}".format(encoded_corpus.shape[1], sequence_length)

      l.getLogger().info("{} kernels were rejected (larger than sequence_length)".format(initial_length - reduced_length))
      l.getLogger().info(
        "Loaded corpus of shape {} ({} kernels remained, multiplied by dupe factor: {}) in {} ms.".format(
                  self.shaped_corpus.shape,
                  reduced_length,
                  dupe_factor,
                  humanize.intcomma(int((time.time() - start_time) * 1000)),
              )
      )
    elif self.config.datapoint_type == "statement":
    ## This branch is legacy data processing

      if shuffle:
        self.rngen.shuffle(encoded_corpus)
      encoded_corpus = np.concatenate(encoded_corpus)
      encoded_corpus = np.tile(encoded_corpus, dupe_factor)

      # Set corpus dimension parameters
      self.steps_per_epoch        = int(len(encoded_corpus) / (batch_size * sequence_length * dupe_factor))
      assert self.steps_per_epoch != 0, "Not enought data. Use smaller sequence_length and/or batch_size"
      self.num_epochs             = int(self.training_opts.num_train_steps / self.steps_per_epoch)

      clipped_corpus_length       = dupe_factor * self.steps_per_epoch * batch_size * sequence_length
      clipped_corpus              = encoded_corpus[:clipped_corpus_length]

      self.shaped_corpus = np.split(clipped_corpus, batch_size * self.steps_per_epoch * dupe_factor, 0)

      np_corpus = np.asarray(self.shaped_corpus)
      assert np_corpus.ndim == 2, "Wrong dimensions for shaped_corpus: {}".format(np_corpus.shape)
      assert np_corpus.shape[1] == sequence_length, "Second dimension is not equal to sequence length: {}".format(np_corpus.shape[1])

      l.getLogger().info(
        "Loaded corpus of {} tokens (clipped last {} tokens) in {} ms.".format(
                  humanize.intcomma(clipped_corpus_length),
                  humanize.intcomma(len(encoded_corpus) - clipped_corpus_length),
                  humanize.intcomma(int((time.time() - start_time) * 1000)),
              )
      )

    else:
      raise ValueError("Unrecognized datapoint_type: {}".format(self.config.datapoint_type))

    return

  def _MaskCorpus(self, 
                 corpus: np.array
                )-> None:
    """
    Entrypoint function that inserts masks or holes to the corpus.

    Arguments:
      corpus: [num_datapoints, sequence_length], 
              where num_datapoints = num_batches * dupe_factor * batch_size
    """
    self.masked_corpus = []

    with progressbar.ProgressBar(max_value = len(corpus)) as bar:
        for idx, kernel in enumerate(corpus):
          if self.config.HasField("mask"):
            masked_seq = self._maskSequence(kernel)
          elif self.config.HasField("hole"):
            masked_seq = self._holeSequence(kernel)
          else:
            raise AttributeError("target predictions can only be mask or hole {}".format(self.config))
          self.masked_corpus.append(masked_seq)
          bar.update(idx)
    self.masked_corpus[0].LogBatchTelemetry(self.training_opts.batch_size, self.steps_per_epoch, self.num_epochs)
    return

  def _holeSequence(self,
                    seq: np.array,
                    ) -> MaskSequence:
    """
    Inserts hole tokens to a given sequence.
    """
    assert seq.ndim == 1, "Input for masking must be single-dimension array."

    ## Tuple representation of mask id/position for easy sorting
    class MaskedLmInstance(typing.NamedTuple):
      pos_index: int
      token_id: int

    # Actual length represents the sequence length before pad begins
    if self.atomizer.padToken in seq:
      actual_length   = np.where(seq == self.atomizer.padToken)[0][0]
    else:
      actual_length   = len(seq)

    candidate_indexes = np.arange(actual_length)
    self.rngen.shuffle(candidate_indexes)

    # total tokens to add in holes.
    # No more than max_predictions_per_seq, 
    # no less than actual seq length x the probability of hiding a token
    holes_to_predict  = min(self.training_opts.max_predictions_per_seq,
                           max(1, int(round(actual_length * self.training_opts.masked_lm_prob))))

    # Processed input sequence
    input_ids         = list(np.copy(seq))
    # List of (seq_idx, token_id) tupless
    masked_lms        = []
    # Offset array. Indices represent elements in the initial array (seq)
    # Values of indices represent current offset position in processed array (input_ids).
    offset_idxs        = np.zeros(len(seq), dtype = np.int32)
    # Total masks placed so far.
    total_predictions = 0
    for pos_index in candidate_indexes:
      assert pos_index < len(seq), "Candidate index is out of bounds: {} >= {}".format(pos_index, len(seq))
      
      # Element in processed array can be found in its original index +/- offset
      input_id_idx = pos_index + offset_idxs[pos_index]
      if total_predictions >= holes_to_predict:
        break
      ## TODO. This condition could be troublesome in case it gets False by accident
      ## i.e. the index has gone wrong due to the whole, BUT still point to an identical element
      elif seq[pos_index] != input_ids[input_id_idx]:
        continue

      assert (input_ids[input_id_idx] == seq[pos_index], 
              "Original and offset-ted sequence have misaligned tokens: {}, {}"
              .format(seq[pos_index], input_ids[input_id_idx]))

      # Random number to represent the length of this hole.
      hole_length = self.rngen.randint(0, self.config.hole.hole_length)
      # Inside range, make sure hole length does not run over input_id_idx bounds
      hole_length = min(hole_length, len(input_ids) - input_id_idx)
      # Confirm there is no conflict with another hole, further down the sequence.
      for i in range(hole_length):
        if input_ids[input_id_idx + i] == self.atomizer.holeToken:
          hole_length = i
          break

      # Target token for classifier is either the first token of the hole, or endholToken if hole is empty
      target = input_ids[input_id_idx] if hole_length > 0 else self.atomizer.endholeToken

      ## TODO. Think about '== self.atomizer.holeToken' condition.
      # if self.config.mask.random_placed_mask and hole_length != 0:
      #   if self.rngen.random() < 0.8:
      #     replacement_token = self.atomizer.holeToken
      #   else:
      #     if self.rngen.random() > 0.5:
      #       # Sometimes keep the original token.
      #       replacement_token = target
      #     else:
      #       # Other times add a random one.
      #       replacement_token = self.rngen.randint(0, self.atomizer.vocab_size - 1)
      # else:
      #   replacement_token = self.atomizer.holeToken
      replacement_token = self.atomizer.holeToken

      input_ids = (input_ids[:input_id_idx] + 
                   [replacement_token] + 
                   input_ids[input_id_idx + hole_length:])

      masked_lms.append(MaskedLmInstance(pos_index=input_id_idx, token_id=target))
      if not self.config.mask.random_placed_mask:
        assert (input_ids[input_id_idx] == self.atomizer.holeToken, 
              "target index does not correspond to hole token: {}".format(self.atomizer.DeatomizeIndices([input_ids[input_id_idx]])))

      # Adjust the offset of all affected tokens, from pos_index and after.
      offset_idxs[pos_index:] += 1 - hole_length
      # An empty hole is counted as a prediction of count 1.
      total_predictions       += max(1, hole_length)

    while len(input_ids) < len(seq):
      input_ids.append(self.atomizer.padToken)
    masked_lms = sorted(masked_lms, key=lambda x: x.pos_index)
    masked_lm_positions, masked_lm_ids, masked_lm_weights = [], [], []

    input_mask = np.ones(len(seq), dtype = np.int32)
    if self.atomizer.padToken in input_ids:
      input_mask[input_ids.index(self.atomizer.padToken):] = 0

    next_sentence_label = np.int32(0)
    ## Related to next_sentence_label: Fix it to 0 for now, as no next_sentence prediction
    ## is intended on kernels. In any other case, check bert's create_instances_from_document
    ## to see how next_sentence_labels are calculated.
    ## Setting this to 0 means that next sentence is NOT random.
    ## Note that if next_sentence prediction is to be embedded, [SEP] token has to be added.

    for p in masked_lms:
      masked_lm_positions.append(p.pos_index)
      masked_lm_ids.append(p.token_id)
      masked_lm_weights.append(1.0)
    while len(masked_lm_positions) < self.training_opts.max_predictions_per_seq:
        masked_lm_positions.append(0)
        masked_lm_ids.append(self.atomizer.padToken)
        masked_lm_weights.append(0.0)

    return MaskSequence(np.asarray(input_ids[:len(seq)]), input_mask,
                        np.asarray(masked_lm_positions),  np.asarray(masked_lm_ids), 
                        np.asarray(masked_lm_weights),    next_sentence_label
                        )

  def _maskSequence(self,
                    seq: np.array,
                    ) -> MaskSequence:
    """Inserts masks to a given sequence."""
    assert seq.ndim == 1, "Input for masking must be single-dimension array."

    ## Tuple representation of mask id/position for easy sorting
    class MaskedLmInstance(typing.NamedTuple):
      pos_index: int
      token_id: int

    # Actual length represents the sequence length before pad begins
    if self.atomizer.padToken in seq:
      actual_length = np.where(seq == self.atomizer.padToken)[0][0]
    else:
      actual_length = len(seq)

    candidate_indexes = np.arange(actual_length)
    self.rngen.shuffle(candidate_indexes)

    masks_to_predict = min(self.training_opts.max_predictions_per_seq,
                           max(1, int(round(actual_length * self.training_opts.masked_lm_prob))))
    input_ids = list(np.copy(seq))
    masked_lms = []

    for pos_index in candidate_indexes:
      if len(masked_lms) >= masks_to_predict:
        break

      if self.config.mask.random_placed_mask:
        # 80% of the time, replace with [MASK]
        if self.rngen.random() < 0.8:
          input_ids[pos_index] = self.atomizer.maskToken
        else:
          # 10% of the time, keep original
          if self.rngen.random() < 0.5:
            pass
          # 10% of the time, replace with random word
          else:
            input_ids[pos_index] = self.rngen.randint(0, self.atomizer.vocab_size - 1)
      else:
        if self.rngen.random() < 0.8:
          input_ids[pos_index] = self.atomizer.maskToken

      masked_lms.append(MaskedLmInstance(pos_index=pos_index, token_id=seq[pos_index]))

    assert len(masked_lms) <= masks_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x.pos_index)

    masked_lm_positions, masked_lm_ids, masked_lm_weights = [], [], []

    input_mask = np.ones(len(seq), dtype = np.int32)
    if self.atomizer.padToken in input_ids:
      input_mask[input_ids.index(self.atomizer.padToken):] = 0
    next_sentence_label = np.int32(0)
    ## Related to next_sentence_label: Fix it to 0 for now, as no next_sentence prediction
    ## is intended on kernels. In any other case, check bert's create_instances_from_document
    ## to see how next_sentence_labels are calculated.
    ## Setting this to 0 means that next sentence is NOT random.
    ## Note that if next_sentence prediction is to be embedded, [SEP] token has to be added.

    for p in masked_lms:
      masked_lm_positions.append(p.pos_index)
      masked_lm_ids.append(p.token_id)
      masked_lm_weights.append(1.0)
    while len(masked_lm_positions) < self.training_opts.max_predictions_per_seq:
        masked_lm_positions.append(0)
        masked_lm_ids.append(self.atomizer.padToken)
        masked_lm_weights.append(0.0)

    return MaskSequence(np.asarray(input_ids),           input_mask,
                        np.asarray(masked_lm_positions), np.asarray(masked_lm_ids), 
                        np.asarray(masked_lm_weights),   next_sentence_label
                        )

  def InitSampleBatch(self) -> None:
    """
    Initializes data_generator for inference.
    self.sampleBatch is initialized with sampler.encoded_start_text
    """
    assert self.sampler.sequence_length <= self.max_position_embeddings, "Sampler sequence length exceeds max position embeddings."
    input_sample = self.sampler.encoded_start_text
    assert np.ndim(input_sample) == 1, "Input samples have to be one-dimensional. {} given.".format(input_sample.shape)

    target_idx = np.where(np.in1d(input_sample, [self.atomizer.maskToken, self.atomizer.holeToken]))[0]
    assert len(target_idx) != 0, "No target prediction in sample text"

    padded_sample = self._padToMaxPosition(input_sample)
    padded_sample = padded_sample[:self.sampler.sequence_length]
    self.sampleBatch = np.repeat(padded_sample[None, :], self.sampler.batch_size, axis = 0)
    return

  def updateSampleBatch(self, 
                        input_ids     : np.array,
                        masked_lm_ids : np.array,
                        ) -> np.array:
    """
    Updates self.sampleBatch with the model's output prediction.
    The output, if still contains hole or mask tokens, is fed back
    to the model's input through the input_fn's sample_gen generator.
    """
    assert len(input_ids) == len(masked_lm_ids), "Inputs and predictions do not have the same batch size."

    updated_sequence = []
    done = True

    for batch_idx, _ in enumerate(input_ids):
      batch = []
      mask_id_index = 0
      for idx, token in enumerate(input_ids[batch_idx]):
        if   token == self.atomizer.maskToken:
          mt = masked_lm_ids[batch_idx][mask_id_index]
          mask_id_index += 1
          batch.append(mt)
        elif token == self.atomizer.holeToken:
          mt = masked_lm_ids[batch_idx][mask_id_index]
          mask_id_index += 1
          if mt != self.atomizer.endholeToken:
            batch.append(mt)
            batch.append(self.atomizer.holeToken)
            done = False
        else:
          batch.append(token)
      batch = np.asarray(batch)
      batch = self._padToMaxPosition(batch)
      # TODO, chop sequence for now, but TODO it: 
      # If a sequence is bigger than it should, crop one or both edges,
      # save them and send max_position_embeddings for next step.
      # Then, concat it back.
      batch = batch[:self.sampler.sequence_length]
      updated_sequence.append(batch)

    self.sampleBatch = np.asarray(updated_sequence)
    return self.sampleBatch, done

  def _saveCorpusTfRecord(self) -> None:
    """Converts corpus nparrays to tf Features and stores corpus to TfRecord"""
    l.getLogger().debug("deeplearning.clgen.models.data_generators.MaskLMBatchGenerator._saveCorpusTfRecord()")
     
    writer = tf.io.TFRecordWriter(str(self.tfRecord))
    if FLAGS.write_text_dataset:
      file_writer = open(self.txtRecord, 'w')

    for (inst_index, instance) in enumerate(self.masked_corpus):
      input_ids  = instance.input_ids
      input_mask = instance.input_mask

      assert len(input_ids) == self.training_opts.sequence_length, "len(input_ids):  {}, sequence_length: {}".format(len(input_ids), self.training_opts.sequence_length)

      masked_lm_positions   = instance.masked_lm_positions
      masked_lm_ids         = instance.masked_lm_ids
      masked_lm_weights     = instance.masked_lm_weights
      next_sentence_label   = instance.next_sentence_label
      features              = collections.OrderedDict()

      features["input_ids"]             = tf.train.Feature(int64_list = tf.train.Int64List(
                                                                value = list(input_ids)))

      features["input_mask"]            = tf.train.Feature(int64_list = tf.train.Int64List(
                                                                value = list(input_mask)))

      features["masked_lm_positions"]   = tf.train.Feature(int64_list = tf.train.Int64List(
                                                                value = list(masked_lm_positions)))

      features["masked_lm_ids"]         = tf.train.Feature(int64_list = tf.train.Int64List(
                                                                value = list(masked_lm_ids)))

      features["masked_lm_weights"]     = tf.train.Feature(float_list = tf.train.FloatList(
                                                                value = list(masked_lm_weights)))

      features["next_sentence_labels"]  = tf.train.Feature(int64_list = tf.train.Int64List(
                                                                value = list([next_sentence_label])))

      tf_example = tf.train.Example(features = tf.train.Features(feature = features))
      writer.write(tf_example.SerializeToString())
      if FLAGS.write_text_dataset:
        file_writer.write("'input_ids': {}\n'input_mask': {}\n'masked_lm_positions': {}\n'masked_lm_ids': {}\n'masked_lm_weights': {}\n'next_sentence_labels': {}\n\n"
                            .format(self.atomizer.DeatomizeIndices(input_ids),
                                    input_mask, 
                                    masked_lm_positions, 
                                    self.atomizer.DeatomizeIndices(masked_lm_ids), 
                                    masked_lm_weights, 
                                    next_sentence_label)
                            )
    writer.close()
    if FLAGS.write_text_dataset:
      file_writer.close()
    l.getLogger().info("Wrote {} instances ({} batches of {} datapoints) to {}"
                      .format(inst_index, self.steps_per_epoch, self.training_opts.batch_size, self.tfRecord))
    return

  def _padToMaxPosition(self, input_sample):
    """
    Pads a given sequence to the maximum allowed sequence length, which is max_position_embeddings
    
    Arguments:
      input_sample: np.array or list that represents a sequence

    Returns:
      padded sequence in np.array format
    """
    return np.concatenate([input_sample, 
                          np.array([self.atomizer.padToken] * 
                              (self.max_position_embeddings - len(input_sample)), dtype = np.int32)
                          ])

  def _addStartEndToken(self, inp: list) -> list:
    """
    Inserts [START] and [END] token at the beginnning and end of a sequence
    
    Arguments:
      inp: input_sequence

    Returns:
      [START] + input_sequence + [END]
    """
    assert len(inp) != 0, "Empty list provided."
    assert self.atomizer.padToken not in inp, "Use this function before padding a sequence!"

    start = [self.atomizer.startToken] if inp[0]  != self.atomizer.startToken else []
    end   = [self.atomizer.endToken  ] if inp[-1] != self.atomizer.endToken   else []
    if isinstance(inp, list):
      return start + inp + end
    elif isinstance(inp, np.ndarray):
      raise NotImplementedError