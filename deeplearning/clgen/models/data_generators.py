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

import numpy as np

from deeplearning.clgen import cache
from deeplearning.clgen.tf import tf
from deeplearning.clgen.proto import model_pb2

from absl import flags
from labm8.py import humanize

from eupy.native import logger as l

FLAGS = flags.FLAGS

flags.DEFINE_boolean(
  "write_text_dataset", 
  False, 
  "Set True for MaskLM data generator to write dataset in text format, along with the tfRecord."
)

flags.DEFINE_boolean(
  "randomize_mask_placement",
  True,
  "When selecting an index in the input tensor, the original BERT model gives 80% chance "
  "to replace it with a MASK, a 10% chance to replace it with another random token "
  "and another 10% to leave it be after all. Set True to enable this behavior. Otherwise, "
  "when selecting an index in the input, this will be replaced by a MASK.",
)

flags.DEFINE_boolean(
  "use_start_end_metatokens", 
  True, 
  "Use [START] and [END] meta tokens at the beginning and end of each sequence."
)

flags.DEFINE_string(
  "mask_or_hole",
  "hole",
  "Set target prediction of MaskLM as [MASK] tokens or [HOLE] sequences."
)

flags.DEFINE_string(
  "datapoint_type",
  "kernel",
  "Represent single training instance as whole padded kernel, or arbitrary statement sequences."
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
    l.getLogger().info("Step shape: X: {}, y" ": {}.".format(batch.X.shape, batch.y.shape))
    l.getLogger().info(
      "Memory: {} per batch, {} per epoch, {} total.".format(
              humanize.BinaryPrefix(self.sizeof_batch, "B"),
              humanize.BinaryPrefix(self.sizeof_batch * steps_per_epoch, "B"),
              humanize.BinaryPrefix(self.sizeof_batch * steps_per_epoch * num_epochs, "B"),
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
              humanize.BinaryPrefix(self.sizeof_sequence * batch_size, "B"),
              humanize.BinaryPrefix(self.sizeof_sequence * batch_size * steps_per_epoch, "B"),
              humanize.BinaryPrefix(self.sizeof_sequence * batch_size * steps_per_epoch * num_epochs, "B"),
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
  def __init__(self):
    l.getLogger().debug("deeplearning.clgen.models.data_generators.MaskLMBatchGenerator.__init__()")

    self.corpus                     = None
    self.atomizer                   = None
    self.shaped_corpus              = None
    self.masked_corpus              = None

    self.training_opts              = None
    self.steps_per_epoch            = None
    self.batch_size                 = None
    self.max_position_embeddings    = None
    self.sequence_length            = None
    self.target_predictions         = None

    self.tfRecord                   = None
    self.txtRecord                  = None
    self.sampleBatch                = None

    self.sampler                    = None
    self.rngen                      = None
    return

  @classmethod
  def TrainMaskLMBatchGenerator(cls,
                               corpus: "corpuses.Corpus",
                               training_opts: model_pb2.TrainingOptions,
                               cache_path
                               ) -> "data_generators.MaskLMBatchGenerator":
    d                     = MaskLMBatchGenerator()
    d.corpus              = corpus
    d.atomizer            = corpus.atomizer
    d.training_opts       = training_opts
    d.target_predictions  = FLAGS.mask_or_hole
    d.tfRecord            = cache_path / "dataset" / "maskedDataset.tf_record"
    d.txtRecord           = cache_path / "dataset" / "maskedDataset.txt"
    d.rngen               = random.Random(training_opts.random_seed)

    d.tfRecord.parent.mkdir(exist_ok = True, parents = True)
    d.CreateCorpus()
    if not d.tfRecord.exists():
      d._MaskCorpus(d.shaped_corpus)
      d._saveCorpusTfRecord()
    return d

  @classmethod
  def SampleMaskLMBatchGenerator(cls,
                                sampler,
                                atomizer,
                                seed,
                                max_position_embeddings,
                                ) -> "data_generators.MaskLMBatchGenerator":
    d                         = MaskLMBatchGenerator()
    d.sampler                 = sampler
    d.atomizer                = atomizer
    d.rngen                   = random.Random(seed)
    d.batch_size              = sampler.batch_size
    d.max_position_embeddings = max_position_embeddings
    return d

  def generateTfDataset(self,
                      sequence_length,
                      is_training,
                      num_cpu_threads,
                      use_tpu = False,
                      ) -> "tf.Dataset":

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

    def data_gen():


      while True:

        (input_ids, input_mask, masked_lm_positions, 
        masked_lm_ids, masked_lm_weights) = [], [], [], [], []


        for sample in self.sampleBatch:
          sample_masks = np.where(np.in1d(sample, [self.atomizer.maskToken, self.atomizer.holeToken]))[0]

          pad_idx      = np.where(sample == self.atomizer.padToken)[0]
          inp_mask     = np.ones(len(sample), dtype = np.int32)
          if len(pad_idx) > 0:
            inp_mask[pad_idx[0]:] = 0

          input_ids.append(list(sample))
          input_mask.append(list(inp_mask))
          masked_lm_positions.append(list(sample_masks))
          masked_lm_ids.append([self.atomizer.padToken] * len(sample_masks))
          masked_lm_weights.append([0.0] * len(sample_masks))
        l.getLogger().warn(self.atomizer.DeatomizeIndices(input_ids))
        yield (input_ids, input_mask, 
          masked_lm_positions, masked_lm_ids, 
          masked_lm_weights, np.zeros([1,1]))

    def input_fn(params):
      # batch_size = params["batch_size"]
      # assert isinstance(self.sampleBatch, np.ndarray), "input sample is not in np.array format"
      # assert batch_size == len(self.sampleBatch), "{}, {}".format(batch_size, len(self.sampleBatch))

      # (input_ids, input_mask, masked_lm_positions, 
      # masked_lm_ids, masked_lm_weights) = [], [], [], [], []

      ## TODO leave this for now. Different batch sizes look troublesome.
      ## TODO. Related to issue #49: In masked_lm_ids, instead of padTokens, use hole/masks with pad
      ## to align all batches together and keep track of different lengths.
      # max_mask_len = max(
      # [len(np.where(np.in1d(np.asarray(x), [self.atomizer.maskToken, self.atomizer.holeToken]))[0]) for x in self.sampleBatch]
      # )
      # for sample in self.sampleBatch:
      #   sample_masks = np.where(np.in1d(sample, [self.atomizer.maskToken, self.atomizer.holeToken]))[0]

      #   pad_idx      = np.where(sample == self.atomizer.padToken)[0]
      #   inp_mask     = np.ones(len(sample), dtype = np.int32)
      #   if len(pad_idx) > 0:
      #     inp_mask[pad_idx[0]:] = 0

      #   input_ids.append(list(sample))
      #   input_mask.append(list(inp_mask))
      #   masked_lm_positions.append(list(sample_masks))
      #   masked_lm_ids.append([self.atomizer.padToken] * len(sample_masks))
      #   masked_lm_weights.append([0.0] * len(sample_masks))

      sample = tf.data.Dataset.from_generator(data_gen, 
        output_types = (tf.int32, tf.int32, tf.int32, tf.int32, tf.float32, tf.int32),
        output_shapes = (tf.TensorShape([1,256]), tf.TensorShape([1,256]),
        tf.TensorShape([1,3]), tf.TensorShape([1,3]), tf.TensorShape([1,3]),
        tf.TensorShape([1,1]) ) )
      sample = sample.repeat(1)
      sample = sample.batch(1)
      it = tf.compat.v1.data.make_one_shot_iterator(sample)
      (input_ids, input_mask,
        masked_lm_positions, masked_lm_ids,
        masked_lm_weights, next_sentence_labels) = it.get_next()
      input_ids = tf.squeeze(input_ids, axis = 0)
      input_mask = tf.squeeze(input_mask, axis = 0)
      masked_lm_positions = tf.squeeze(masked_lm_positions, axis = 0)
      masked_lm_ids = tf.squeeze(masked_lm_ids, axis = 0)
      masked_lm_weights = tf.squeeze(masked_lm_weights, axis = 0)
      next_sentence_labels = tf.squeeze(next_sentence_labels, axis = 0)
      l.getLogger().critical(input_ids)
      l.getLogger().critical(input_mask)
      l.getLogger().critical(masked_lm_positions)
      l.getLogger().critical(masked_lm_ids)
      l.getLogger().critical(masked_lm_weights)
      l.getLogger().critical(next_sentence_labels)
      return {
          'input_ids'             : input_ids,
          'input_mask'            : input_mask,
          'masked_lm_positions'   : masked_lm_positions,
          'masked_lm_ids'         : masked_lm_ids,
          'masked_lm_weights'     : masked_lm_weights,
          'next_sentence_labels'  : next_sentence_labels,
      }
      # return {
      #     'input_ids'             : tf.convert_to_tensor(input_ids,           dtype = tf.int32),
      #     'input_mask'            : tf.convert_to_tensor(input_mask,          dtype = tf.int32),
      #     'masked_lm_positions'   : tf.convert_to_tensor(masked_lm_positions, dtype = tf.int32),
      #     'masked_lm_ids'         : tf.convert_to_tensor(masked_lm_ids,       dtype = tf.int32),
      #     'masked_lm_weights'     : tf.convert_to_tensor(masked_lm_weights,   dtype = tf.float32),
      #     'next_sentence_labels'  : tf.zeros((batch_size, 1),                 dtype = tf.int32)
      # }
    return input_fn

  def CreateCorpus(self) -> None:
    l.getLogger().debug("deeplearning.clgen.models.data_generators.MaskLMBatchGenerator.CreateBatches()")
    start_time = time.time()

    # Set corpus dimension parameters
    self.sequence_length        = self.training_opts.sequence_length
    self.batch_size             = self.training_opts.batch_size
    dupe_factor                 = self.training_opts.dupe_factor
    shuffle                     = self.training_opts.shuffle_corpus_contentfiles_between_epochs
    pad                         = [self.atomizer.padToken   ]
    start                       = [self.atomizer.startToken ]
    end                         = [self.atomizer.endToken   ]

    # generate a kernel corpus
    encoded_corpus       = self.corpus.GetTrainingData()

    if FLAGS.datapoint_type == "kernel":

      # Reject larger than sequence length
      initial_length       = copy.deepcopy(len(encoded_corpus))
      encoded_corpus       = [list(x) for x in encoded_corpus if 
                             len(x) <= self.sequence_length - (2 if FLAGS.use_start_end_metatokens else 0)] # Account for start and end token
      reduced_length       = copy.deepcopy(len(encoded_corpus))
      # Add start/end tokens
      if FLAGS.use_start_end_metatokens:
        encoded_corpus     = [self._addStartEndToken(kf) for kf in encoded_corpus]
      # pad sequences to sequence length
      encoded_corpus       = np.array([x + pad * (self.sequence_length - len(x)) for x in encoded_corpus])
      # Clone datapoints dupe_factor times
      self.shaped_corpus   = np.repeat(encoded_corpus, dupe_factor, axis = 0)
      # Shuffle
      if shuffle:
        self.rngen.shuffle(self.shaped_corpus)
      assert len(self.shaped_corpus) != 0, "Not enought data. All kernels have been rejected."

      # Set corpus epoch parameters
      self.steps_per_epoch = min(self.training_opts.num_train_steps, 250) ## TODO add this as flag or pb param
      self.num_epochs      = int(self.training_opts.num_train_steps / self.steps_per_epoch)

      assert self.shaped_corpus.ndim     == 2, "corpus dim: {}".format(self.shaped_corpus.shape)
      assert self.shaped_corpus.shape[1] == self.sequence_length, "Dim 1 shape mismatch: {}, target: {}".format(encoded_corpus.shape[1], self.sequence_length)

      l.getLogger().info("{} kernels were rejected (larger than sequence_length)".format(initial_length - reduced_length))
      l.getLogger().info(
        "Loaded corpus of shape {} ({} kernels remained, multiplied by dupe factor: {}) in {} ms.".format(
                  self.shaped_corpus.shape,
                  reduced_length,
                  dupe_factor,
                  humanize.Commas(int((time.time() - start_time) * 1000)),
              )
      )
    elif FLAGS.datapoint_type == "statement":
    ## This branch is legacy data processing

      if shuffle:
        self.rngen.shuffle(encoded_corpus)
      encoded_corpus = np.concatenate(encoded_corpus)
      encoded_corpus = np.tile(encoded_corpus, dupe_factor)

      # Set corpus dimension parameters
      self.steps_per_epoch        = int(len(encoded_corpus) / (self.batch_size * self.sequence_length * dupe_factor))
      assert self.steps_per_epoch != 0, "Not enought data. Use smaller sequence_length and/or batch_size"
      self.num_epochs             = int(self.training_opts.num_train_steps / self.steps_per_epoch)

      clipped_corpus_length       = dupe_factor * self.steps_per_epoch * self.batch_size * self.sequence_length
      clipped_corpus              = encoded_corpus[:clipped_corpus_length]

      self.shaped_corpus = np.split(clipped_corpus, self.batch_size * self.steps_per_epoch * dupe_factor, 0)

      np_corpus = np.asarray(self.shaped_corpus)
      assert np_corpus.ndim == 2, "Wrong dimensions for shaped_corpus: {}".format(np_corpus.shape)
      assert np_corpus.shape[1] == self.sequence_length, "Second dimension is not equal to sequence length: {}".format(np_corpus.shape[1])

      l.getLogger().info(
        "Loaded corpus of {} tokens (clipped last {} tokens) in {} ms.".format(
                  humanize.Commas(clipped_corpus_length),
                  humanize.Commas(len(encoded_corpus) - clipped_corpus_length),
                  humanize.Commas(int((time.time() - start_time) * 1000)),
              )
      )

    else:
      raise ValueError("Unrecognized datapoint_type: {}".format(FLAGS.datapoint_type))

    return

  def InitSampleBatch(self,
                      input_sample,
                      sequence_length,
                      ) -> None:

    assert np.ndim(input_sample) == 1, "Input samples have to be one-dimensional. {} given.".format(input_sample.shape)

    target_idx = np.where(np.in1d(input_sample, [self.atomizer.maskToken, self.atomizer.holeToken]))[0]
    assert len(target_idx) != 0, "No target prediction in sample text"

    # if len(target_idx) > 1:
    #   raise NotImplementedError(
    #     """\
    #       Having more than one [MASK]/[HOLE] tokens in a to-be-predicted \
    #       text is something I cannot handle now (and don't know if I should \
    #       anyway). There are some reasons for this:
    #       a) When two any of [MASK] or [HOLE] appear in a sampling text \
    #       one should think through which should be given first for inference.\
    #       This, should make a difference given the model takes into account the context \
    #       of the whole sentence.
    #       b) generateTfSamples constructs the arrays that hold indices of all masks/holes \
    #       to be predicted. If a text contains multiple masks/holes in a single timestep, \
    #       then some algorithm will have to be embedded to make a queue of these many masks/holes \
    #       to provide them one at a time. Alternatively, they could be provided alltogether in a \
    #       batched timestep fashion. Now, that sounds weird so, let me provide an example:
    #       If the sampler encounters this: 'kernel [HOLE] A([HOLE)', then this could be fed \
    #       into the model with both holes at the same time, and then ask for a single prediction for \
    #       each. Then two predictions replace the HOLES, a new HOLE is appended after the predicted \
    #       token for any of these two predictions does not match the ENDHOLE token, and that goes on \
    #       in this batched-timestep fashion. 

    #       But for now, let's assume input_sample contains either one MASK or one HOLE.
    #     """
    #     )

    # expanded_sample = self._expandHoleToMasks(
    #       input_sample, sequence_length - len(input_sample) + 1
    #       )
    padded_sample = self._padToMaxPosition(input_sample)
    assert len(padded_sample) == self.max_position_embeddings, "Padded sequence does not match max_position_embeddings"
    self.sampleBatch = np.repeat(padded_sample[None, :], self.batch_size, axis = 0)
    return

  def updateSampleBatch(self, 
                        input_ids     : np.array,
                        masked_lm_ids : np.array,
                        ) -> np.array:
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
      batch = batch[:self.max_position_embeddings]
      assert (len(masked_lm_ids[batch_idx]) == mask_id_index,
        "Not all predicted masks have been popped: {} remaining".format(len(masked_lm_ids)))
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

      assert len(input_ids) == self.sequence_length, "len(input_ids):  {}, self.sequence_length: {}".format(len(input_ids), self.sequence_length)

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
        file_writer.write("'input_ids': {}\n'input_mask': {}\n'masked_lm_positions': {}\n'masked_lm_ids': {}\'nmasked_lm_weights': {}\n'next_sentence_labels': {}\n\n"
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
                      .format(inst_index, self.steps_per_epoch, self.batch_size, self.tfRecord))
    return

  def _MaskCorpus(self, 
                 corpus: np.array
                )-> None:
    l.getLogger().debug("deeplearning.clgen.models.data_generators.MaskLMBatchGenerator._MaskCorpus()")
    self.masked_corpus = []

    with progressbar.ProgressBar(max_value = len(corpus)) as bar:
        for idx, kernel in enumerate(corpus):
          if self.target_predictions == "mask":
            masked_seq = self._maskSequence(kernel)
          elif self.target_predictions == "hole":
            masked_seq = self._holeSequence(kernel)
          else:
            raise AttributeError("target predictions cannot be {}".format(self.target_predictions))
          self.masked_corpus.append(masked_seq)
          bar.update(idx)
    self.masked_corpus[0].LogBatchTelemetry(self.batch_size, self.steps_per_epoch, self.num_epochs)
    return

  def _holeSequence(self,
                    seq: np.array,
                    ) -> MaskSequence:

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
      hole_length = self.rngen.randint(0, 3)
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
      # if FLAGS.randomize_mask_placement and hole_length != 0:
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
      if not FLAGS.randomize_mask_placement:
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

      if FLAGS.randomize_mask_placement:
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

  def _expandHoleToMasks(self,
                        sample: np.array, 
                        length: int
                        ) -> np.array:

    ## TODO. Essentially this snippet below finds the indices of a random token
    ## and instructs a function to replace that token  with a sequence of masks
    ## That is too specific over "replacing holes with masks" but can/should be 
    ## generalized to anything
    hole_index = np.where(sample == self.atomizer.holeToken)[0]
    if len(hole_index) == 0: ## Nothing to do, no holes found
      return sample
    if len(hole_index) > 1:
      l.getLogger().warning("Multiple instances of {} are found. \
                              Selecting the first one.".format(self.atomizer.holeLabel))

    fhidx = hole_index[0]
    return np.concatenate([sample[:fhidx], 
                            np.array([self.atomizer.maskToken] * length, dtype = np.int32),
                            sample[fhidx + 1:]])

  def _padToMaxPosition(self, input_sample):
    return np.concatenate([input_sample, 
                          np.array([self.atomizer.padToken] * 
                              (self.max_position_embeddings - len(input_sample)), dtype = np.int32)
                          ])

  def _addStartEndToken(self, inp: list) -> list:
    assert len(inp) != 0, "Empty list provided."
    assert self.atomizer.padToken not in inp, "Use this function before padding a sequence!"

    start = [self.atomizer.startToken] if inp[0]  != self.atomizer.startToken else []
    end   = [self.atomizer.endToken  ] if inp[-1] != self.atomizer.endToken   else []
    if isinstance(inp, list):
      return start + inp + end
    elif isinstance(inp, np.ndarray):
      raise NotImplementedError