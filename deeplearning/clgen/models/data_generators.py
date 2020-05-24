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
from deeplearning.clgen.tf import tf
from deeplearning.clgen.proto import model_pb2

from absl import flags
from labm8.py import humanize

from eupy.native import logger as l

FLAGS = flags.FLAGS


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
    l.getLogger().debug("deeplearning.clgen.models.data_generators.DataBatch.LogBatchTelemetry()")
    l.getLogger().info("Step shape: X: {}, y" ": {}.".format(batch.X.shape, batch.y.shape))
    l.getLogger().info(
      "Memory: {} per batch, {} per epoch, {} total.".format(
              humanize.BinaryPrefix(self.sizeof_batch, "B"),
              humanize.BinaryPrefix(self.sizeof_batch * steps_per_epoch, "B"),
              humanize.BinaryPrefix(self.sizeof_batch * steps_per_epoch * num_epochs, "B"),
          )
    )
    return

class MaskBatch(typing.NamedTuple):
  """Tuple representation of a single MaskLM batch"""

  input_ids            : np.array
  masked_lm_positions  : np.array
  masked_lm_ids        : np.array
  masked_lm_weights    : np.array
  next_sentence_label  : np.int32

  @property
  def sizeof_batch(self):
    return (sys.getsizeof(self) + self.input_ids.nbytes +
           self.masked_lm_positions.nbytes + self.masked_lm_ids.nbytes +
           self.masked_lm_weights.nbytes + self.next_sentence_label.nbytes
           )

  def LogBatchTelemetry(self,
                        steps_per_epoch: int,
                        num_epochs: int,
                        ) -> None:
    """Log analytics about the batch."""
    l.getLogger().debug("deeplearning.clgen.models.data_generators.MaskBatch.LogBatchTelemetry()")
    l.getLogger().info("Step shape: Input_ids: {}, masked_lm_positions: {}, masked_lm_ids: {}, masked_lm_weights: {}, next_sentence_label: {}"
                        .format(self.input_ids.shape, 
                                self.masked_lm_positions.shape, 
                                self.masked_lm_ids.shape,
                                self.masked_lm_weights.shape,
                                self.next_sentence_label.shape
                                )
                        )
    l.getLogger().info(
      "Memory: {} per batch, {} per epoch, {} total.".format(
              humanize.BinaryPrefix(self.sizeof_batch, "B"),
              humanize.BinaryPrefix(self.sizeof_batch * steps_per_epoch, "B"),
              humanize.BinaryPrefix(self.sizeof_batch * steps_per_epoch * num_epochs, "B"),
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
    self.shaped_corpus              = None
    self.masked_corpus              = None

    self.training_opts              = None
    self.steps_per_epoch            = None
    self.batch_size                 = None
    self.max_position_embeddings    = None
    self.sequence_length            = None

    self.tfRecord                   = None
    self.sampler                    = None
    self.rngen                      = None
    return

  @classmethod
  def TrainMaskLMBatchGenerator(cls,
                               corpus: "corpuses.Corpus",
                               training_opts: model_pb2.TrainingOptions,
                               cache_path
                               ) -> "data_generators.MaskLMBatchGenerator":
    d               = MaskLMBatchGenerator()
    d.corpus        = corpus
    d.training_opts = training_opts
    d.tfRecord      = cache_path / "dataset" / "maskedDataset.tf_record"
    d.rngen         = random.Random(d.training_opts.random_seed)

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
    d = MaskLMBatchGenerator()
    d.sampler = sampler
    d.atomizer = atomizer
    d.rngen = random.Random(seed)
    d.max_position_embeddings = max_position_embeddings
    d.is_training = False
    return d

  def generateTfDataset(self,
                      max_seq_length,
                      is_training,
                      num_cpu_threads,
                      use_tpu = False,
                      ) -> "tf.Dataset":

    def _decode_record(record, name_to_features):
      """Decodes a record to a TensorFlow example."""
      ## This function assumes record is still a file (expressed as TF dataset)
      ## It decodes this record to tf scalars.
      ## You already have them so this will be skipped
      example = tf.io.parse_single_example(record, name_to_features)

      # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
      # So cast all int64 to int32.
      for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
          # t = tf.compat.v1.to_int32(t)
          t = tf.cast(t, dtype = tf.int32)

        example[name] = t

      return example

    def input_fn(params):

      batch_size = params["batch_size"]
      name_to_features = {
          "input_ids":
              tf.io.FixedLenFeature([max_seq_length], tf.int64),
          "masked_lm_positions":
              tf.io.FixedLenFeature([self.training_opts.max_predictions_per_seq], tf.int64),
          "masked_lm_ids":
              tf.io.FixedLenFeature([self.training_opts.max_predictions_per_seq], tf.int64),
          "masked_lm_weights":
              tf.io.FixedLenFeature([self.training_opts.max_predictions_per_seq], tf.float32),
          "next_sentence_labels":
              tf.io.FixedLenFeature([1], tf.int64),
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


  def InitSampleBatch(self,
                      input_sample,
                      max_seq_length,
                      batch_size,
                      ) -> None:

    assert np.ndim(input_sample) == 1, "Input samples have to be one-dimensional. {} given.".format(input_sample.shape)
    expanded_sample = self._expandHoleToMasks(
          input_sample, max_seq_length - len(input_sample) + 1
          )

    padded_sample = self._padToMaxPosition(expanded_sample)

    assert len(padded_sample) == self.max_position_embeddings, "Padded sequence does not match max_position_embeddings"
    self.tfRecord = np.repeat(padded_sample[None, :], batch_size, axis = 0)
    return

  def _expandHoleToMasks(self,
                        sample: np.array, 
                        length: int
                        ) -> np.array:

    ## TODO. Essentially this snippet below finds the indices of a random token
    ## and instructs a function to replace that token  with a sequence of masks
    ## That is too specific over "replacing holes with masks" but can/should be 
    ## generalized to anything
    hole_index = np.where(sample == self.corpus.atomizer.holeToken)[0]
    if len(hole_index) == 0: ## Nothing to do
      return sample
    if len(hole_index) > 1:
      l.getLogger().warning("Multiple instances of {} are found. \
                              Selecting the first one.".format(self.corpus.atomizer.holeLabel))

    fhidx = hole_index[0]
    return np.concatenate([sample[:fhidx], 
                            np.array([self.corpus.atomizer.maskToken] * length, dtype = np.int32),
                            sample[fhidx + 1:]
                          ])

  def _padToMaxPosition(self, input_sample):
    return np.concatenate([
                input_sample, np.array([self.corpus.atomizer.padToken] * (
                                              self.max_position_embeddings - len(input_sample)
                                              ), 
                                        dtype = np.int32
                                      )
            ])

  def generateTfSamples(self):

    def input_fn(params):
      batch_size = params["batch_size"]
      assert batch_size == len(self.tfRecord)

      tfSampleBatch = {
          'input_ids'             : [[] * batch_size],
          'masked_lm_positions'   : [[] * batch_size],
          'masked_lm_ids'         : [[] * batch_size],
          'masked_lm_weights'     : [[] * batch_size],
          'next_sentence_labels'  : tf.convert_to_tensor([[1] * batch_size], dtype = tf.int32)
        }
      for bidx, sample in enumerate(self.tfRecord):
        input_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights = [], [], [], []
        for tidx, token in enumerate(sample):
          input_ids.append(token)
          if token == self.corpus.atomizer.maskToken:
            masked_lm_positions.append(tidx)
            masked_lm_ids.append(token)
            masked_lm_weights.append(0.0)
        tfSampleBatch['input_ids'][bidx]            = input_ids
        tfSampleBatch['masked_lm_positions'][bidx]  = masked_lm_positions
        tfSampleBatch['masked_lm_ids'][bidx]        = masked_lm_ids
        tfSampleBatch['masked_lm_weights'][bidx]    = masked_lm_weights

      tfSampleBatch['input_ids']            = tf.convert_to_tensor(tfSampleBatch['input_ids'], dtype = tf.int32)
      tfSampleBatch['masked_lm_positions']  = tf.convert_to_tensor(tfSampleBatch['masked_lm_positions'], dtype = tf.int32)
      tfSampleBatch['masked_lm_ids']        = tf.convert_to_tensor(tfSampleBatch['masked_lm_ids'], dtype = tf.int32)
      tfSampleBatch['masked_lm_weights']    = tf.convert_to_tensor(tfSampleBatch['masked_lm_weights'], dtype = tf.float32)

      return tfSampleBatch
    return input_fn

  def CreateCorpus(self) -> None:
    l.getLogger().debug("deeplearning.clgen.models.data_generators.MaskLMBatchGenerator.CreateBatches()")
    start_time = time.time()

    # generate a kernel corpus
    encoded_corpus = np.concatenate(self.corpus.GetTrainingData())
    encoded_corpus = np.repeat(encoded_corpus, self.training_opts.dupe_factor)

    if self.training_opts.shuffle_corpus_contentfiles_between_epochs:
      self.rngen.shuffle(encoded_corpus)

    # Set corpus dimension parameters
    self.sequence_length        = self.training_opts.sequence_length
    self.batch_size             = self.training_opts.batch_size
    self.steps_per_epoch        = int(len(encoded_corpus) / (self.batch_size * self.sequence_length))
    assert self.steps_per_epoch != 0, "Not enought data. Use smaller sequence_length and/or batch_size"
    self.num_epochs             = int(self.training_opts.num_train_steps / self.steps_per_epoch)

    # split into batches (TODO remove clipping)
    clipped_corpus_length       = self.steps_per_epoch * self.batch_size * self.sequence_length
    clipped_corpus              = encoded_corpus[:clipped_corpus_length]

    self.shaped_corpus          = np.split(clipped_corpus.reshape(self.batch_size, -1), self.steps_per_epoch, 1)

    l.getLogger().info(
      "Loaded corpus of {} tokens (clipped last {} tokens) in {} ms.".format(
                humanize.Commas(clipped_corpus_length),
                humanize.Commas(len(encoded_corpus) - clipped_corpus_length),
                humanize.Commas(int((time.time() - start_time) * 1000)),
            )
    )
    return

  def _saveCorpusTfRecord(self) -> None:
    """Converts corpus nparrays to tf Features and stores corpus to TfRecord"""
    l.getLogger().debug("deeplearning.clgen.models.data_generators.MaskLMBatchGenerator._saveCorpusTfRecord()")
     
    writer = tf.io.TFRecordWriter(str(self.tfRecord))
    for (inst_index, instance) in enumerate(self.masked_corpus):
      input_ids = instance.input_ids

      assert len(input_ids) == self.sequence_length

      masked_lm_positions   = instance.masked_lm_positions
      masked_lm_ids         = instance.masked_lm_ids
      masked_lm_weights     = instance.masked_lm_weights
      next_sentence_label   = instance.next_sentence_label
      features              = collections.OrderedDict()

      features["input_ids"]             = tf.train.Feature(int64_list = tf.train.Int64List(
                                                                value = list(input_ids)))

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

    writer.close()
    l.getLogger().info("Wrote {} instances ({} batches of {} datapoints) to {}"
                      .format(inst_index, self.steps_per_epoch, self.batch_size, self.tfRecord))
    return

  def _MaskCorpus(self, 
                 corpus: np.array
                )-> None:
    l.getLogger().debug("deeplearning.clgen.models.data_generators.MaskLMBatchGenerator._MaskCorpus()")
    l.getLogger().warn("Masking Corpus is a slow process. Assign multiple threads to it")

    self.masked_corpus = []

    with progressbar.ProgressBar(max_value = len(corpus)) as bar:
        for idx, batch in enumerate(corpus):
          self.masked_corpus.extend(self._maskBatch(batch))
          bar.update(idx)
    self.masked_corpus[0].LogBatchTelemetry(self.steps_per_epoch, self.num_epochs)
    return

  def _maskBatch(self, 
                batch: np.array
                ) -> typing.List[MaskBatch]:
    out_batch = []
    for seq in batch:
      inp_id, mask_pos, mask_tok, mask_wei, next_sentence_label = self._maskSequence(seq)
      out_batch.append(MaskBatch(np.asarray(inp_id), 
                                 np.asarray(mask_pos), 
                                 np.asarray(mask_tok), 
                                 np.asarray(mask_wei),
                                 next_sentence_label
                                 )
                      )
    return out_batch

  def _maskSequence(self,
                   seq: np.array,
                  ) -> MaskBatch:
    l.getLogger().debug("deeplearning.clgen.models.data_generators.MaskLMBatchGenerator._maskSequence()")
      
    assert seq.ndim == 1, "Input for masking must be single-dimension array."
    candidate_indexes = np.arange(len(seq))
    self.rngen.shuffle(candidate_indexes)

    masks_to_predict = min(self.training_opts.max_predictions_per_seq,
                          max(1, int(round(len(seq) * self.training_opts.masked_lm_prob))))
    input_ids = np.copy(seq)
    masked_lms = []

    for pos_index in candidate_indexes:
      if len(masked_lms) >= masks_to_predict:
        break

      # 80% of the time, replace with [MASK]
      if self.rngen.random() < 0.8:
        input_ids[pos_index] = self.corpus.atomizer.maskToken ## TODO ?????
      # The else block below is debatable for this use case. So comment out for now
      # else:
      #   # 10% of the time, keep original
      #   if self.rngen.random() < 0.5:
      #     pass
      #   # 10% of the time, replace with random word
      #   else:
      #     input_ids[pos_index] = self.rngen.randint(0, self.corpus.atomizer.vocab_size - 1)

      class MaskedLmInstance(typing.NamedTuple):
        pos_index: int
        token_id: int

      masked_lms.append(MaskedLmInstance(pos_index=pos_index, token_id=seq[pos_index]))

    assert len(masked_lms) <= masks_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x.pos_index)

    masked_lm_positions, masked_lm_ids, masked_lm_weights = [], [], []
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
        masked_lm_ids.append(self.corpus.atomizer.padToken)
        masked_lm_weights.append(0.0)

    return (input_ids, masked_lm_positions, 
            masked_lm_ids, masked_lm_weights, next_sentence_label)
