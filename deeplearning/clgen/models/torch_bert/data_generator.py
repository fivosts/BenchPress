"""This file defines the streaming generators for model training data.

We train models on overlapping one-hot encoded text sequences. For a corpus of
a reasonable size, the full training data may not fit in memory. This modules
provides Python Generator classes for use by a sequential Keras model's
fit_generator() method to stream batches of training data.
"""
import sys
import os
import time
import typing
import random
import progressbar
import collections
import copy
import glob
import humanize
import multiprocessing
import functools
import pickle
import numpy as np

from deeplearning.clgen.util import cache
from deeplearning.clgen.util import pbutil
from deeplearning.clgen.util import distributions
from deeplearning.clgen.util import pytorch
from deeplearning.clgen.util.pytorch import torch
from deeplearning.clgen.proto import model_pb2
from absl import flags
from eupy.native import logger as l

FLAGS = flags.FLAGS

# flags.DEFINE_boolean(
#   "write_text_dataset", 
#   False, 
#   "Set True for MaskLM data generator to write dataset in text format, along with the tf_record."
# )

# flags.DEFINE_integer(
#   "memory_limit",
#   4,
#   "Set maximum amount of available memory used for masking sequences in Gb. [Default]: 4",
# )
# flags.DEFINE_boolean(
#   "force_remake_dataset",
#   False,
#   "Force data generator to re-mask encoded dataset and store tf_record."
# )

class MaskSequence(typing.NamedTuple):
  """
  Tuple representation of a single MaskLM Instance. 
  This is not batch! generateTfDataset applies native batching,
  so this class represents a single instance!
  """

  seen_in_training     : np.int64
  original_input       : np.array
  input_ids            : np.array
  input_mask           : np.array
  position_ids         : np.array
  mask_labels          : np.array
  masked_lm_lengths    : np.array
  next_sentence_label  : np.int64

  @staticmethod
  def estimatedSize(batch_size, sequence_length, max_predictions_per_seq):
    return (
      2 * np.zeros([batch_size, 1], dtype = np.int64).nbytes + 
      5 * np.zeros([batch_size, sequence_length], dtype = np.int64).nbytes +
      1 * np.zeros([batch_size, max_predictions_per_seq], dtype = np.int64).nbytes
      )

  @property
  def sizeof_sequence(self):
    return (sys.getsizeof(self) + 
           self.seen_in_training.nbytes  + self.original_input.nbytes + 
           self.input_ids.nbytes         + self.input_mask.nbytes +
           self.position_ids.nbytes      + self.mask_labels.nbytes +
           self.masked_lm_lengths.nbytes + self.next_sentence_label.nbytes
           )

  @staticmethod
  def LogBatchTelemetry(batch_size: int,
                        sequence_length: int,
                        max_predictions_per_seq: int,
                        steps_per_epoch: int,
                        num_epochs: int,
                        ) -> None:
    """Log analytics about the batch."""
    l.getLogger().info(
      "Memory: {} per batch, {} per epoch, {} total.".format(
              humanize.naturalsize(MaskSequence.estimatedSize(1, sequence_length, max_predictions_per_seq) * batch_size, binary = True),
              humanize.naturalsize(MaskSequence.estimatedSize(1, sequence_length, max_predictions_per_seq) * batch_size * steps_per_epoch, binary = True),
              humanize.naturalsize(MaskSequence.estimatedSize(1, sequence_length, max_predictions_per_seq) * batch_size * steps_per_epoch * num_epochs, binary = True),
          )
    )

def _holeSequence(seq: np.array,
                  train_set: bool,
                  max_predictions: int,
                  pickled_distribution: distributions.Distribution,
                  pickled_atomizer,
                  rngen: random.Random,
                  use_start_end: bool,
                  training_opts,
                  ) -> typing.Dict:
  """
  Inserts hole tokens to a given sequence.
  """
  assert seq.ndim == 1, "Input for masking must be single-dimension array."

  ## Tuple representation of mask id/position/hole_length for easy sorting
  class MaskedLmInstance():
    def __init__(self, 
                 pos_index: int, 
                 token_id: int, 
                 hole_length: int,
                 ):
      self.pos_index   = pos_index
      self.token_id    = token_id
      self.hole_length = hole_length

  # Unpack atomizer and sampler
  distribution = pickle.loads(pickled_distribution)
  atomizer     = pickle.loads(pickled_atomizer)

  # Actual length represents the sequence length before pad begins
  if use_start_end:
    actual_length   = np.where(seq == atomizer.endToken)[0][0]
  elif padToken in seq:
    actual_length   = np.where(seq == atomizer.padToken)[0][0]
  else:
    actual_length   = len(seq)

  if use_start_end:
    candidate_indexes = np.arange(1, actual_length)
  else:
    candidate_indexes = np.arange(actual_length)
  rngen.shuffle(candidate_indexes)

  # total tokens to add in holes.
  # No more than max_predictions_per_seq (or otherwise specified), no less than actual seq length x the probability of hiding a token
  holes_to_predict  = min(max_predictions,
                         max(1, int(round(actual_length * training_opts.masked_lm_prob))))

  # Flip input sequence to spread the hole lenghts to both directions uniformly.
  reverse_sequence = True if rngen.random() > 0.5 else False
  if reverse_sequence:
    input_ids         = list(np.copy(np.flip(seq)))
    candidate_indexes = (len(seq) - 1) - candidate_indexes
    actual_length     = len(seq) - (1 if use_start_end else 0)
  else:
    input_ids       = list(np.copy(seq))
  # List of (seq_idx, token_id, hole_length) tuples
  masked_lms        = []
  # Offset array. Indices represent elements in the initial array (seq)
  # Values of indices represent current offset position in processed array (input_ids).
  offset_idxs       = np.zeros(len(seq), dtype = np.int64)
  # Set with all candidate_indexes that have been holed.
  visited_indices   = set()
  # Total masks placed so far.
  total_predictions = 0
  for pos_index in candidate_indexes:
    assert pos_index < len(seq), "Candidate index is out of bounds: {} >= {}".format(pos_index, len(seq))
    
    # Element in processed array can be found in its original index +/- offset
    input_id_idx = pos_index + offset_idxs[pos_index]
    if total_predictions >= holes_to_predict:
      break
    elif pos_index in visited_indices:
      # Do not target an index, already holed
      continue
    elif input_id_idx > len(seq):
      # Do not mask a part of input_ids that is going to be cropped.
      continue

    assert (input_ids[input_id_idx] == seq[pos_index], 
            "Original and offset-ted sequence have misaligned tokens: {}, {}"
            .format(seq[pos_index], input_ids[input_id_idx]))

    # Sampled number from distribution to represent the actual hole length
    hole_length = distribution.sample()
    # Inside range, make sure hole length does not run over input_id_idx bounds
    hole_length = min(hole_length, actual_length - input_id_idx)
    # Confirm there is no conflict with another hole, further down the sequence.
    for i in range(hole_length):
      if input_ids[input_id_idx + i] == atomizer.holeToken:
        hole_length = i
        break
    distribution.register(hole_length)
    
    # Target token for classifier is either the first token of the hole, or endholToken if hole is empty
    target = input_ids[input_id_idx] if hole_length > 0 else atomizer.endholeToken

    ## TODO. Think about '== self.atomizer.holeToken' condition.
    # if config.mask.random_placed_mask and hole_length != 0:
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
    replacement_token = atomizer.holeToken

    input_ids = (input_ids[:input_id_idx] + 
                 [replacement_token] + 
                 input_ids[input_id_idx + hole_length:])
    # This pos_index will get deprecated when someone before this index alters the offset array
    # So store position index, and after making all masks, update with updated offset array
    masked_lms.append(MaskedLmInstance(pos_index = pos_index, token_id = target, hole_length = hole_length))

    # Adjust the offset of all affected tokens, from pos_index and after.
    offset_idxs[pos_index + 1:] += 1 - hole_length
    # An empty hole is counted as a prediction of count 1.
    total_predictions           += max(1, hole_length)
    visited_indices.update(range(pos_index, pos_index + hole_length))

    for lm in masked_lms:
      ## TODO, remove this extensive-expensive check after you make sure that this function is bug free.
      test_index = lm.pos_index + offset_idxs[lm.pos_index]
      if input_ids[test_index] != atomizer.holeToken:
        assert False

  # Now update the entries with offset index.
  for lm in masked_lms:
    prev_index = lm.pos_index
    lm.pos_index = lm.pos_index + offset_idxs[lm.pos_index]

  # Un-reverse sequence
  if reverse_sequence:
    input_ids = list(reversed(input_ids))
    for lm in masked_lms:
      lm.pos_index = len(input_ids) - 1 - lm.pos_index

  # Now check that targets point only hole tokens
  for lm in masked_lms:
    assert input_ids[lm.pos_index] == atomizer.holeToken

  while len(input_ids) < len(seq):
    input_ids.append(atomizer.padToken)
  masked_lms = sorted(masked_lms, key=lambda x: x.pos_index)

  input_mask = np.ones(len(seq), dtype = np.int64)
  if atomizer.padToken in input_ids:
    first_pad_index = input_ids.index(atomizer.padToken)
    input_mask[first_pad_index:] = 0
    # Check that the pad index is likely correct.
    assert input_ids[first_pad_index] == atomizer.padToken, "{}".format(input_ids)
    assert input_ids[first_pad_index - 1] != atomizer.padToken

  seen_in_training    = np.ones([1]) if train_set else np.zeros([1])
  next_sentence_label = np.zeros([1])
  """
    Related to next_sentence_label: Fix it to 0 for now, as no next_sentence prediction
    is intended on kernels. In any other case, check bert's create_instances_from_document
    to see how next_sentence_label are calculated.
    Setting this to 0 means that next sentence is NOT random.
    Note that if next_sentence prediction is to be embedded, [SEP] token has to be added.    
  """
  if len(masked_lms) == 0:
    l.getLogger().warn("No HOLE added to datapoint. Increase probability of hole occuring.")

  masked_lm_lengths = np.full(max_predictions, -1, dtype = np.int64)
  mask_labels = np.full(len(seq), -100, dtype = np.int64)
  ind = 0
  for p in masked_lms:
    if p.pos_index < len(seq):
      mask_labels[p.pos_index] = p.token_id
      masked_lm_lengths[ind]   = p.hole_length
      ind += 1

  return {
    'seen_in_training'   : seen_in_training,
    'original_input'     : seq,
    'input_ids'          : np.asarray(input_ids[:len(seq)], dtype = np.int64),
    'input_mask'         : input_mask,
    'position_ids'       : np.arange(len(seq), dtype = np.int64),
    'mask_labels'        : mask_labels,
    'masked_lm_lengths'  : masked_lm_lengths,
    'next_sentence_label': next_sentence_label,
  }

def _maskSequence(seq: np.array,
                  train_set: bool,
                  max_predictions: int,
                  pickled_atomizer,
                  rngen: random.Random,
                  training_opts,
                  config,
                  ) -> typing.Dict:
  """Inserts masks to a given sequence."""
  assert seq.ndim == 1, "Input for masking must be single-dimension array."

  ## Tuple representation of mask id/position for easy sorting
  class MaskedLmInstance(typing.NamedTuple):
    pos_index: int
    token_id: int

  # Unpack atomizer
  atomizer = pickle.loads(pickled_atomizer)

  # Actual length represents the sequence length before pad begins
  if atomizer.padToken in seq:
    actual_length = np.where(seq == atomizer.padToken)[0][0]
  else:
    actual_length = len(seq)

  candidate_indexes = np.arange(actual_length)
  rngen.shuffle(candidate_indexes)

  masks_to_predict = min(max_predictions,
                         max(1, int(round(actual_length * training_opts.masked_lm_prob))))
  input_ids = list(np.copy(seq))
  masked_lms = []

  for pos_index in candidate_indexes:
    if len(masked_lms) >= masks_to_predict:
      break

    if config.mask.random_placed_mask:
      # 80% of the time, replace with [MASK]
      if rngen.random() < 0.8:
        input_ids[pos_index] = atomizer.maskToken
      else:
        # 10% of the time, keep original
        if rngen.random() < 0.5:
          pass
        # 10% of the time, replace with random word
        else:
          random_token = rngen.randint(0, atomizer.vocab_size - 1)
          while any(atomizer.vocab[t] == random_token for (idx, t) in atomizer.metaTokens.items()):
            random_token = rngen.randint(0, atomizer.vocab_size - 1)
          input_ids[pos_index] = rngen.randint(0, atomizer.vocab_size - 1)
    else:
      if rngen.random() < 0.8:
        input_ids[pos_index] = atomizer.maskToken

    masked_lms.append(MaskedLmInstance(pos_index=pos_index, token_id=seq[pos_index]))

  assert len(masked_lms) <= masks_to_predict
  masked_lms = sorted(masked_lms, key=lambda x: x.pos_index)

  input_mask = np.ones(len(seq), dtype = np.int64)
  if atomizer.padToken in input_ids:
    input_mask[input_ids.index(atomizer.padToken):] = 0

  seen_in_training    = np.ones([1]) if train_set else np.zeros([1])
  next_sentence_label = np.zeros([1])
  ## Related to next_sentence_label: Fix it to 0 for now, as no next_sentence prediction
  ## is intended on kernels. In any other case, check bert's create_instances_from_document
  ## to see how next_sentence_label are calculated.
  ## Setting this to 0 means that next sentence is NOT random.
  ## Note that if next_sentence prediction is to be embedded, [SEP] token has to be added.

  masked_lm_lengths = np.full(max_predictions, -1, dtype = np.int64)
  mask_labels = np.full(len(seq), -100, dtype = np.int64)
  ind = 0
  for p in masked_lms:
    if p.pos_index < len(seq):
      mask_labels[p.pos_index] = p.token_id
      masked_lm_lengths[ind]   = p.hole_length
      ind += 1

  return {
    'seen_in_training'   : seen_in_training,
    'original_input'     : seq,
    'input_ids'          : np.asarray(input_ids[:len(seq)], dtype = np.int64),
    'input_mask'         : input_mask,
    'position_ids'       : np.arange(len(seq), dtype = np.int64),
    'mask_labels'        : mask_labels,
    'masked_lm_lengths'  : masked_lm_lengths,
    'next_sentence_label': next_sentence_label,
  }

class MaskLMBatchGenerator(object):
  def __init__(self):

    self.dataset                 = None
    self.corpus                  = None
    self.atomizer                = None
    self.config                  = None
    self.cache                   = None

    self.training_opts           = None
    self.steps_per_epoch         = None
    self.max_position_embeddings = None

    self.dataloader              = None
    self.sampleBatch             = None
    self.sampleIndices           = None

    self.sampler                 = None
    self.tfRecordSampler         = None
    self.rngen                   = None
    return

  @classmethod
  def TrainMaskLMBatchGenerator(cls,
                               corpus: "corpuses.Corpus",
                               training_opts: model_pb2.TrainingOptions,
                               cache_path,
                               ) -> "data_generator.MaskLMBatchGenerator":
    """Initializes data generator for training."""
    d               = MaskLMBatchGenerator()
    d.cache         = cache.mkcache(cache_path, "dataset")
    d.dataset       = {}
    d.corpus        = corpus
    d.atomizer      = corpus.atomizer
    d.config        = training_opts.data_generator
    d.training_opts = training_opts
    d.rngen         = random.Random(training_opts.random_seed)

    d.cache.path.mkdir(exist_ok = True, parents = True)
    shaped_corpus = d.createCorpus()
    d.configDataset(shaped_corpus)
    d.initDataloader()
    return d

  @classmethod
  def SampleMaskLMBatchGenerator(cls,
                                sampler,
                                atomizer,
                                seed: int,
                                max_position_embeddings: int,
                                cache_path,
                                ) -> "data_generator.MaskLMBatchGenerator":
    """Initializes data generator for inference."""
    d                         = MaskLMBatchGenerator()
    d.cache                   = cache.mkcache(cache_path, "dataset")
    d.sampler                 = sampler
    d.atomizer                = atomizer
    d.rngen                   = random.Random(seed)
    d.max_position_embeddings = max_position_embeddings
    if not d.sampler.isFixedStr:
      d.tfRecordSampler = d.tfRecordSampleGenerator()
    return d

  def configDataset(self, shaped_corpus) -> None:
    """
      Configs all necessary training and validation 
      sets described in the model protobuf.
      First constructs training set and optionally 
      splits it into validation set, if selected in config.
      Then configValidationSets is called which
      constructs any additional validation_set elements
      provided in the model's config.
    """
    assert self.config.validation_split >= 0 and self.config.validation_split <= 100

    if FLAGS.force_remake_dataset:
      l.getLogger().warn("Force remaking datasets can cause lots of problems on an already trained model. Are you sure you want to proceed ? [y/n]")
      a = input()
      if a.lower() != "yes" and a.lower() != "y":
        l.getLogger().warn("Overwriting dataset process was aborted. Good call.")
        return

    if len(glob.glob(str(self.cache.path / "train_dataset_*.pt_record"))) == 0 or FLAGS.force_remake_dataset:
      if self.config.validation_split == 0:
        self._maskCorpus(
          shaped_corpus, set_name = "train_dataset", train_set = True
        )
      else:
        split_index  = int((len(shaped_corpus) / 100) * self.config.validation_split)
        self._maskCorpus(
          shaped_corpus[split_index:], set_name = "train_dataset", train_set = True
        )
        self._maskCorpus(
          shaped_corpus[:split_index], set_name = "validation_dataset", train_set = False
        )
    else:
      self.dataset["train_dataset"] = {
        "pt_record": glob.glob(str(self.cache.path / "train_dataset_*.pt_record")),
        "txt"      : glob.glob(str(self.cache.path / "train_dataset_*.txt")),
      }
      self.dataset["validation_dataset"] = {
        "pt_record": glob.glob(str(self.cache.path / "validation_dataset_*.pt_record")),
        "txt"      : glob.glob(str(self.cache.path / "validation_dataset_*.txt")),
      }

    self.configValidationSets(self.config.validation_set, shaped_corpus)
    return

  def configValidationSets(self, valset_list, shaped_corpus) -> None:
    """
      Mask and store any extra validation datasets defined into
      model protobuf.
      Example:
        validation_set {
          max_predictions_per_seq: 10
          hole {
            hole_length: 15
            uniform_distribution: true
          }
        }

      Arguments:
        valset_list: list of validation_set items
      Returns:
        None
    """
    for valset in valset_list:
      set_name = "pred_{}_{}".format(
        valset.max_predictions_per_seq,
        "mask" if valset.HasField("mask") else "hole_{}".format(valset.hole.hole_length)
      )
      if set_name in self.dataset or len(glob.glob(str(self.cache.path / "{}_*.pt_record".format(set_name)))) > 0:
        continue
      self._maskCorpus(
        shaped_corpus, train_set = False, set_name = set_name, config = valset
      )
    return

  def initDataloader(self) -> None:
    """Pytorch dataloader that assembles all dataset files into a single-mapped dataset."""
    dataset = torch.utils.data.ConcatDataset(
                [torch.load(x) for x in self.dataset['train_dataset']['pt_record']]
              )
    self.dataloader = torch.utils.data.dataloader.DataLoader(
      dataset    = dataset,
      batch_size = self.training_opts.batch_size,
      sampler    = (
            torch.utils.data.RandomSampler(dataset, replacement = False)
            if not pytorch.torch_tpu_available or pytorch.torch_xla.xrt_world_size() <= 1
            else torch.utils.data.distributed.DistributedSampler(
                  dataset, 
                  num_replicas = pytorch.torch_xla.xrt_world_size(), 
                  rank = pytorch.torch_xla.get_ordinal()
                 )
            ),
      num_workers = os.cpu_count(),
      drop_last   = True,
    )
    return

  def createCorpus(self) -> None:
    """
    Constructs training corpus in text format, stores it in
    shaped_corpus

    Each corpus datapoint is either a single kernel or a random
    sequence of size sequence_length (legacy).
    """
    start_time = time.time()

    # Set corpus dimension parameters
    sequence_length = self.training_opts.sequence_length
    batch_size      = self.training_opts.batch_size
    dupe_factor     = self.training_opts.dupe_factor
    shuffle         = self.training_opts.shuffle_corpus_contentfiles_between_epochs
    pad             = [self.atomizer.padToken   ]
    start           = [self.atomizer.startToken ]
    end             = [self.atomizer.endToken   ]
    shaped_corpus   = None

    # generate a kernel corpus
    encoded_corpus  = self.corpus.GetTrainingData()

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
      # shaped_corpus   = np.repeat(encoded_corpus, dupe_factor, axis = 0)
      shaped_corpus     = encoded_corpus
      # Shuffle
      if shuffle:
        self.rngen.shuffle(shaped_corpus)
      assert len(shaped_corpus) != 0, "Not enought data. All kernels have been rejected."

      # Set corpus epoch parameters
      self.num_epochs      = int(self.training_opts.num_train_steps / self.config.steps_per_epoch)
      self.steps_per_epoch = self.config.steps_per_epoch

      assert shaped_corpus.ndim     == 2, "corpus dim: {}".format(shaped_corpus.shape)
      assert shaped_corpus.shape[1] == sequence_length, "Dim 1 shape mismatch: {}, target: {}".format(encoded_corpus.shape[1], sequence_length)

      l.getLogger().info("{} kernels were rejected (larger than sequence_length)".format(initial_length - reduced_length))
      l.getLogger().info(
        "Loaded corpus of shape {} multiplied by dupe factor: {} in {} ms.".format(
                  shaped_corpus.shape,
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

      # shaped_corpus = np.split(clipped_corpus, batch_size * self.steps_per_epoch * dupe_factor, 0)
      shaped_corpus = np.split(clipped_corpus, batch_size * self.steps_per_epoch, 0)

      np_corpus = np.asarray(shaped_corpus)
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

    return shaped_corpus

  def _maskCorpus(self,
                  corpus: np.array,
                  train_set: bool,
                  set_name: str,
                  config   = None,
                  )-> None:
    """
    Entrypoint function that inserts masks or holes to the corpus.

    Arguments:
      corpus: [num_datapoints, sequence_length], 
              where num_datapoints = num_batches * dupe_factor * batch_size
    Returns:
      The masked corpus
    """

    # Set-up self.dataset entry
    self.dataset[set_name] = {
      'pt_record': [],
      'txt'      : [],
    }

    # Set up max predictions
    if config is None:
      config = self.config
      max_predictions = self.training_opts.max_predictions_per_seq
    else:
      max_predictions = config.max_predictions_per_seq

    # Apply dupe factor in stages to avoid stressing RAM.
    # Limit has been set to 4GB.
    single_item_bytes = MaskSequence.estimatedSize(
      1, self.training_opts.sequence_length, self.training_opts.max_predictions_per_seq
    )
    corpus_bytes = single_item_bytes * len(corpus) + sys.getsizeof(corpus)
    max_dupe     = min(int((FLAGS.memory_limit * (1024**3)) / corpus_bytes), self.training_opts.dupe_factor)
    assert max_dupe != 0, "Increase RAM limit to fit corpus."

    iterations   = int(self.training_opts.dupe_factor / max_dupe)
    remaining    = self.training_opts.dupe_factor % max_dupe

    extended_corpus   = np.repeat(corpus, max_dupe, axis = 0)
    remaining_corpus   = np.repeat(corpus, remaining, axis = 0)

    l.getLogger().info("Estimated element size: {}. Dupe factor {} split into {} iterations of {} (plus {} remaining)".format(
        humanize.naturalsize(single_item_bytes), self.training_opts.dupe_factor, iterations, max_dupe, remaining
      )
    )
    pool = multiprocessing.Pool()
    distribution = None
    # Specify the desired masking routine
    if config.HasField("hole"):
      distribution = distributions.Distribution.FromHoleConfig(config.hole, self.cache.path, set_name)
      maskedSeq    = lambda c: pool.imap_unordered(
        functools.partial(_holeSequence,
                          train_set            = train_set,
                          max_predictions      = max_predictions,
                          pickled_distribution = pickle.dumps(distribution),
                          pickled_atomizer     = pickle.dumps(self.atomizer),
                          rngen                = self.rngen, 
                          use_start_end        = self.config.use_start_end,
                          training_opts        = self.training_opts,
                          ),
        c
      )
    elif config.HasField("mask"):
      maskedSeq    = lambda c: pool.imap_unordered(
        functools.partial(_maskSequence,
                          train_set          = train_set,
                          max_predictions    = max_predictions,
                          pickled_atomizer   = pickle.dumps(self.atomizer),
                          rngen              = self.rngen, 
                          training_opts      = self.training_opts,
                          config             = config,
                          ),
        c
      )
    else:
      raise AttributeError("target predictions can only be mask or hole {}".format(self.config))

    ## Core loop of masking.
    masked_corpus = []
    with progressbar.ProgressBar(max_value = len(corpus) * self.training_opts.dupe_factor) as bar:
      kernel_idx = 0
      try:
        for iteration in range(iterations + 1):
          masked_corpus = []
          # Select between normal iterations or dupe factor residual and shuffle
          if iteration != iterations:
            multiproc_corpus = maskedSeq(extended_corpus)
            if self.training_opts.shuffle_corpus_contentfiles_between_epochs:
              self.rngen.shuffle(extended_corpus)
          elif remaining != 0:
            multiproc_corpus = maskedSeq(remaining_corpus)
            if self.training_opts.shuffle_corpus_contentfiles_between_epochs:
              self.rngen.shuffle(remaining_corpus)
          else:
            continue

          # Do parallel masking over corpus
          for kernel in multiproc_corpus:
            masked_corpus.append(kernel)
            bar.update(kernel_idx)
            kernel_idx += 1
            if kernel_idx == 1:
              MaskSequence.LogBatchTelemetry(
                self.training_opts.batch_size, self.training_opts.sequence_length,
                max_predictions, self.steps_per_epoch, self.num_epochs
                )

          # write masked_corpus before flushing the list
          self.dataset[set_name]['pt_record'].append(
            self.cache.path / "{}_{}.pt_record".format(set_name, iteration)
            )
          self.dataset[set_name]['txt'].append(
            self.cache.path / "{}_{}.txt".format(set_name, iteration)
            )
          self._saveCorpusRecord({
              'corpus'   : masked_corpus,
              'pt_record': self.cache.path / "{}_{}.pt_record".format(set_name, iteration),
              'txt'      : self.cache.path / "{}_{}.txt".format(set_name, iteration)
            })
        pool.close()
      except KeyboardInterrupt as e:
        pool.terminate()
        raise e
      except Exception as e:
        pool.terminate()
        raise e

    if distribution:
      distribution.plot()
    return

  def InitSampleBatch(self) -> None:
    """
      Initializes data_generator for inference.
      self.sampleBatch is initialized with sampler.encoded_start_text
    """
    if not self.sampler.isFixedStr:
      try:
        start_text = next(self.tfRecordSampler)[:self.sampler.sequence_length]
      except StopIteration:
        l.getLogger().warn("Repeating iterator on dataset...")
        self.tfRecordSampler = self.tfRecordSampleGenerator()
        try:
          start_text = next(self.tfRecordSampler)[:self.sampler.sequence_length]
        except Exception as e:
          raise e
      self.sampler.setStartText(self.atomizer.DeatomizeIndices(start_text))
      self.sampler.Specialize(self.atomizer)
    
    assert self.sampler.sequence_length <= self.max_position_embeddings, "Sampler sequence length exceeds max position embeddings."
    input_sample = self.sampler.encoded_start_text
    assert np.ndim(input_sample) == 1, "Input samples have to be one-dimensional. {} given.".format(input_sample.shape)

    target_idx = np.where(np.in1d(input_sample, [self.atomizer.maskToken, self.atomizer.holeToken]))[0]
    assert len(target_idx) != 0, "No target prediction in sample text"

    num_masks = np.count_nonzero(input_sample == self.atomizer.maskToken)
    num_holes = np.count_nonzero(input_sample == self.atomizer.holeToken)
    num_targets = num_masks + num_holes

    padded_sample = self._padToMaxPosition(input_sample)
    padded_sample = padded_sample[:self.sampler.sequence_length]
    self.sampleBatch   = np.repeat(padded_sample[None, :], self.sampler.batch_size, axis = 0)
    self.sampleIndices = [[[] for i in range(num_targets)] for j in range(self.sampler.batch_size)]
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
      mask_id_index     = 0
      closed_hole_index = 0
      for idx, token in enumerate(input_ids[batch_idx]):
        if   token == self.atomizer.maskToken:
          mt = masked_lm_ids[batch_idx][mask_id_index]
          if mt == self.atomizer.maskToken or mt == self.atomizer.holeToken:
            continue
          if len(self.sampleIndices[batch_idx][mask_id_index]) > 0:
            while(self.sampleIndices[batch_idx][mask_id_index + closed_hole_index][-1]) == self.atomizer.endholeToken:
              closed_hole_index += 1
          self.sampleIndices[batch_idx][mask_id_index + closed_hole_index].append(mt)
          mask_id_index += 1
          batch.append(mt)
        elif token == self.atomizer.holeToken:
          mt = masked_lm_ids[batch_idx][mask_id_index]
          if mt == self.atomizer.maskToken or mt == self.atomizer.holeToken:
            continue
          if len(self.sampleIndices[batch_idx][mask_id_index]) > 0:
            while(self.sampleIndices[batch_idx][mask_id_index + closed_hole_index][-1]) == self.atomizer.endholeToken:
              closed_hole_index += 1
          self.sampleIndices[batch_idx][mask_id_index + closed_hole_index].append(mt)
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
      if self.sampler.sequence_length > len(batch):
        l.getLogger().warn("Cropped {} tokens from sample batch".format(self.sampler.sequence_length - len(batch)))
      batch = batch[:self.sampler.sequence_length]
      updated_sequence.append(batch)

    self.sampleBatch = np.asarray(updated_sequence)
    return self.sampleBatch, self.sampleIndices

  def _saveCorpusRecord(self, masked_corpus: typing.Dict) -> None:
    """Converts corpus nparrays to tf Features and stores corpus to TfRecord"""

    torch.save(
      [{k: torch.from_numpy(v)} for x in masked_corpus['corpus'] for k, v in x.items()], 
      masked_corpus['pt_record']
    )
    if FLAGS.write_text_dataset:
      with open(masked_corpus['txt'], 'w') as file_writer:
        for instance in masked_corpus['corpus']:
          file_writer.write("'seen_in_training': {}\n'original_input': {}\n'input_ids': {}\n'input_mask': {}\n'position_ids': {}\n'mask_labels': {}\n'masked_lm_lengths': {}\n'next_sentence_label': {}\n\n"
                              .format((True if instance['seen_in_training'] == 1 else False),
                                      self.atomizer.DeatomizeIndices(instance['original_input'], ignore_token = self.atomizer.padToken),
                                      self.atomizer.DeatomizeIndices(instance['input_ids'], ignore_token = self.atomizer.padToken),
                                      instance['input_mask'],
                                      instance['position_ids'],
                                      instance['mask_labels'],
                                      instance['masked_lm_lengths'],
                                      instance['next_sentence_label']
                                    )
                              )
    l.getLogger().info("Wrote {} instances ({} batches of {} datapoints) to {}"
                 .format(len(masked_corpus['corpus']), self.steps_per_epoch, self.training_opts.batch_size, masked_corpus['pt_record']))
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
                              (self.max_position_embeddings - len(input_sample)), dtype = np.int64)
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