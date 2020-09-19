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
  # Workaround for hole length distributions in multiprocessing environment.
  hole_length_list = []
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
    hole_length_list.append(hole_length)
    
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

  seen_in_training     = np.ones([1], dtype = np.int64) if train_set else np.zeros([1], dtype = np.int64)
  next_sentence_labels = np.zeros([1], dtype = np.int64)
  """
    Related to next_sentence_labels: Fix it to 0 for now, as no next_sentence prediction
    is intended on kernels. In any other case, check bert's create_instances_from_document
    to see how next_sentence_labels are calculated.
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

  return ({
      'seen_in_training'    : seen_in_training,
      'original_input'      : seq,
      'input_ids'           : np.asarray(input_ids[:len(seq)], dtype = np.int64),
      'input_mask'          : input_mask,
      'position_ids'        : np.arange(len(seq), dtype = np.int64),
      'mask_labels'         : mask_labels,
      'masked_lm_lengths'   : masked_lm_lengths,
      'next_sentence_labels': next_sentence_labels,
    }, hole_length_list)

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

  seen_in_training     = np.ones([1], dtype = np.int64) if train_set else np.zeros([1], dtype = np.int64)
  next_sentence_labels = np.zeros([1], dtype = np.int64)
  ## Related to next_sentence_labels: Fix it to 0 for now, as no next_sentence prediction
  ## is intended on kernels. In any other case, check bert's create_instances_from_document
  ## to see how next_sentence_labels are calculated.
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

  return ({
      'seen_in_training'    : seen_in_training,
      'original_input'      : seq,
      'input_ids'           : np.asarray(input_ids[:len(seq)], dtype = np.int64),
      'input_mask'          : input_mask,
      'position_ids'        : np.arange(len(seq), dtype = np.int64),
      'mask_labels'         : mask_labels,
      'masked_lm_lengths'   : masked_lm_lengths,
      'next_sentence_labels': next_sentence_labels,
    }, [])

class torchLMDataGenerator(lm_data_generator.MaskLMDataGenerator):
  def __init__(self):
    super(self, torchLMDataGenerator).__init__("pt_record")
    self.dataloader = None # Extra
    return

  @classmethod
  def TrainMaskLMBatchGenerator(cls,
                               corpus: "corpuses.Corpus",
                               training_opts: model_pb2.TrainingOptions,
                               cache_path,
                               ) -> "data_generator.MaskLMBatchGenerator":
    """Initializes data generator for training."""
    d = super(self, torchLMDataGenerator).TrainMaskLMBatchGenerator(
              corpus, training_opts, cache_path
        )
    d.train_dataloader() # Extra
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
    d = super(self, torchLMDataGenerator).SampleMaskLMBatchGenerator(
              sampler, atomizer, seed, max_position_embeddings,
        )
    d.predict_dataloader() # Extra
    return d

  # Extra
  def train_dataloader(self) -> None:
    """Pytorch dataloader that assembles all dataset files into a single-mapped dataset."""
    dataset = torch.utils.data.ConcatDataset(
                [torch.load(x) for x in self.dataset['train_dataset']['file']]
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

  # Extra
  def eval_dataloaders(self):
    for set_name in self.dataset:
      dataset = torch.utils.data.ConcatDataset(
                  [torch.load(x) for x in self.dataset[set_name]['file']]
                )
      dataloader = torch.utils.data.dataloader.DataLoader(
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
      yield set_name, dataloader

  # Extra
  def predict_dataloader(self):

    if self.sampler.isFixedStr:
      input_sample = self.sampler.encoded_start_text
      assert np.ndim(input_sample) == 1, "Input samples have to be one-dimensional. {} given.".format(input_sample.shape)

      target_idx = np.where(np.in1d(input_sample, [self.atomizer.maskToken, self.atomizer.holeToken]))[0]
      assert len(target_idx) != 0, "No target prediction in sample text"
      num_targets = (np.count_nonzero(input_sample == self.atomizer.maskToken) + 
                     np.count_nonzero(input_sample == self.atomizer.holeToken))

      seen_in_training     = 0
      original_input       = np.full((self.sampler.sequence_length), 0, dtype = np.int64)
      input_ids            = self._padToMaxPosition(input_sample)[:self.sampler.sequence_length]
      input_mask           = np.concatenate([
                                  np.ones(len(input_sample), dtype = np.int64),
                                  np.zeros(len(input_ids) - len(input_sample), dtype = np.int64)
                                ])      
      position_ids         = np.arange(self.sampler.sequence_length, dtype = np.int64)
      mask_labels          = np.full((self.sampler.sequence_length), -100, dtype = np.int64)
      masked_lm_lengths    = np.full((self.sampler.sequence_length), -1, dtype = np.int64)
      next_sentence_labels = 0
      raise ValuError("Check here that the metrics are correct.")
      sample_element = {
        'seen_in_training'    : seen_in_training,
        'original_input'      : original_input,
        'input_ids'           : input_ids,
        'input_mask'          : input_mask,
        'position_ids'        : position_ids,
        'mask_labels'         : mask_labels,
        'masked_lm_lengths'   : masked_lm_lengths,
        'next_sentence_labels': next_sentence_labels,
      }
      dataset = [{k: torch.from_numpy(v) for (k, v) in sample_element.items()}]
    else:
      if self.sampler.config.HasField("train_set"):
        sampledDataset = "train_dataset"
      elif self.sampler.config.HasField("validation_set"):
        sampledDataset = "validation_dataset"
      elif self.sampler.config.HasField("sample_set"):
        sampledDataset = "pred_{}_{}".format(
          self.sampler.config.sample_set.max_predictions_per_seq,
          "mask" if self.sampler.config.sample_set.HasField("mask") 
                 else "hole_{}".format(self.sampler.config.sample_set.hole.hole_length)
        )
      path_list = glob.glob(str(self.cache.path / "{}_*.{}".format(sampledDataset, self.file_extension)))
      if len(path_list) == 0:
        # Config dataset
        raise FileNotFoundError
        shaped_corpus = self.createCorpus()
        self.configValidationSets(self.sampler.config.sample_set, shaped_corpus)
      dataset = torch.utils.data.ConcatDataset(
                  [torch.load(x) for x in path_list]
                )
    self.dataloader = torch.utils.data.dataloader.DataLoader(
      dataset    = dataset,
      batch_size = 1,
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

  def _saveCorpusRecord(self, masked_corpus: typing.Dict) -> None:
    """Converts corpus nparrays to tf Features and stores corpus to TfRecord"""

    torch.save(
      [{k: torch.from_numpy(v) for (k, v) in inst.items()} for inst in masked_corpus['corpus']],
      masked_corpus['file']
    )
    if FLAGS.write_text_dataset:
      with open(masked_corpus['txt'], 'w') as file_writer:
        for instance in masked_corpus['corpus']:
          file_writer.write("'seen_in_training': {}\n'original_input': {}\n'input_ids': {}\n'input_mask': {}\n'position_ids': {}\n'mask_labels': {}\n'masked_lm_lengths': {}\n'next_sentence_labels': {}\n\n"
                              .format((True if instance['seen_in_training'] == 1 else False),
                                      self.atomizer.DeatomizeIndices(instance['original_input'], ignore_token = self.atomizer.padToken),
                                      self.atomizer.DeatomizeIndices(instance['input_ids'], ignore_token = self.atomizer.padToken),
                                      instance['input_mask'],
                                      instance['position_ids'],
                                      instance['mask_labels'],
                                      instance['masked_lm_lengths'],
                                      instance['next_sentence_labels']
                                    )
                              )
    l.getLogger().info("Wrote {} instances ({} batches of {} datapoints) to {}"
                 .format(len(masked_corpus['corpus']), self.steps_per_epoch, self.training_opts.batch_size, masked_corpus['file']))
    return

  def estimatedSize(batch_size, sequence_length, max_predictions_per_seq):
    """
    Calculate estimated size of single training example as a dictionary.
    """
    return (
      2 * np.zeros([batch_size, 1], dtype = np.int64).nbytes + 
      5 * np.zeros([batch_size, sequence_length], dtype = np.int64).nbytes +
      1 * np.zeros([batch_size, max_predictions_per_seq], dtype = np.int64).nbytes
      )

  def LogBatchTelemetry(self,
                        batch_size: int,
                        sequence_length: int,
                        max_predictions_per_seq: int,
                        steps_per_epoch: int,
                        num_epochs: int,
                        ) -> None:
    """Log analytics about the batch."""
    l.getLogger().info(
      "Memory: {} per batch, {} per epoch, {} total.".format(
              humanize.naturalsize(self.estimatedSize(1, sequence_length, max_predictions_per_seq) * batch_size, binary = True),
              humanize.naturalsize(self.estimatedSize(1, sequence_length, max_predictions_per_seq) * batch_size * steps_per_epoch, binary = True),
              humanize.naturalsize(self.estimatedSize(1, sequence_length, max_predictions_per_seq) * batch_size * steps_per_epoch * num_epochs, binary = True),
          )
    )
