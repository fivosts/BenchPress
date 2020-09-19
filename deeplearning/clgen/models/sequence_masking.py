"""Core algorithm of sequence masking"""
import sys
import random
import typing
import copy
import humanize
import pickle
import numpy as np

from deeplearning.clgen.util import distributions
from deeplearning.clgen.util.tf import tf
from eupy.native import logger as l


class MaskSequence(typing.NamedTuple):
  """
  Tuple representation of a single MaskLM Instance. 
  This is not batch! generateTfDataset applies native batching,
  so this class represents a single instance!
  """

  seen_in_training     : np.int32
  original_input       : np.array
  input_ids            : np.array
  input_mask           : np.array
  masked_lm_positions  : np.array
  masked_lm_ids        : np.array
  masked_lm_weights    : np.array
  masked_lm_lengths    : np.array
  next_sentence_label  : np.int32

  @staticmethod
  def tfTypes():
    return (tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.float32, tf.int32, tf.int32)

  @staticmethod
  def npTypes():
    return (np.int32, np.int32, np.int32, np.int32, np.int32, np.int32, np.float32, np.int32, np.int32)

  @staticmethod
  def tfShapes(batch_size, sequence_length, max_position_embeddings = None):
    return (tf.TensorShape([batch_size, 1]),
            tf.TensorShape([batch_size, sequence_length]),
            tf.TensorShape([batch_size, sequence_length]),
            tf.TensorShape([batch_size, sequence_length]),
            tf.TensorShape([batch_size, max_position_embeddings]),
            tf.TensorShape([batch_size, max_position_embeddings]),
            tf.TensorShape([batch_size, max_position_embeddings]),
            tf.TensorShape([batch_size, max_position_embeddings]),
            tf.TensorShape([batch_size, 1]),
           )

def tfHoleSequence(seq: np.array,
                  train_set: bool,
                  max_predictions: int,
                  pickled_distribution: distributions.Distribution,
                  pickled_atomizer,
                  rngen: random.Random,
                  use_start_end: bool,
                  training_opts,
                  ) -> MaskSequence:
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
  offset_idxs       = np.zeros(len(seq), dtype = np.int32)
  # Set with all candidate_indexes that have been holed.
  visited_indices   = set()
  # Total masks placed so far.
  total_predictions = 0
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
  masked_lm_positions, masked_lm_ids, masked_lm_weights, masked_lm_lengths = [], [], [], []

  input_mask = np.ones(len(seq), dtype = np.int32)
  if atomizer.padToken in input_ids:
    first_pad_index = input_ids.index(atomizer.padToken)
    input_mask[first_pad_index:] = 0
    # Check that the pad index is likely correct.
    assert input_ids[first_pad_index] == atomizer.padToken, "{}".format(input_ids)
    assert input_ids[first_pad_index - 1] != atomizer.padToken

  seen_in_training    = np.int32(1 if train_set else 0)
  next_sentence_label = np.int32(0)
  """
    Related to next_sentence_label: Fix it to 0 for now, as no next_sentence prediction
    is intended on kernels. In any other case, check bert's create_instances_from_document
    to see how next_sentence_labels are calculated.
    Setting this to 0 means that next sentence is NOT random.
    Note that if next_sentence prediction is to be embedded, [SEP] token has to be added.    
  """
  if len(masked_lms) == 0:
    l.getLogger().warn("No HOLE added to datapoint. Increase probability of hole occuring.")
  for p in masked_lms:
    if p.pos_index < len(seq):
      """
        Adding holes can increase or decrease the length of the original sequence.
        It is important in the end, to end up with an input sequence compatible
        with the model's sequence length, i.e. len(seq). If any mask is found 
        beyond that point, will have to be rejected.
      """
      masked_lm_positions.append(p.pos_index)
      masked_lm_ids.append(p.token_id)
      masked_lm_weights.append(1.0)
      masked_lm_lengths.append(p.hole_length)
  num_holes = len(masked_lm_positions)
  while len(masked_lm_positions) < training_opts.max_predictions_per_seq:
      masked_lm_positions.append(0)
      masked_lm_ids.append(atomizer.padToken)
      masked_lm_weights.append(0.0)
      masked_lm_lengths.append(-1)

  assert (input_ids[:len(seq)].count(atomizer.holeToken) == num_holes,
    "Number of targets {} does not correspond to hole number in final input sequence: {}"
    .format(num_holes, input_ids[:len(seq)].count(atomizer.holeToken))
  )
  return MaskSequence(seen_in_training, seq,
                      np.asarray(input_ids[:len(seq)]), input_mask,
                      np.asarray(masked_lm_positions),  np.asarray(masked_lm_ids), 
                      np.asarray(masked_lm_weights),    np.asarray(masked_lm_lengths),
                      next_sentence_label 
                      ), hole_length_list

def tfMaskSequence(seq: np.array,
                  train_set: bool,
                  max_predictions: int,
                  pickled_atomizer,
                  rngen: random.Random,
                  training_opts,
                  config,
                  ) -> MaskSequence:
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

  masked_lm_positions, masked_lm_ids, masked_lm_weights, masked_lm_lengths = [], [], [], []

  input_mask = np.ones(len(seq), dtype = np.int32)
  if atomizer.padToken in input_ids:
    input_mask[input_ids.index(atomizer.padToken):] = 0

  seen_in_training    = np.int32(1 if train_set else 0)
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
    masked_lm_lengths.append(1)
  while len(masked_lm_positions) < training_opts.max_predictions_per_seq:
      masked_lm_positions.append(0)
      masked_lm_ids.append(atomizer.padToken)
      masked_lm_weights.append(0.0)
      masked_lm_lengths.append(-1)

  return MaskSequence(seen_in_training, seq,
                      np.asarray(input_ids),           input_mask,
                      np.asarray(masked_lm_positions), np.asarray(masked_lm_ids), 
                      np.asarray(masked_lm_weights),   np.asarray(masked_lm_lengths),
                      next_sentence_label
                      )

def torchHoleSequence(seq: np.array,
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

def torchMaskSequence(seq: np.array,
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
