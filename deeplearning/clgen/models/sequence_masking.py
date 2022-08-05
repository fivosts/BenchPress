"""Core algorithm of sequence masking"""
import sys
import typing
import copy
import humanize
import pickle
import numpy as np
import progressbar

from deeplearning.benchpress.util import distributions
from deeplearning.benchpress.util.tf import tf
from deeplearning.benchpress.util import logging as l

class tfSequence(typing.NamedTuple):
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

## Tuple representation of mask id/position/hole_length for easy sorting
class MaskedLmInstance():
  def __init__(self, 
               pos_index: int, 
               token_id: int, 
               hole_length: int,
               extend_left: bool,
               ):
    self.pos_index   = pos_index
    self.token_id    = token_id
    self.hole_length = hole_length
    self.extend_left = extend_left

def MPHoleSequence(seq: np.array,
                   train_set: bool,
                   max_predictions: int,
                   pickled_distribution: distributions.Distribution,
                   pickled_tokenizer,
                   training_opts,
                   is_torch: bool,
                   repair_locations: typing.List[int] = None,
                   ) -> typing.Tuple[
                          typing.Union[typing.Dict[str, np.array], tfSequence],
                          typing.List[MaskedLmInstance],
                        ]:
  """
  Inserts hole tokens to a given sequence.

  If repair_locations is set, then algorithm places holes over syntactic errors
  for the model to repair them. Default is None, where hole-d indices are randomly
  selected.

  This function is compatible for multiprocessing. There is an optimized single-core
  version below.
  """
  assert seq.ndim == 1, "Input for masking must be single-dimension array."

  # Unpack tokenizer and sampler
  distribution = pickle.loads(pickled_distribution)
  tokenizer     = pickle.loads(pickled_tokenizer)
  use_start_end = True if seq[0] == tokenizer.startToken else False

  # Actual length represents the sequence length before pad begins
  if use_start_end:
    actual_length   = np.where(seq == tokenizer.endToken)[0][0]
    last_elem       = actual_length
  elif tokenizer.padToken in seq:
    actual_length   = np.where(seq == tokenizer.padToken)[0][0]
    last_elem       = actual_length - 1
  else:
    actual_length   = len(seq)
    last_elem       = actual_length - 1

  # total tokens to add in holes.
  # No more than max_predictions_per_seq (or otherwise specified), no less than actual seq length x the probability of hiding a token
  holes_to_predict  = min(max_predictions,
                         max(1, int(round(actual_length * training_opts.masked_lm_prob))))

  extend_left = True if np.random.RandomState().randint(0, 2) == 1 else False
  input_ids   = list(np.copy(seq))
  # List of (seq_idx, token_id, hole_length) tuples
  masked_lms        = []
  # Offset array. Indices represent elements in the initial array (seq)
  # Values of indices represent current offset position in processed array (input_ids).
  offset_idxs       = np.zeros(len(seq), dtype = np.int64)
  # Set with all candidate_indexes that have been holed.
  visited_indices   = set()
  # Total masks placed so far.
  total_predictions = 0
  while total_predictions < holes_to_predict:
    if repair_locations:
      pos_index = repair_locations[np.random.RandomState().randint(0, len(repair_locations))]
    else:
      pos_index = np.random.RandomState().randint(0, actual_length) # Fixed seed doesn't work!
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
    elif input_ids[input_id_idx] in {tokenizer.startToken, tokenizer.endToken}:
      # Do not target [START] or [END] token
      continue

    assert (input_ids[input_id_idx] == seq[pos_index], 
            "Original and offset-ted sequence have misaligned tokens: {}, {}"
            .format(seq[pos_index], input_ids[input_id_idx]))

    # Sampled number from distribution to represent the actual hole length
    hole_length = distribution.sample(actual_length)

    # Increase hole length a little bit, if too many empty holes have pushed rightmost elements
    # over the edge.
    while last_elem + offset_idxs[last_elem] + 1 - hole_length >= len(seq):
      hole_length += 1

    # Inside range, make sure hole length does not run over input_id_idx bounds
    # This may be redundant given the next for loop
    if extend_left:
      hole_length = min(hole_length, input_id_idx)
    else:
      hole_length = min(hole_length, (last_elem + offset_idxs[last_elem]) - input_id_idx)
    
    # Confirm there is no conflict with another hole, further down the sequence.
    for i in range(hole_length):
      if extend_left:
        if (input_ids[input_id_idx - i] == tokenizer.holeToken
         or input_ids[input_id_idx - i] == tokenizer.startToken
         or input_ids[input_id_idx - i] == tokenizer.endToken
         # or input_id_idx - i == 0
         ):
          hole_length = i
          break
      else:
        if (input_ids[input_id_idx + i] == tokenizer.holeToken
         or input_ids[input_id_idx + i] == tokenizer.startToken
         or input_ids[input_id_idx + i] == tokenizer.endToken
         # or input_id_idx + i == len(input_ids)
         ):
          hole_length = i
          break

    if offset_idxs[last_elem] + 1 - hole_length >= len(seq):
      # This hole can't help but explode the sequence. Go find a new position.
      continue

    assert hole_length >= 0, "hole length is negative: {}".format(hole_length)

    pos_index  -= hole_length - 1 if hole_length != 0 and extend_left else 0
    input_id_idx = pos_index + offset_idxs[pos_index]
    
    # Target token for classifier is either the first token of the hole, or endholeToken if hole is empty
    target = input_ids[input_id_idx] if hole_length > 0 else tokenizer.endholeToken
    input_ids = input_ids[:input_id_idx] + [tokenizer.holeToken] + input_ids[input_id_idx + hole_length:]

    # Store position index, and after making all masks, update with updated offset array
    masked_lms.append(MaskedLmInstance(
        pos_index = pos_index, token_id = target, hole_length = hole_length, extend_left = extend_left
      )
    )
    # Adjust the offset of all affected tokens, from pos_index and after.
    offset_idxs[pos_index + 1:] += 1 - hole_length
    total_predictions           += max(1, hole_length)
    visited_indices.update(range(pos_index, pos_index + hole_length))

  hole_analytics = copy.deepcopy(masked_lms)

  # Now update the entries with offset index.
  for lm in masked_lms:
    prev_index = lm.pos_index
    lm.pos_index = lm.pos_index + offset_idxs[lm.pos_index]
    assert input_ids[lm.pos_index] == tokenizer.holeToken, "{}".format(lm.hole_length)

  while len(input_ids) < len(seq):
    input_ids.append(tokenizer.padToken)
  masked_lms = sorted(masked_lms, key=lambda x: x.pos_index)

  input_mask = np.ones(len(seq), dtype = np.int64)
  if tokenizer.padToken in input_ids:
    first_pad_index = input_ids.index(tokenizer.padToken)
    input_mask[first_pad_index:] = 0
    # Check that the pad index is likely correct.
    assert input_ids[first_pad_index] == tokenizer.padToken, "{}".format(input_ids)
    assert input_ids[first_pad_index - 1] != tokenizer.padToken

  """
    Related to next_sentence_labels: Fix it to 0 for now, as no next_sentence prediction
    is intended on kernels. In any other case, check bert's create_instances_from_document
    to see how next_sentence_labels are calculated.
    Setting this to 0 means that next sentence is NOT random.
    Note that if next_sentence prediction is to be embedded, [SEP] token has to be added.    
  """
  if len(masked_lms) == 0:
    l.logger().warn("No HOLE added to datapoint. Increase probability of hole occuring.")

  if is_torch:
    seen_in_training     = np.int64([1] if train_set else [0])
    next_sentence_labels = np.int64([0])

    masked_lm_lengths = np.full(holes_to_predict, -1, dtype = np.int64)
    mask_labels = np.full(len(seq), -100, dtype = np.int64)
    ind = 0
    for p in masked_lms:
      if p.pos_index < len(seq):
        mask_labels[p.pos_index] = p.token_id
        masked_lm_lengths[ind]   = p.hole_length
        ind += 1

    return {
        'seen_in_training'    : seen_in_training,
        'original_input'      : seq,
        'input_ids'           : np.asarray(input_ids[:len(seq)], dtype = np.int64),
        'input_mask'          : input_mask,
        'position_ids'        : np.arange(len(seq), dtype = np.int64),
        'mask_labels'         : mask_labels,
        'masked_lm_lengths'   : masked_lm_lengths,
        'next_sentence_labels': next_sentence_labels,
      }, hole_analytics
  
  else: # TF 1.X, 2.[0-2]
    
    seen_in_training    = np.int32(1 if train_set else 0)
    next_sentence_label = np.int32(0)
    masked_lm_positions, masked_lm_ids, masked_lm_weights, masked_lm_lengths = [], [], [], []
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
        masked_lm_ids.append(tokenizer.padToken)
        masked_lm_weights.append(0.0)
        masked_lm_lengths.append(-1)

    assert (input_ids[:len(seq)].count(tokenizer.holeToken) == num_holes,
      "Number of targets {} does not correspond to hole number in final input sequence: {}"
      .format(num_holes, input_ids[:len(seq)].count(tokenizer.holeToken))
    )
    return tfSequence(seen_in_training, seq,
                        np.asarray(input_ids[:len(seq)]), input_mask,
                        np.asarray(masked_lm_positions),  np.asarray(masked_lm_ids),
                        np.asarray(masked_lm_weights),    np.asarray(masked_lm_lengths),
                        next_sentence_label
                        ), hole_analytics

def MPMaskSequence(seq: np.array,
                   train_set: bool,
                   max_predictions: int,
                   pickled_tokenizer,
                   training_opts,
                   config,
                   is_torch: bool,
                   ) -> typing.Dict:
  """
  Inserts masks to a given sequence.

  This function is compatible for multiprocessing. There is an optimized single-core
  version below.
  """
  assert seq.ndim == 1, "Input for masking must be single-dimension array."

  ## Tuple representation of mask id/position for easy sorting
  class MaskedLmInstance(typing.NamedTuple):
    pos_index: int
    token_id: int

  # Unpack tokenizer
  tokenizer = pickle.loads(pickled_tokenizer)

  use_start_end = True if seq[0] == tokenizer.startToken else False
  # Actual length represents the sequence length before pad begins
  if use_start_end:
    actual_length   = np.where(seq == tokenizer.endToken)[0][0]
  elif tokenizer.padToken in seq:
    actual_length   = np.where(seq == tokenizer.padToken)[0][0]
  else:
    actual_length   = len(seq)

  candidate_indexes = np.arange(actual_length)
  np.random.RandomState().shuffle(candidate_indexes)

  masks_to_predict = min(max_predictions,
                         max(1, int(round(actual_length * training_opts.masked_lm_prob))))
  input_ids = list(np.copy(seq))
  masked_lms = []

  for pos_index in candidate_indexes:
    if len(masked_lms) >= masks_to_predict:
      break

    if config.mask.random_placed_mask:
      # 80% of the time, replace with [MASK]
      if np.random.RandomState().random() < 0.8:
        input_ids[pos_index] = tokenizer.maskToken
      else:
        # 10% of the time, keep original
        if np.random.RandomState().random() < 0.5:
          pass
        # 10% of the time, replace with random word
        else:
          random_token = np.random.RandomState().randint(0, tokenizer.vocab_size)
          while any(tokenizer.vocab[t] == random_token for (idx, t) in tokenizer.metaTokens.items()):
            random_token = np.random.RandomState().randint(0, tokenizer.vocab_size)
          input_ids[pos_index] = np.random.RandomState().randint(0, tokenizer.vocab_size)
    else:
      if np.random.RandomState().random() < 0.8:
        input_ids[pos_index] = tokenizer.maskToken

    masked_lms.append(MaskedLmInstance(pos_index=pos_index, token_id=seq[pos_index]))

  assert len(masked_lms) <= masks_to_predict
  masked_lms = sorted(masked_lms, key=lambda x: x.pos_index)

  input_mask = np.ones(len(seq), dtype = np.int64)
  if tokenizer.padToken in input_ids:
    input_mask[input_ids.index(tokenizer.padToken):] = 0

  ## Related to next_sentence_labels: Fix it to 0 for now, as no next_sentence prediction
  ## is intended on kernels. In any other case, check bert's create_instances_from_document
  ## to see how next_sentence_labels are calculated.
  ## Setting this to 0 means that next sentence is NOT random.
  ## Note that if next_sentence prediction is to be embedded, [SEP] token has to be added.

  if is_torch:
    seen_in_training     = np.int64([1] if train_set else [0])
    next_sentence_labels = np.int64([0])

    masked_lm_lengths = np.full(masks_to_predict, -1, dtype = np.int64)
    mask_labels = np.full(len(seq), -100, dtype = np.int64)
    ind = 0
    for p in masked_lms:
      if p.pos_index < len(seq):
        mask_labels[p.pos_index] = p.token_id
        masked_lm_lengths[ind]   = 1
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

  else: # TF 1.X, 2.[0-2]

    masked_lm_positions, masked_lm_ids, masked_lm_weights, masked_lm_lengths = [], [], [], []
    seen_in_training    = np.int32(1 if train_set else 0)
    next_sentence_label = np.int32(0)
    for p in masked_lms:
      masked_lm_positions.append(p.pos_index)
      masked_lm_ids.append(p.token_id)
      masked_lm_weights.append(1.0)
      masked_lm_lengths.append(1)
    while len(masked_lm_positions) < training_opts.max_predictions_per_seq:
        masked_lm_positions.append(0)
        masked_lm_ids.append(tokenizer.padToken)
        masked_lm_weights.append(0.0)
        masked_lm_lengths.append(-1)

    return tfSequence(seen_in_training, seq,
                        np.asarray(input_ids),           input_mask,
                        np.asarray(masked_lm_positions), np.asarray(masked_lm_ids), 
                        np.asarray(masked_lm_weights),   np.asarray(masked_lm_lengths),
                        next_sentence_label
                        ), [], []

def HoleSequence(seq: np.array,
                 train_set: bool,
                 max_predictions: int,
                 masked_lm_prob: int,
                 distribution: distributions.Distribution,
                 tokenizer,
                 ) -> typing.Dict[str, np.array]:
  """
  Inserts hole tokens to a given sequence.
  Used for online training.
  """
  use_start_end = True if seq[0] == tokenizer.startToken else False

  # Actual length represents the sequence length before pad begins
  if use_start_end:
    actual_length   = np.where(seq == tokenizer.endToken)[0][0]
    last_elem       = actual_length
  elif tokenizer.padToken in seq:
    actual_length   = np.where(seq == tokenizer.padToken)[0][0]
    last_elem       = actual_length - 1
  else:
    actual_length   = len(seq)
    last_elem       = actual_length - 1

  # total tokens to add in holes.
  # No more than max_predictions_per_seq (or otherwise specified), no less than actual seq length x the probability of hiding a token
  holes_to_predict  = min(max_predictions,
                         max(1, int(round(actual_length * masked_lm_prob))))

  extend_left = True if np.random.RandomState().randint(0, 2) == 1 else False
  input_ids   = list(np.copy(seq))
  # List of (seq_idx, token_id, hole_length) tuples
  masked_lms        = []
  # Offset array. Indices represent elements in the initial array (seq)
  # Values of indices represent current offset position in processed array (input_ids).
  offset_idxs       = np.zeros(len(seq), dtype = np.int64)
  # Set with all candidate_indexes that have been holed.
  visited_indices   = set()
  # Total masks placed so far.
  total_predictions = 0
  while total_predictions < holes_to_predict:
    try:
      pos_index = np.random.RandomState().randint(0, actual_length) # Fixed seed doesn't work!
    except ValueError as e:
      l.logger().error(actual_length)
      l.logger().error(tokenizer.tokensToString(seq))
      raise e
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
    elif input_ids[input_id_idx] in {tokenizer.startToken, tokenizer.endToken}:
      # Do not target [START] or [END] token
      continue

    # Sampled number from distribution to represent the actual hole length
    hole_length = distribution.sample(actual_length)

    # Increase hole length a little bit, if too many empty holes have pushed rightmost elements
    # over the edge.
    while last_elem + offset_idxs[last_elem] + 1 - hole_length >= len(seq):
      hole_length += 1

    # Inside range, make sure hole length does not run over input_id_idx bounds
    # This may be redundant given the next for loop
    if extend_left:
      hole_length = min(hole_length, input_id_idx)
    else:
      hole_length = min(hole_length, (last_elem + offset_idxs[last_elem]) - input_id_idx)

    # Confirm there is no conflict with another hole, further down the sequence.
    for i in range(hole_length):
      if extend_left:
        if (input_ids[input_id_idx - i] == tokenizer.holeToken
         or input_ids[input_id_idx - i] == tokenizer.startToken
         or input_ids[input_id_idx - i] == tokenizer.endToken
         # or input_id_idx - i == 0
         ):
          hole_length = i
          break
      else:
        if (input_ids[input_id_idx + i] == tokenizer.holeToken
         or input_ids[input_id_idx + i] == tokenizer.startToken
         or input_ids[input_id_idx + i] == tokenizer.endToken
         # or input_id_idx + i == len(input_ids)
         ):
          hole_length = i
          break

    if offset_idxs[last_elem] + 1 - hole_length >= len(seq):
      # This hole can't help but explode the sequence. Go find a new position.
      continue

    assert hole_length >= 0, "hole length is negative: {}".format(hole_length)

    pos_index  -= hole_length - 1 if hole_length != 0 and extend_left else 0
    input_id_idx = pos_index + offset_idxs[pos_index]

    # Target token for classifier is either the first token of the hole, or endholeToken if hole is empty
    target = input_ids[input_id_idx] if hole_length > 0 else tokenizer.endholeToken
    input_ids = input_ids[:input_id_idx] + [tokenizer.holeToken] + input_ids[input_id_idx + hole_length:]

    # Store position index, and after making all masks, update with updated offset array
    masked_lms.append(MaskedLmInstance(
        pos_index = pos_index, token_id = target, hole_length = hole_length, extend_left = extend_left
      )
    )
    # Adjust the offset of all affected tokens, from pos_index and after.
    offset_idxs[pos_index + 1:] += 1 - hole_length
    total_predictions           += max(1, hole_length)
    visited_indices.update(range(pos_index, pos_index + hole_length))

  # Now update the entries with offset index.
  for lm in masked_lms:
    prev_index = lm.pos_index
    lm.pos_index = lm.pos_index + offset_idxs[lm.pos_index]

  while len(input_ids) < len(seq):
    input_ids.append(tokenizer.padToken)
  masked_lms = sorted(masked_lms, key=lambda x: x.pos_index)

  input_mask = np.ones(len(seq), dtype = np.int64)
  if tokenizer.padToken in input_ids:
    first_pad_index = input_ids.index(tokenizer.padToken)
    input_mask[first_pad_index:] = 0

  seen_in_training     = np.int64([1] if train_set else [0])
  next_sentence_labels = np.int64([0])
  masked_lm_lengths = np.full(holes_to_predict, -1, dtype = np.int64)
  mask_labels = np.full(len(seq), -100, dtype = np.int64)
  ind = 0
  for p in masked_lms:
    if p.pos_index < len(seq):
      mask_labels[p.pos_index] = p.token_id
      masked_lm_lengths[ind]   = p.hole_length
      ind += 1

  return {
      'seen_in_training'    : seen_in_training,
      'original_input'      : seq,
      'input_ids'           : np.asarray(input_ids[:len(seq)], dtype = np.int64),
      'input_mask'          : input_mask,
      'position_ids'        : np.arange(len(seq), dtype = np.int64),
      'mask_labels'         : mask_labels,
      'masked_lm_lengths'   : masked_lm_lengths,
      'next_sentence_labels': next_sentence_labels,
    }

def HoleSequenceSeqMasks(seq: np.array,
                         train_set: bool,
                         max_predictions: int,
                         masked_lm_prob: int,
                         distribution: distributions.Distribution,
                         tokenizer,
                         ) -> typing.Dict[str, np.array]:
  """
  Instead of a hole, place left context on the leftmost part,
  the right context on the rightmost part and all remaining
  stuff are the masks in the middle. When the actual to-predict
  sentence.

  This is PLDI Reviewer B's idea.
  """
  use_start_end = True if seq[0] == tokenizer.startToken else False

  # Actual length represents the sequence length before pad begins
  if use_start_end:
    actual_length   = np.where(seq == tokenizer.endToken)[0][0]
    last_elem       = actual_length
  elif tokenizer.padToken in seq:
    actual_length   = np.where(seq == tokenizer.padToken)[0][0]
    last_elem       = actual_length - 1
  else:
    actual_length   = len(seq)
    last_elem       = actual_length - 1

  # total tokens to add in holes.
  # No more than max_predictions_per_seq (or otherwise specified), no less than actual seq length x the probability of hiding a token
  holes_to_predict  = min(max_predictions,
                         max(1, int(round(actual_length * masked_lm_prob))))

  assert holes_to_predict == 1, "This mode only supports a single hole."

  extend_left = True if np.random.RandomState().randint(0, 2) == 1 else False
  input_ids   = list(np.copy(seq))
  # List of (seq_idx, token_id, hole_length) tuples
  masked_lms        = []
  # Set with all candidate_indexes that have been holed.
  visited_indices   = set()
  # Total masks placed so far.
  total_predictions = 0
  while total_predictions < holes_to_predict:
    pos_index = np.random.RandomState().randint(0, actual_length) # Fixed seed doesn't work!
    # Element in processed array can be found in its original index +/- offset
    if total_predictions >= holes_to_predict:
      break
    elif pos_index in visited_indices:
      # Do not target an index, already holed
      continue
    elif input_ids[pos_index] in {tokenizer.startToken, tokenizer.endToken}:
      # Do not target [START] or [END] token
      continue

    # Sampled number from distribution to represent the actual hole length
    hole_length = distribution.sample(actual_length)

    # Increase hole length a little bit, if too many empty holes have pushed rightmost elements
    # over the edge.
    while last_elem + 1 - hole_length >= len(seq):
      hole_length += 1

    # Inside range, make sure hole length does not run over pos_index bounds
    # This may be redundant given the next for loop
    if extend_left:
      hole_length = min(hole_length, pos_index)
    else:
      hole_length = min(hole_length, last_elem - pos_index)

    # Confirm there is no conflict with another hole, further down the sequence.
    for i in range(hole_length):
      if extend_left:
        if (input_ids[pos_index - i] == tokenizer.holeToken
         or input_ids[pos_index - i] == tokenizer.startToken
         or input_ids[pos_index - i] == tokenizer.endToken
         # or pos_index - i == 0
         ):
          hole_length = i
          break
      else:
        if (input_ids[pos_index + i] == tokenizer.holeToken
         or input_ids[pos_index + i] == tokenizer.startToken
         or input_ids[pos_index + i] == tokenizer.endToken
         # or pos_index + i == len(input_ids)
         ):
          hole_length = i
          break

    if 1 - hole_length >= len(seq):
      # This hole can't help but explode the sequence. Go find a new position.
      continue

    assert hole_length >= 0, "hole length is negative: {}".format(hole_length)

    pos_index -= hole_length - 1 if hole_length != 0 and extend_left else 0

    # Target token for classifier is either the first token of the hole, or endholeToken if hole is empty
    targets = input_ids[pos_index: pos_index + hole_length]

    lc = input_ids[:pos_index]
    rc = input_ids[pos_index + hole_length:actual_length+1]
    pad_len = len(seq) - len(lc) - len(rc) - len(targets)

    if pad_len == 0:
      if len(rc) > 1:
        # input_ids = input_ids[:-2] + [input_ids[-1]]
        input_ids = lc + [tokenizer.maskToken]*(len(targets) + pad_len + 1) + rc[:-2] + [rc[-1]]
        targets   += [tokenizer.endholeToken]
      else:
        targets[-1] = tokenizer.endholeToken
        input_ids = lc + [tokenizer.maskToken]*(len(targets) + pad_len) + rc
    else:
      input_ids = lc + [tokenizer.maskToken]*(len(targets) + pad_len) + rc
      targets   += [tokenizer.endholeToken] * pad_len

    # Store position index, and after making all masks, update with updated offset array
    masked_lms.append(MaskedLmInstance(
        pos_index = pos_index, token_id = targets, hole_length = hole_length, extend_left = extend_left
      )
    )
    # Adjust the offset of all affected tokens, from pos_index and after.
    total_predictions += max(1, hole_length)
    visited_indices.update(range(pos_index, pos_index + hole_length))

  assert len(input_ids) == len(seq), "Input sequence and sequence length mismatch: {} / {}, {}".format(len(input_ids), len(seq), tokenizer.tokensToString(input_ids))
  assert input_ids[0] == tokenizer.startToken, "{}".format(tokenizer.tokensToString(input_ids[0]))
  assert input_ids[-1] == tokenizer.endToken, "{}".format(tokenizer.tokensToString(input_ids[-1]))
  # Now update the entries with offset index.
  masked_lms = sorted(masked_lms, key=lambda x: x.pos_index)
  mask_labels = np.full(len(seq), -100, dtype = np.int64)
  for p in masked_lms:
    if p.pos_index < len(seq):
      for idx, tid in enumerate(p.token_id):
        mask_labels[p.pos_index + idx] = tid

  return {
      'seen_in_training'    : np.int64([1] if train_set else [0]),
      'original_input'      : seq,
      'input_ids'           : np.asarray(input_ids[:len(seq)], dtype = np.int64),
      'input_mask'          : np.ones(len(seq), dtype = np.int64),
      'position_ids'        : np.arange(len(seq), dtype = np.int64),
      'mask_labels'         : mask_labels,
      'masked_lm_lengths'   : np.int64([1]),
      'next_sentence_labels': np.int64([0]),
    }

def MaskedSeqToBlob(enc_text: np.array,
                    tokenizer,
                    sequence_length: int,
                    max_position_embeddings: int,
                    ):
  """
  Constructs training/sampling instance from plain input text.
  """
  input_sample = enc_text
  target_idx   = np.where(np.in1d(input_sample, [tokenizer.maskToken, tokenizer.holeToken]))[0]
  num_targets  = (np.count_nonzero(input_sample == tokenizer.maskToken) + 
                 np.count_nonzero(input_sample == tokenizer.holeToken))

  assert np.ndim(input_sample) == 1, "Input samples have to be one-dimensional. {} given.".format(input_sample.shape)
  # if tokenizer.requires_mask:
  #   assert len(target_idx)       != 0, "No target prediction in sample text"

  seen_in_training     = np.zeros([1], dtype = np.int32)
  original_input       = np.full((1), tokenizer.padToken, dtype = np.int64)
  input_ids            = np.concatenate([
                            input_sample, np.array([tokenizer.padToken] * (max_position_embeddings - len(input_sample)), dtype = np.int64)
                         ])[:sequence_length]
  input_mask           = np.concatenate([
                              np.ones(len(input_sample), dtype = np.int64),
                              np.zeros(len(input_ids) - len(input_sample), dtype = np.int64)
                            ])      
  position_ids         = np.arange(sequence_length, dtype = np.int64)
  mask_labels          = np.full((sequence_length), -100, dtype = np.int64)
  masked_lm_lengths    = np.full((1), -1, dtype = np.int64)
  next_sentence_labels = np.zeros([1], dtype = np.int32)
  return {
    'seen_in_training'    : seen_in_training,
    'original_input'      : original_input,
    'input_ids'           : input_ids,
    'input_mask'          : input_mask,
    'position_ids'        : position_ids,
    'mask_labels'         : mask_labels,
    'masked_lm_lengths'   : masked_lm_lengths,
    'next_sentence_labels': next_sentence_labels,
  }

def ExhaustiveHoleSequence(all_seq: np.array,
                           train_set: bool,
                           # max_predictions: int,
                           # pickled_distribution: distributions.Distribution,
                           pickled_tokenizer,
                           # training_opts,
                           # is_torch: bool,
                           # repair_locations: typing.List[int] = None,
                           ) -> typing.Generator:
  """
  Placing random holes seems to introduce an overfitting bias on the model.
  It doesn't learn a good distribution of what should go in a specific hole
  for a given index, a left and a right context. This function may be solving
  this, hopefully in a sustainable way. 

  No holes are placed randomly. Each index produces many holed seq instances;
  starting from empty hole up to hiding everything until the end.

  Given one sequence, returns a list of instances, one for each hole instances.

  !!!WARNING: Currently only supported for PyTorch.
  """
  with progressbar.ProgressBar(max_value = len(all_seq)) as bar:
    for seq in bar(all_seq):
      assert seq.ndim == 1, "Input for masking must be single-dimension array."

      # Unpack tokenizer
      tokenizer     = pickle.loads(pickled_tokenizer)
      use_start_end = True if seq[0] == tokenizer.startToken else False

      # Actual length represents the sequence length before pad begins
      start_idx = 0
      if use_start_end:
        start_idx = 1
        end       = np.where(seq == tokenizer.endToken)[0][0]
      elif tokenizer.padToken in seq:
        end       = np.where(seq == tokenizer.padToken)[0][0]
      else:
        end       = len(seq)

      st_input_ids = list(seq)
      for idx in range(start_idx, end):
        for hole_len in range(0, end - idx):
          if end + 1 - hole_len >= len(seq):
            continue 
          input_ids = st_input_ids[:idx] + [tokenizer.holeToken] + st_input_ids[idx + hole_len:]
          input_ids += [tokenizer.padToken] * (len(seq) - len(input_ids))
          input_ids = input_ids[:len(seq)]

          mask_labels = np.full(len(seq), -100, dtype = np.int64)
          target = seq[idx] if hole_len else tokenizer.endholeToken
          mask_labels[ idx  if hole_len else idx - 1] = target

          mlm_inst = MaskedLmInstance(
            pos_index   = idx,      token_id    = target,
            hole_length = hole_len, extend_left = False
          )
          yield ({
            'seen_in_training'    : np.int64([1] if train_set else [0]),
            'original_input'      : seq,
            'input_ids'           : np.asarray(input_ids, dtype = np.int64),
            'input_mask'          : (seq != tokenizer.padToken),
            'position_ids'        : np.arange(len(seq), dtype = np.int64),
            'mask_labels'         : mask_labels,
            'masked_lm_lengths'   : np.array([hole_len]),
            'next_sentence_labels': np.int64([0]),
          }, [mlm_inst])
  return