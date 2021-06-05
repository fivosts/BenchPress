import numpy as np
import typing
import concurrent.futures
import time

from deeplearning.clgen.preprocessors import opencl
from deeplearning.clgen.corpuses import tokenizers
from deeplearning.clgen.util import pytorch
from deeplearning.clgen.util.pytorch import torch

from eupy.native import logger as l

times = {
  'generateSampleBatch': 0,
  'get_output': 0,
  'StepSampleSeq': 0,
}

class CompilationSampler(object):
  """
  Compilation driven generation handler.
  
  Used during training to iteratively fill a sequence
  and feed to Clang for compilation status.

  Also used during sampling to fill sequence and get
  compilation status.
  """
  def __init__(self,
               tokenizer        : tokenizers.TokenizerBase,
               use_categorical : bool,
               temperature     : float,
               ):
    self.tokenizer        = tokenizer
    self.temperature     = temperature
    self.use_categorical = use_categorical
    if self.use_categorical:
      self.argmax = lambda x: torch.argmax(
        torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(
          temperature = self.temperature if self.temperature is not None else 1.0, logits = x
        ).sample()
      )
    else:
      self.argmax = lambda x: torch.argmax(x)
    return

  def checkIfBatchCompiles(self,
                           sample: np.array
                           ) -> int:
    """Sends a filled sequence to the compiler"""
    try:
      stdout = opencl.Compile(self.tokenizer.ArrayToCode(sample))
      return 1
    except ValueError:
      return 0

  def generateTrainingBatch(self,
                            model             : typing.TypeVar("model.BertPreTrainedModel"),
                            device            : torch.device,
                            input_ids         : torch.LongTensor,
                            prediction_scores : torch.FloatTensor,
                            position_ids      : torch.LongTensor,
                            masked_lm_labels  : torch.LongTensor,
                            ) -> typing.Tuple[typing.List[np.array], typing.List[int]]:
    batch_size, sequence_length = tuple(input_ids.shape)
    with concurrent.futures.ThreadPoolExecutor() as executor:
      jobs = [executor.submit(self.iterTrainingSeq,
                              model             = model,
                              device            = device,
                              input_ids         = input_ids        [i],
                              prediction_scores = prediction_scores[i],
                              position_ids      = position_ids     [i],
                              masked_lm_labels  = masked_lm_labels [i],
                          ) for i in range(batch_size)]

      results          = [j.result() for j in jobs]
      samples          = [x.numpy() for (x, _, _) in results]
      compile_flag     = [y         for (_, y, _) in results]
      masked_lm_labels = torch.LongTensor([z for (_, _, z) in results]).to(device)
      return samples, compile_flag, masked_lm_labels

  def iterTrainingSeq(self,
                      model             : typing.TypeVar("model.BertPreTrainedModel"),
                      device            : torch.device,
                      input_ids         : torch.LongTensor,
                      prediction_scores : torch.FloatTensor,
                      position_ids      : torch.LongTensor,
                      masked_lm_labels  : torch.LongTensor,
                      ) -> typing.Tuple[torch.LongTensor, int]:
    """
    Main training sequence filling loop.
    
    Function takes model's initial input, prediction and states.
    Fills input sequence with step predictions and keeps asking
    iteratively for predictions until target [MASK] or [HOLE] tokens
    are closed.

    Compiler is invoked for final sequence to get binary compilation status.
    ##!! This function is designed to work with multithreading and exercises
         said functionalities on a single sequence. CANNOT be applied to the
         whole batch at the same time.
    """
    new_holes, next_input_ids, attention_mask = self.StepTrainingSeq(input_ids, prediction_scores)
    with torch.no_grad():
      while new_holes:
        next_prediction_scores, _, _, _ = model.get_output(
          next_input_ids.to(device), attention_mask.to(device), position_ids,
        )
        new_holes, next_input_ids, attention_mask = self.StepTrainingSeq(
          next_input_ids[0], next_prediction_scores[0],
        )
    compile_flag = self.checkIfBatchCompiles(next_input_ids[0].numpy())
    if compile_flag:
      masked_lm_labels = np.full(masked_lm_labels.shape, -100, dtype = np.int64)
    return next_input_ids[0], compile_flag, masked_lm_labels

  def StepTrainingSeq(self,
                      seq               : torch.LongTensor,
                      prediction_scores : torch.FloatTensor,
                      ) -> typing.Tuple[bool, torch.LongTensor, np.array]:
    """
    Applies step predictions to input sequence.
    Specifically optimized for training; does not compute sample indices for speed-up.
    """
    seq_length   = tuple(seq.shape)[0]
    allowed_incr = (seq_length - int(torch.where(seq==self.tokenizer.padToken)[0][0])
                    if self.tokenizer.padToken in seq
                    else 0)

    endTokens   = self.tokenizer.metaTokenValues
    closed_hole = np.zeros(seq_length, dtype=np.bool)
    new_hole = np.zeros(seq_length, dtype=np.bool)
    temp_seq = seq.numpy().copy()

    for target_idx in torch.where((seq == self.tokenizer.holeToken) | (seq == self.tokenizer.maskToken))[0]:
      idx        = int(target_idx)
      prediction = int(self.argmax(prediction_scores[target_idx]))
      is_hole = temp_seq[idx] == self.tokenizer.holeToken

      if prediction in endTokens:
        # Model predicted sth that will close the hole.
        closed_hole[idx] = True
        continue

      # We replace the hole with a prediction
      temp_seq[idx] = prediction
      rem_adds = allowed_incr + np.sum(closed_hole) - np.sum(new_hole)
      if is_hole and rem_adds:
        # if this was a hole and we have more empty space, reinsert the hole
        new_hole[idx] = True

    new_seq = np.full(seq_length, self.tokenizer.padToken, dtype=np.int64)
    new_idx = 0
    for idx, t in enumerate(temp_seq):
      if closed_hole[idx]:
        continue
      try:
        new_seq[new_idx] = t
      except IndexError:
        l.getLogger().info("seq: {}".format(self.tokenizer.tokensToString([x for x in seq.cpu().numpy()])))
        l.getLogger().info("temp_seq {}".format(self.tokenizer.tokensToString([x for x in temp_seq])))
        l.getLogger().info("pred idx: {}".format(torch.where((seq == self.tokenizer.holeToken) | (seq == self.tokenizer.maskToken))[0]))
        l.getLogger().info("pred_toks {}".format(self.tokenizer.tokensToString([int(self.argmax(prediction_scores[idx])) for idx in torch.where((seq == self.tokenizer.holeToken) | (seq == self.tokenizer.maskToken))[0]])))
        l.getLogger().info("allowed_incr: {}".format(allowed_incr))
        l.getLogger().info("new_hole: {}".format(new_hole))
        l.getLogger().info("closed_hole: {}".format(closed_hole))
      new_idx += 1
      if new_hole[idx]:
        try:
          new_seq[new_idx] = self.tokenizer.holeToken
        except IndexError:
          l.getLogger().warn("seq: {}".format(self.tokenizer.tokensToString([x for x in seq.cpu().numpy()])))
          l.getLogger().warn("temp_seq {}".format(self.tokenizer.tokensToString([x for x in temp_seq])))
          l.getLogger().warn("pred idx: {}".format(torch.where((seq == self.tokenizer.holeToken) | (seq == self.tokenizer.maskToken))[0]))
          l.getLogger().warn("pred_toks {}".format(self.tokenizer.tokensToString([int(self.argmax(prediction_scores[idx])) for idx in torch.where((seq == self.tokenizer.holeToken) | (seq == self.tokenizer.maskToken))[0]])))
          l.getLogger().warn("allowed_incr: {}".format(allowed_incr))
          l.getLogger().warn("new_hole: {}".format(new_hole))
          l.getLogger().warn("closed_hole: {}".format(closed_hole))
        new_idx += 1
      if new_idx >= seq_length:
        break

    new_seq = torch.LongTensor([new_seq])
    attention_mask = (new_seq != self.tokenizer.padToken)
    return np.any(new_hole), new_seq, attention_mask

  def generateSampleBatch(self,
                          model             : typing.TypeVar("model.BertPreTrainedModel"),
                          device            : torch.device,
                          input_ids         : torch.LongTensor,
                          prediction_scores : torch.FloatTensor,
                          position_ids      : torch.LongTensor,
                          is_live           : bool,
                          ) -> typing.Tuple[typing.List[np.array], typing.List[typing.List[int]]]:
    t1 = time.time()
    batch_size, sequence_length = tuple(input_ids.shape)
    samples, sample_indices, scores_history = [], [], []
    with concurrent.futures.ThreadPoolExecutor() as executor:
      jobs = [executor.submit(self.iterSampleSeq,
                              model             = model,
                              device            = device,
                              input_ids         = input_ids        [i],
                              prediction_scores = prediction_scores[i],
                              position_ids      = position_ids     [i],
                              is_live           = is_live,
                          ) for i in range(batch_size)]
      for j in concurrent.futures.as_completed(jobs):
        s, si, sh = j.result()
        samples.append(s.numpy())
        sample_indices.append(si)
        scores_history.append(sh)
      # results          = [j.result() for j in jobs]
      # samples          = [x.numpy() for (x, _, _) in results]
      # sample_indices   = [y for (_, y, _) in results]
      # scores_history   = [z for (_, _, z) in results]
    t2 = time.time()
    times['generateSampleBatch'] += t2-t1
    return samples, sample_indices, scores_history

  def iterSampleSeq(self,
                    model             : typing.TypeVar("model.BertPreTrainedModel"),
                    device            : torch.device,
                    input_ids         : torch.LongTensor,
                    prediction_scores : torch.LongTensor,
                    position_ids      : torch.LongTensor,
                    is_live           : bool,
                    ) -> typing.Tuple[torch.LongTensor, typing.List[typing.List[int]], typing.List[np.array]]:
    """
    Main sampling sequence filling loop.
    
    Function takes model's initial input, prediction and states.
    Fills input sequence with step predictions and keeps asking
    iteratively for predictions until target [MASK] or [HOLE] tokens
    are closed.

    Compiler is invoked for final sequence to get binary compilation status.
    ##!! This function is designed to work with multithreading and exercises
         said functionalities on a single sequence. CANNOT be applied to the
         whole batch at the same time.
    """
    sample_indices = [
      [] for _ in range(
        len(torch.where((input_ids == self.tokenizer.holeToken) | (input_ids == self.tokenizer.maskToken))[0])
      )
    ]
    if is_live:
      scores_history = []
    else:
      scores_history = None
    t1 = time.time()
    holes, next_input_ids, attention_mask = self.StepSampleSeq(
      input_ids, prediction_scores, sample_indices, scores_history
    )
    t2 = time.time()
    times['StepSampleSeq'] += t2-t1
    while holes:
      t1 = time.time()
      next_prediction_scores, _, _, _ = model.get_output(
        next_input_ids.to(device), attention_mask.to(device), position_ids,
      )
      t2 = time.time()
      times['get_output'] += t2-t1
      t1 = time.time()
      holes, next_input_ids, attention_mask = self.StepSampleSeq(
        next_input_ids[0],
        next_prediction_scores[0].detach().cpu(),
        sample_indices,
        scores_history,
      )
      t2 = time.time()
      times['StepSampleSeq'] += t2-t1
    return next_input_ids[0], sample_indices, scores_history

  def StepSampleSeq(self,
                    seq               : torch.LongTensor,
                    prediction_scores : torch.LongTensor,
                    sample_indices    : typing.List[typing.List[int]],
                    scores_history    : typing.List[np.array],
                    ) -> typing.Tuple[
                          bool,
                          torch.LongTensor,
                          np.array,
                         ]:
    """
    Applies sample step predictions to input sequence.
    """
    step_indices  = []
    seq_length    = tuple(seq.shape)[0]
    allowed_incr  = (seq_length - int(torch.where(seq==self.tokenizer.padToken)[0][0])
                     if self.tokenizer.padToken in seq
                     else 0)

    endTokens = self.tokenizer.metaTokenValues
    closed_hole = np.zeros(seq_length, dtype=np.bool)
    new_hole = np.zeros(seq_length, dtype=np.bool)
    temp_seq = seq.numpy().copy()

    for target_idx in torch.where((seq == self.tokenizer.holeToken) | (seq == self.tokenizer.maskToken))[0]:
      idx        = int(target_idx)
      if scores_history is not None:
        scores_history.append(prediction_scores[target_idx].numpy())
      prediction = int(self.argmax(prediction_scores[target_idx]))
      step_indices.append([prediction])
      is_hole = temp_seq[idx] == self.tokenizer.holeToken

      if prediction in endTokens:
        # Model predicted sth that will close the hole.
        closed_hole[idx] = True
        continue

      # We replace the hole with a prediction
      temp_seq[idx] = prediction
      rem_adds = allowed_incr + np.sum(closed_hole) - np.sum(new_hole)
      if is_hole and rem_adds:
        # if this was a hole and we have more empty space, reinsert the hole
        new_hole[idx] = True
      else:
        step_indices[-1].append(self.tokenizer.endholeToken)

    new_seq = np.full(seq_length, self.tokenizer.padToken, dtype=np.int64)
    new_idx = 0
    for idx, t in enumerate(temp_seq):
      if closed_hole[idx]:
        continue
      new_seq[new_idx] = t
      new_idx += 1
      if new_hole[idx]:
        new_seq[new_idx] = self.tokenizer.holeToken
        new_idx += 1
      if new_idx >= seq_length:
        break

    new_seq = torch.LongTensor([new_seq])
    attention_mask = (new_seq != self.tokenizer.padToken)

    # Update sample indices
    t_idx = 0
    for target_indices, _ in enumerate(sample_indices):
      if len(sample_indices[target_indices]) == 0 or sample_indices[target_indices][-1] not in endTokens:
        sample_indices[target_indices] += step_indices[t_idx]
        t_idx += 1

    return np.any(new_hole), new_seq, attention_mask
