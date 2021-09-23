import numpy as np
import typing
import concurrent.futures

from deeplearning.clgen.preprocessors import opencl
from deeplearning.clgen.corpuses import tokenizers
from deeplearning.clgen.util import pytorch
from deeplearning.clgen.util.pytorch import torch

from eupy.native import logger as l

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
    # if self.use_categorical:
    #   self.argmax = lambda x: torch.argmax(
    #     torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(
    #       temperature = self.temperature if self.temperature is not None else 1.0, logits = x
    #     ).sample()
    #   )
    # else:
    # self.argmax = torch.argmax
    return

  def argmax(self, t):
    """Sample argmax from a tensor."""
    if self.use_categorical:
      t = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(
          temperature = self.temperature if self.temperature is not None else 1.0, logits = t
        ).sample()
    return torch.argmax(t, dim = -1)

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
    batch_size, sequence_length = tuple(input_ids.shape)
    # samples, sample_indices, scores_history = [], [], []
    samples, sample_indices, scores_history = self.BatchiterSampleSeq(
      model = model,
      device = device,
      input_ids = input_ids,
      prediction_scores = prediction_scores,
      position_ids = position_ids,
      is_live = is_live,
    )
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #   jobs = [executor.submit(self.iterSampleSeq,
    #                           model             = model,
    #                           device            = device,
    #                           input_ids         = input_ids        [i],
    #                           prediction_scores = prediction_scores[i],
    #                           position_ids      = position_ids     [i],
    #                           is_live           = is_live,
    #                       ) for i in range(batch_size)]
    #   for j in concurrent.futures.as_completed(jobs):
    #     s, si, sh = j.result()
    #     samples.append(s.numpy())
    #     sample_indices.append(si)
    #     scores_history.append(sh)
    #   results          = [j.result() for j in jobs]
    #   samples          = [x.numpy() for (x, _, _) in results]
    #   sample_indices   = [y for (_, y, _) in results]
    #   scores_history   = [z for (_, _, z) in results]
    return samples, sample_indices, scores_history

  def generateSampleWorkload(self,
                             model             : typing.TypeVar("model.BertPreTrainedModel"),
                             device            : torch.device,
                             input_ids         : torch.LongTensor,
                             attention_mask    : torch.LongTensor,
                             prediction_scores : torch.FloatTensor,
                             position_ids      : torch.LongTensor,
                             # queue,
                             ) -> typing.Tuple[typing.List[np.array], typing.List[typing.List[int]]]:
    return self.WorkloaditerSampleSeq(
      model  = model,
      device = device,
      # queue  = queue,
      workload_input_ids      = input_ids,
      workload_attention_mask = attention_mask,
      prediction_scores       = prediction_scores,
      position_ids            = position_ids,
    )

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
    holes, next_input_ids, attention_mask = self.StepSampleSeq(
      input_ids, prediction_scores, sample_indices, scores_history
    )
    while holes:
      next_prediction_scores, _, _, _ = model.get_output(
        next_input_ids.to(device), attention_mask.to(device), position_ids,
      )
      holes, next_input_ids, attention_mask = self.StepSampleSeq(
        next_input_ids[0],
        next_prediction_scores[0].detach().cpu(),
        sample_indices,
        scores_history,
      )
    return next_input_ids[0], sample_indices, scores_history

  def BatchiterSampleSeq(self,
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
    sample_indices = []
    for inpids in input_ids:
      sample_indices.append([
        [] for _ in range(len(torch.where((inpids == self.tokenizer.holeToken) | (inpids == self.tokenizer.maskToken))[0]))
      ])
    res_idx = 0
    results = torch.zeros_like(input_ids)

    new_holes = self.BatchStepSampleSeq(input_ids, prediction_scores, device)
    open_holes = torch.where(new_holes == True)[0]
    closed_holes = torch.where(new_holes == False)[0]

    results[res_idx: res_idx + len(closed_holes)] = input_ids[closed_holes]
    res_idx += len(closed_holes)
    input_ids = torch.index_select(input_ids, 0, open_holes.to(device))
    attention_mask = (input_ids != self.tokenizer.padToken)

    while torch.any(new_holes):

      prediction_scores, _, _, _ = model.get_output(
        input_ids, attention_mask, position_ids[:len(input_ids)],
      )

      new_holes = self.BatchStepSampleSeq(input_ids, prediction_scores, device)
      open_holes = torch.where(new_holes == True)[0]
      closed_holes = torch.where(new_holes == False)[0]

      results[res_idx: res_idx + len(closed_holes)] = input_ids[closed_holes]
      res_idx += len(closed_holes)
      input_ids = torch.index_select(input_ids, 0, open_holes.to(device))
      attention_mask = (input_ids != self.tokenizer.padToken)

    return results, sample_indices, None

  def WorkloaditerSampleSeq(self,
                            model              : typing.TypeVar("model.BertPreTrainedModel"),
                            device             : torch.device,
                            # queue              ,
                            workload_input_ids : torch.LongTensor,
                            workload_attention_mask: torch.LongTensor,
                            prediction_scores  : torch.LongTensor,
                            position_ids       : torch.LongTensor,
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
    # [workload_size x batch_size x sequence_length]
    wload_size, batch_size, sequence_length = tuple(workload_input_ids.shape)
    # Number of sequences
    nseq  = wload_size * batch_size
    # Iteration idx of workload
    w_idx = batch_size

    # Get current input_ids - attention mask.
    input_ids      = workload_input_ids[0]
    input_idxs     = torch.arange(batch_size).to(device)
    attention_mask = workload_attention_mask[0]

    # Workload of input_ids and attention_mask pairs.
    # queue input_idxs ensure direct ordering from inputs -> outputs.
    queue_input_ids      = torch.reshape(workload_input_ids, (1, nseq, sequence_length)).squeeze()
    queue_input_idxs     = torch.arange(nseq).to(device)
    queue_attention_mask = torch.reshape(workload_attention_mask, (1, nseq, sequence_length)).squeeze()

    #! This is the return queue [nseq x sequence_length].
    queue = torch.zeros(tuple(queue_input_ids.shape)).to(device)

    new_holes    = self.BatchStepSampleSeq(input_ids, prediction_scores, device)
    open_holes   = torch.where(new_holes == True)[0].to(device)
    closed_holes = torch.where(new_holes == False)[0]

    for i in closed_holes:
      queue[input_idxs[i]] = input_ids[i]

    input_ids      = torch.index_select(input_ids, 0, open_holes)
    input_idxs     = torch.index_select(input_idxs, 0, open_holes)
    attention_mask = (input_ids != self.tokenizer.padToken)

    res = batch_size - len(input_ids)
    if res > 0:
      input_ids      = torch.cat((input_ids, queue_input_ids[w_idx: w_idx + res]), 0)
      input_idxs     = torch.cat((input_idxs, queue_input_idxs[w_idx: w_idx + res]), 0)
      attention_mask = torch.cat((attention_mask, queue_attention_mask[w_idx: w_idx + res]), 0)
      w_idx += res

    while w_idx < nseq or torch.any(new_holes):

      prediction_scores, _, _, _ = model.get_output(
        input_ids, attention_mask, position_ids[:len(input_ids)],
      )
      # Array of new hole existence per seq idx
      new_holes    = self.BatchStepSampleSeq(input_ids, prediction_scores, device)
      # Fill these holes.
      open_holes   = torch.where(new_holes == True)[0].to(device)
      # Those are done.
      closed_holes = torch.where(new_holes == False)[0]

      # Add to return queue those that have finished.
      for i in closed_holes:
        queue[input_idxs[i]] = input_ids[i]

      input_ids      = torch.index_select(input_ids, 0, open_holes)
      input_idxs     = torch.index_select(input_idxs, 0, open_holes)
      attention_mask = (input_ids != self.tokenizer.padToken)

      res = batch_size - len(input_ids)
      if res > 0:
        input_ids      = torch.cat((input_ids, queue_input_ids[w_idx: w_idx + res]), 0)
        input_idxs     = torch.cat((input_idxs, queue_input_idxs[w_idx: w_idx + res]), 0)
        attention_mask = torch.cat((attention_mask, queue_attention_mask[w_idx: w_idx + res]), 0)
        w_idx += res
    return queue

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

  def BatchStepSampleSeq(self,
                         batch             : torch.LongTensor,
                         prediction_scores : torch.LongTensor,
                         device,
                         ) -> typing.Tuple[
                                bool,
                                torch.LongTensor,
                                np.array,
                              ]:
    """
    Applies sample step predictions to input batch of sequences.
    """
    endTokens = self.tokenizer.metaTokenValues
    # Array of boolean values, shows where holes are still left.
    new_hole = torch.zeros(len(batch), dtype=np.bool)

    # [seq_idx, hole_idx] of batch.
    idxs, targets = torch.where(batch == self.tokenizer.holeToken)
    # Predictions for these indices.
    predictions = self.argmax(prediction_scores[(idxs, targets)])

    for seq_idx, el_idx in zip(idxs, targets):
      if int(predictions[seq_idx]) in endTokens:
        # Close hole, shift left one position, add pad to the end.
        batch[seq_idx] = torch.cat((batch[seq_idx][:el_idx], batch[seq_idx][el_idx+1:], torch.LongTensor([self.tokenizer.padToken]).to(device)), 0)
      elif int(batch[seq_idx][-1]) != self.tokenizer.padToken:
        # No pads remaining to the right, replace hole with prediction but don't insert new hole.
        batch[seq_idx] = torch.cat((batch[seq_idx][:el_idx], predictions[seq_idx].unsqueeze(0), batch[seq_idx][el_idx+1:]), 0)
      else:
        # Replace with prediction and keep hole.
        batch[seq_idx] = torch.cat((batch[seq_idx][:el_idx], predictions[seq_idx].unsqueeze(0), batch[seq_idx][el_idx:][:-1]), 0)
        new_hole[seq_idx] = True

    return new_hole
