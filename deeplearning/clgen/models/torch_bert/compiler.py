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
                            attention_mask    : torch.LongTensor,
                            position_ids      : torch.LongTensor,
                            masked_lm_labels  : torch.LongTensor,
                            ) -> typing.Tuple[typing.List[np.array], typing.List[int], torch.LongTensor]:
    batch_size, sequence_length = tuple(input_ids.shape)
    with concurrent.futures.ThreadPoolExecutor() as executor:
      jobs = [executor.submit(self.iterTrainingSeq,
                              model           = model,
                              device          = device,
                              seq             = input_ids        [i].cpu(),
                              prediction      = prediction_scores[i].detach().cpu(),
                              attention       = attention_mask   [i].cpu(),
                              position_ids    = position_ids     [i].cpu().unsqueeze(0),
                              masked_lm_label = masked_lm_labels [i].cpu().numpy(),
                          ) for i in range(batch_size)]

      results          = [j.result() for j in jobs]
      samples          = [x.numpy() for (x, _, _) in results]
      compile_flag     = [y         for (_, y, _) in results]
      masked_lm_labels = torch.LongTensor([z for (_, _, z) in results]).to(device)
      return samples, compile_flag, masked_lm_labels

  def iterTrainingSeq(self,
                      model           : typing.TypeVar("model.BertPreTrainedModel"),
                      device          : torch.device,
                      seq             : torch.LongTensor,
                      prediction      : torch.FloatTensor,
                      attention       : torch.LongTensor,
                      position_ids    : torch.LongTensor,
                      masked_lm_label : torch.LongTensor,
                      ) -> typing.Tuple[torch.LongTensor, int, torch.LongTensor]:
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
    holes, new_seq, new_attention = self.StepTrainingSeq(
      seq, prediction, attention
    )
    while holes:
      new_prediction, new_seq_relationship_score, _, _ = model.get_output(
        new_seq.to(device), new_attention.to(device), position_ids.to(device),
      )
      holes, new_seq, new_attention = self.StepTrainingSeq(
        new_seq[0],
        new_prediction.detach().cpu()[0],
        new_attention,
      )
    compile_flag = self.checkIfBatchCompiles(new_seq[0].numpy())
    if compile_flag:
      for idx, t in enumerate(masked_lm_label):
        if t != -100:
          masked_lm_label[idx] = -100
    return new_seq[0], compile_flag, masked_lm_label

  def StepTrainingSeq(self,
                      seq               : torch.LongTensor,
                      prediction_scores : torch.FloatTensor,
                      attention_mask    : torch.LongTensor,
                      ) -> typing.Tuple[bool, torch.LongTensor, np.array]:
    """
    Applies step predictions to input sequence.
    Specifically optimized for training; does not compute sample indices for speed-up.
    """
    seq_length    = tuple(seq.shape)[0]
    allowed_incr = (seq_length - int(torch.where(seq==self.tokenizer.padToken)[0][0])
                    if self.tokenizer.padToken in seq
                    else 0)

    endTokens   = self.tokenizer.metaTokenValues
    closed_hole = np.zeros(seq_length, dtype=np.bool)
    new_hole = np.zeros(seq_length, dtype=np.bool)
    temp_seq = seq.cpu().detach().numpy().copy()

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
                          attention_mask    : torch.LongTensor,
                          position_ids      : torch.LongTensor,
                          ) -> typing.Tuple[typing.List[np.array], typing.List[typing.List[int]]]:
    ###
    batch_size, sequence_length = tuple(input_ids.shape)
    with concurrent.futures.ThreadPoolExecutor() as executor:
      jobs = [executor.submit(self.iterSampleSeq,
                              model          = model,
                              device         = device,
                              seq            = input_ids        [i].cpu(),
                              prediction     = prediction_scores[i].detach().cpu(),
                              attention      = attention_mask   [i].cpu(),
                              position_ids   = position_ids     [i].cpu().unsqueeze(0),
                          ) for i in range(batch_size)]

      results          = [j.result() for j in jobs]
      samples          = [x.numpy() for (x, _, _) in results]
      sample_indices   = [y for (_, y, _) in results]
      scores_history   = [z for (_, _, z) in results]
      return samples, sample_indices, scores_history

  def iterSampleSeq(self,
                    model        : typing.TypeVar("model.BertPreTrainedModel"),
                    device       : torch.device,
                    seq          : torch.LongTensor,
                    prediction   : torch.LongTensor,
                    attention    : torch.LongTensor,
                    position_ids : torch.LongTensor,
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
        len(torch.where((seq == self.tokenizer.holeToken) | (seq == self.tokenizer.maskToken))[0])
      )
    ]
    scores_history = []
    holes, new_seq, new_attention, sample_indices = self.StepSampleSeq(
      seq, prediction, attention, sample_indices, scores_history
    )
    while holes:
      new_prediction, new_seq_relationship_score, _, _ = model.get_output(
        new_seq.to(device), new_attention.to(device), position_ids.to(device),
      )
      holes, new_seq, new_attention, sample_indices = self.StepSampleSeq(
        new_seq[0],
        new_prediction.detach().cpu()[0],
        new_attention,
        sample_indices,
        scores_history
      )
    return new_seq[0], sample_indices, scores_history

  def StepSampleSeq(self,
                    seq               : torch.LongTensor,
                    prediction_scores : torch.LongTensor,
                    attention_mask    : torch.LongTensor,
                    sample_indices    : typing.List[typing.List[int]],
                    scores_history    : typing.List[np.array],
                    ) -> typing.Tuple[
                          bool,
                          torch.LongTensor,
                          np.array,
                          typing.List[typing.List[int]]
                         ]:
    """
    Applies step predictions to input sequence.
    Specifically optimized for training; does not compute sample indices for speed-up.
    """
    step_indices  = []
    seq_length    = tuple(seq.shape)[0]
    allowed_incr  = (seq_length - int(torch.where(seq==self.tokenizer.padToken)[0][0])
                     if self.tokenizer.padToken in seq
                     else 0)

    endTokens = self.tokenizer.metaTokenValues
    closed_hole = np.zeros(seq_length, dtype=np.bool)
    new_hole = np.zeros(seq_length, dtype=np.bool)
    temp_seq = seq.cpu().detach().numpy().copy()

    for target_idx in torch.where((seq == self.tokenizer.holeToken) | (seq == self.tokenizer.maskToken))[0]:
      idx        = int(target_idx)
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

    return np.any(new_hole), new_seq, attention_mask, sample_indices
