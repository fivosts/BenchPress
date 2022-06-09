import numpy as np
import typing
import pathlib
import concurrent.futures

from deeplearning.clgen.preprocessors import opencl
from deeplearning.clgen.corpuses import tokenizers
from deeplearning.clgen.util import environment
from deeplearning.clgen.util import pytorch
from deeplearning.clgen.util.pytorch import torch
from absl import flags

from deeplearning.clgen.util import logging as l

FLAGS = flags.FLAGS

flags.DEFINE_integer(
  "sample_indices_limit",
  None,
  "Hard-stop model generating more indices per sample than this specified integer."
)

class CompilationSampler(object):
  """
  Compilation driven generation handler.
  
  Used during training to iteratively fill a sequence
  and feed to Clang for compilation status.

  Also used during sampling to fill sequence and get
  compilation status.
  """
  def __init__(self,
               tokenizer       : tokenizers.TokenizerBase,
               use_categorical : bool,
               temperature     : float,
               target_lm       : str,
               ):
    self.tokenizer       = tokenizer
    self.temperature     = temperature
    self.use_categorical = use_categorical
    if target_lm == "hole":
      self.step_batch = self.StepHoleSeq
    elif target_lm == "mask":
      self.step_batch = self.StepMaskSeq
    else:
      raise KeyError(target_lm)
    return

  def argmax(self, t):
    """Sample argmax from a tensor."""
    if self.use_categorical:
      try:
        ct = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(
            temperature = self.temperature if self.temperature is not None else 1.0,
            logits = t,
            validate_args = False if "1.9." in torch.__version__ else None,
          ).sample()
      except ValueError as e:
        dump_cf = ""
        dump_types = ""
        p = pathlib.Path("./dump_argmax_error.log").absolute()
        if not p.exists():
          l.logger().error(t.shape)
          l.logger().error(p)
          for d0 in t:
            for d1 in d0:
              dump_cf += str(d1) + ", "
              if isinstance(d1, torch.Tensor):
                dump_types += str(d1.type()) + ", "
              else:
                dump_types += str(type(d1)) + ", "
          with open(p, 'w') as outf:
            outf.write(str(t.shape) + "\n\n\n" + dump_cf + "\n\n\n" + dump_types)
        raise e

    return torch.argmax(ct, dim = -1)

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
                            input_features    : torch.LongTensor,
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
                              input_features    = input_features   [i] if input_features else None,
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
                      input_features    : torch.LongTensor,
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
          next_input_ids.to(device), attention_mask.to(device), position_ids, input_features
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
        l.logger().info("seq: {}".format(self.tokenizer.tokensToString([x for x in seq.cpu().numpy()])))
        l.logger().info("temp_seq {}".format(self.tokenizer.tokensToString([x for x in temp_seq])))
        l.logger().info("pred idx: {}".format(torch.where((seq == self.tokenizer.holeToken) | (seq == self.tokenizer.maskToken))[0]))
        l.logger().info("pred_toks {}".format(self.tokenizer.tokensToString([int(self.argmax(prediction_scores[idx])) for idx in torch.where((seq == self.tokenizer.holeToken) | (seq == self.tokenizer.maskToken))[0]])))
        l.logger().info("allowed_incr: {}".format(allowed_incr))
        l.logger().info("new_hole: {}".format(new_hole))
        l.logger().info("closed_hole: {}".format(closed_hole))
      new_idx += 1
      if new_hole[idx]:
        try:
          new_seq[new_idx] = self.tokenizer.holeToken
        except IndexError:
          l.logger().warn("seq: {}".format(self.tokenizer.tokensToString([x for x in seq.cpu().numpy()])))
          l.logger().warn("temp_seq {}".format(self.tokenizer.tokensToString([x for x in temp_seq])))
          l.logger().warn("pred idx: {}".format(torch.where((seq == self.tokenizer.holeToken) | (seq == self.tokenizer.maskToken))[0]))
          l.logger().warn("pred_toks {}".format(self.tokenizer.tokensToString([int(self.argmax(prediction_scores[idx])) for idx in torch.where((seq == self.tokenizer.holeToken) | (seq == self.tokenizer.maskToken))[0]])))
          l.logger().warn("allowed_incr: {}".format(allowed_incr))
          l.logger().warn("new_hole: {}".format(new_hole))
          l.logger().warn("closed_hole: {}".format(closed_hole))
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
                          input_features    : torch.LongTensor,
                          prediction_scores : torch.FloatTensor,
                          position_ids      : torch.LongTensor,
                          is_live           : bool,
                          ) -> typing.Tuple[typing.List[np.array], typing.List[typing.List[int]]]:
    """
    Get a batch of input ids and iteratively fill the holes and return a batch of samples.
    """
    batch_size, sequence_length = tuple(input_ids.shape)
    input_idxs = torch.arange(batch_size).to(device)
    sample_indices = torch.full((batch_size, sequence_length), self.tokenizer.padToken, dtype = torch.int64).to(device)

    res_idx = 0
    samples = torch.zeros_like(input_ids)

    new_holes = self.step_batch(input_ids, input_idxs, sample_indices, None, prediction_scores, device)
    open_holes = torch.where(new_holes == True)[0]
    closed_holes = torch.where(new_holes == False)[0]

    samples[res_idx: res_idx + len(closed_holes)] = input_ids[closed_holes]
    res_idx += len(closed_holes)
    input_ids = torch.index_select(input_ids, 0, open_holes.to(device))
    attention_mask = (input_ids != self.tokenizer.padToken)

    while torch.any(new_holes):

      prediction_scores, _, _, _ = model.get_output(
        input_ids, attention_mask, position_ids[:len(input_ids)], input_features,
      )

      new_holes = self.step_batch(input_ids, input_idxs, sample_indices, None, prediction_scores, device)
      open_holes = torch.where(new_holes == True)[0]
      closed_holes = torch.where(new_holes == False)[0]

      samples[res_idx: res_idx + len(closed_holes)] = input_ids[closed_holes]
      res_idx += len(closed_holes)
      input_ids = torch.index_select(input_ids, 0, open_holes.to(device))
      attention_mask = (input_ids != self.tokenizer.padToken)

    return samples, sample_indices, None

  def generateSampleWorkload(self,
                             model                   : typing.TypeVar("model.BertPreTrainedModel"),
                             device                  : torch.device,
                             workload_input_ids      : torch.LongTensor,
                             workload_attention_mask : torch.LongTensor,
                             workload_input_features : torch.LongTensor,
                             prediction_scores       : torch.FloatTensor,
                             position_ids            : torch.LongTensor,
                             bar                     : 'tqdm.tqdm' = None,
                             ) -> typing.Tuple[typing.List[np.array], typing.List[typing.List[int]]]:
    """
    This function receives a full workload of input ids to be sampled.
    Heavy optimisations are perfmormed to keep the GPU busy at all times.

    The workload is streamed online and when a sequence is finished it is replaced
    with a new one from the workload queue.

    Returns a fullworkload of sampled instances.
    """
    # [workload_size x batch_size x sequence_length]
    wload_size, batch_size, sequence_length = tuple(workload_input_ids.shape)
    # Also compute feature embeddings sequence length.
    if workload_input_features is not None:
      _, _, feature_sequence_length = tuple(workload_input_features.shape)
    # Number of sequences
    nseq  = wload_size * batch_size
    # Iteration idx of workload
    w_idx = batch_size

    # Get current input_ids - attention mask.
    input_ids      = workload_input_ids[0]
    input_idxs     = torch.arange(batch_size).to(device)
    attention_mask = workload_attention_mask[0]
    if workload_input_features is not None:
      input_features = workload_input_features[0]
    else:
      input_features = None
    # sample indices array that will be returned.
    sample_indices = torch.full((nseq, sequence_length), self.tokenizer.padToken, dtype = torch.int64).to(device)

    if FLAGS.sample_indices_limit is not None:
      sidx_length = torch.full((batch_size, 1), 0, dtype = torch.int64).to(device)

    # Workload of input_ids and attention_mask pairs.
    # queue input_idxs ensure direct ordering from inputs -> outputs.
    queue_input_ids      = torch.reshape(workload_input_ids, (1, nseq, sequence_length)).squeeze()
    queue_input_idxs     = torch.arange(nseq).to(device)
    queue_attention_mask = torch.reshape(workload_attention_mask, (1, nseq, sequence_length)).squeeze()
    if workload_input_features is not None:
      queue_input_features = torch.reshape(workload_input_features, (1, nseq, feature_sequence_length)).squeeze()

    #! This is the return queue [nseq x sequence_length].
    queue = torch.zeros(tuple(queue_input_ids.shape), dtype = torch.int64).to(device)

    new_holes = self.step_batch(
      input_ids,
      input_idxs,
      sample_indices,
      sidx_length if FLAGS.sample_indices_limit else None,
      prediction_scores,
      device
    )
    open_holes   = torch.where(new_holes == True)[0].to(device)
    closed_holes = torch.where(new_holes == False)[0]

    for i in closed_holes:
      queue[input_idxs[i]] = input_ids[i]
      if bar:
        bar.update(1)

    input_ids      = torch.index_select(input_ids, 0, open_holes)
    input_idxs     = torch.index_select(input_idxs, 0, open_holes)
    attention_mask = (input_ids != self.tokenizer.padToken)
    if input_features is not None:
      input_features = torch.index_select(input_features, 0, open_holes)
    if FLAGS.sample_indices_limit:
      sidx_length  = torch.index_select(sidx_length, 0, open_holes)

    res = batch_size - len(input_ids)
    if res > 0:
      input_ids      = torch.cat((input_ids, queue_input_ids[w_idx: w_idx + res]), 0)
      input_idxs     = torch.cat((input_idxs, queue_input_idxs[w_idx: w_idx + res]), 0)
      attention_mask = torch.cat((attention_mask, queue_attention_mask[w_idx: w_idx + res]), 0)
      if input_features is not None:
        input_features = torch.cat((input_features, queue_input_features[w_idx: w_idx + res]), 0)
      if FLAGS.sample_indices_limit:
        sidx_length  = torch.cat((sidx_length, torch.full((res, 1), 0, dtype = torch.int64).to(device)), 0)
      w_idx += res

    while w_idx < nseq or torch.any(new_holes):

      prediction_scores, _, _, _ = model.get_output(
        input_ids, attention_mask, position_ids[:len(input_ids)], input_features
      )
      # Array of new hole existence per seq idx
      new_holes = self.step_batch(
        input_ids,
        input_idxs,
        sample_indices,
        sidx_length if FLAGS.sample_indices_limit else None,
        prediction_scores,
        device
      )
      # Fill these holes.
      open_holes   = torch.where(new_holes == True)[0].to(device)
      # Those are done.
      closed_holes = torch.where(new_holes == False)[0]

      # Add to return queue those that have finished.
      for i in closed_holes:
        queue[input_idxs[i]] = input_ids[i]
        if bar:
          bar.update(1)
    
      input_ids      = torch.index_select(input_ids, 0, open_holes)
      input_idxs     = torch.index_select(input_idxs, 0, open_holes)
      attention_mask = (input_ids != self.tokenizer.padToken)
      if input_features is not None:
        input_features = torch.index_select(input_features, 0, open_holes)
      if FLAGS.sample_indices_limit:
        sidx_length  = torch.index_select(sidx_length, 0, open_holes)

      res = batch_size - len(input_ids)
      if res > 0:
        input_ids      = torch.cat((input_ids, queue_input_ids[w_idx: w_idx + res]), 0)
        input_idxs     = torch.cat((input_idxs, queue_input_idxs[w_idx: w_idx + res]), 0)
        attention_mask = torch.cat((attention_mask, queue_attention_mask[w_idx: w_idx + res]), 0)
        if input_features is not None:
          input_features = torch.cat((input_features, queue_input_features[w_idx: w_idx + res]), 0)
        if FLAGS.sample_indices_limit:
          sidx_length  = torch.cat((sidx_length, torch.full((res, 1), 0, dtype = torch.int64).to(device)), 0)
        w_idx += res
    return queue, sample_indices

  def StepHoleSeq(self,
                  batch             : torch.LongTensor,
                  batch_idxs        : torch.LongTensor,
                  sample_indices    : torch.LongTensor,
                  indices_lengths   : torch.LongTensor,
                  prediction_scores : torch.LongTensor,
                  device,
                  ) -> typing.Tuple[
                         bool,
                         torch.LongTensor,
                         np.array,
                       ]:
    """
    Applies sample step with hole predictions to input batch.

    !!!!!!WARNING!!!!!
    This function works appropriately ONLY for 1 [HOLE] per sequence.
    If more HOLES existed, then further operations would be needed to
    re-calculate the proceeding hole indices, which would lead to unnecessary
    operations. Removing this feature keeps things faster for 1 hole scenario.
    """
    endTokens = self.tokenizer.metaTokenValues
    # Array of boolean values, shows where holes are still left.
    new_hole = torch.zeros(len(batch), dtype=np.bool)

    # [seq_idx, hole_idx] of batch.
    idxs, targets = torch.where(batch == self.tokenizer.holeToken)
    # Predictions for these indices.
    predictions = self.argmax(prediction_scores[(idxs, targets)])

    for seq_idx, el_idx in zip(idxs, targets):
      # seq_idx -> indices within the batch
      # el_idx  -> element index within a sequence
      if int(predictions[seq_idx]) in endTokens:
        # Close hole, shift left one position, add pad to the end.
        batch[seq_idx] = torch.cat((batch[seq_idx][:el_idx], batch[seq_idx][el_idx+1:], torch.LongTensor([self.tokenizer.padToken]).to(device)), 0)
      elif int(batch[seq_idx][-1]) != self.tokenizer.padToken or (indices_lengths is not None and indices_lengths[seq_idx] >= FLAGS.sample_indices_limit-1):
        # No pads remaining to the right, replace hole with prediction but don't insert new hole.
        # batch[seq_idx] = torch.cat((batch[seq_idx][:el_idx], predictions[seq_idx].unsqueeze(0), batch[seq_idx][el_idx+1:]), 0)
        batch[seq_idx][el_idx] = predictions[seq_idx]
      else:
        # Replace with prediction and keep hole.
        batch[seq_idx] = torch.cat((batch[seq_idx][:el_idx], predictions[seq_idx].unsqueeze(0), batch[seq_idx][el_idx:][:-1]), 0)
        new_hole[seq_idx] = True
      q_idx = batch_idxs[seq_idx]
      sample_indices[q_idx][el_idx] = predictions[seq_idx]
      if indices_lengths is not None:
        indices_lengths[seq_idx] += 1

    return new_hole

  def StepMaskSeq(self,
                  batch             : torch.LongTensor,
                  batch_idxs        : torch.LongTensor,
                  sample_indices    : torch.LongTensor,
                  indices_lengths   : torch.LongTensor,
                  prediction_scores : torch.LongTensor,
                  device,
                  ) -> typing.Tuple[
                         bool,
                         torch.LongTensor,
                         np.array,
                       ]:
    """
    Applies sample step with mask predictions to input batch.
    """
    # [seq_idx, hole_idx] of batch.
    idxs, targets = torch.where(batch == self.tokenizer.maskToken)
    # Predictions for these indices.
    predictions = self.argmax(prediction_scores[(idxs, targets)])
    for p_idx, (seq_idx, el_idx) in enumerate(zip(idxs.flip(dims = (0,)), targets.flip(dims = (0,)))):
      # seq_idx -> indices within the batch
      # el_idx  -> element index within a sequence
      # Casually replace the [MASK] with the single predicted token.
      batch[seq_idx][el_idx] = predictions[idxs.size(0) - 1 - p_idx]
      q_idx = batch_idxs[seq_idx]
      sample_indices[q_idx][el_idx] = predictions[idxs.size(0) - 1 - p_idx]
      if indices_lengths is not None:
        indices_lengths[seq_idx] += 1

    return torch.zeros(len(batch), dtype=np.bool)
