import numpy as np
import typing
import concurrent.futures

from deeplearning.clgen.models.torch_bert import model
from deeplearning.clgen.corpuses import atomizers
from deeplearning.clgen.utils.pytorch import torch

class CompilationSampler(object):
  """
  Compilation driven generation handler.
  
  Used during training to iteratively fill a sequence
  and feed to Clang for compilation status.

  Also used during sampling to fill sequence and get
  compilation status.
  """
  def __init__(self,
               model           : model.BertPreTrainedModel,
               atomizer        : atomizers.AtomizerBase,
               use_categorical : bool,
               temperature     : float,
               ):
    self.model           = model
    self.atomizer        = atomizer
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
      stdout = opencl.Compile(self.atomizer.ArrayToCode(sample))
      return 1
    except ValueError:
      return 0

  def generateTrainingBatch(self,
                            input_ids         : torch.LongTensor,
                            prediction_scores : torch.FloatTensor,
                            attention_mask    : torch.LongTensor,
                            position_ids      : torch.LongTensor,
                            masked_lm_labels  : torch.LongTensor,
                            ) -> typing.Tuple[np.array, typing.List[int], typing.LongTensor]:
    batch_size, sequence_length = tuple(input_ids.shape)
    with concurrent.futures.ThreadPoolExecutor() as executor:
      jobs = [executor.submit(self.iterTrainingSeq,
                              batch           = input_ids        [i].cpu(),
                              prediction      = prediction_scores[i].detach().cpu(),
                              attention       = attention_mask   [i].cpu(),
                              position_ids    = position_ids     [i].cpu().unsqueeze(0),
                              masked_lm_label = masked_lm_labels [i].cpu().numpy()
                          ) for i in range(batch_size)]

      results          = [j.result() for j in jobs]
      samples          = [x.numpy() for (x, _, _) in results]
      compile_flag     = [y         for (_, y, _) in results]
      masked_lm_labels = torch.LongTensor([z for (_, _, z) in results]).to(pytorch.device)
      return samples, compile_flag, masked_lm_labels

  def generateSampleBatch(self,
                          input_ids         : torch.LongTensor,
                          prediction_scores : torch.FloatTensor,
                          attention_mask    : torch.LongTensor,
                          position_ids      : torch.LongTensor,
                          ) -> typing.Tuple[np.array, typing.List[int], typing.LongTensor]:
    num_targets = sum([x for x in input_ids[0] if x == self.atomizer.maskToken or x == self.atomizer.holeToken])
    sample_indices = [[[] for i in range(num_targets)] for j in range(len(input_ids))]
    ###
    batch_size, sequence_length = tuple(input_ids.shape)
    with concurrent.futures.ThreadPoolExecutor() as executor:
      jobs = [executor.submit(self.iterSampleSeq,
                              batch          = input_ids        [i].cpu(),
                              prediction     = prediction_scores[i].detach().cpu(),
                              attention      = attention_mask   [i].cpu(),
                              position_ids   = position_ids     [i].cpu().unsqueeze(0),
                              sample_indices = sample_indices   [i]
                          ) for i in range(batch_size)]

      results          = [j.result() for j in jobs]
      samples          = [x.numpy() for (x, _, _) in results]
      compile_flag     = [y         for (_, y, _) in results]
      sample_indices   = [z for (_, _, z) in results]
      return samples, compile_flag, sample_indices

  def iterTrainingSeq(self,
                      batch           : np.array,
                      prediction      : np.array,
                      attention       : np.array,
                      position_ids    : np.array,
                      masked_lm_label : np.array,
                      ) -> typing.Tuple[np.array, np.array, np.array]:
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
    holes, new_batch, new_attention = self.StepTrainingSeq(
      batch, prediction, attention
    )
    while holes:
      new_prediction, new_seq_relationship_score = self.model.get_output(
        new_batch.to(pytorch.device), new_attention.to(pytorch.device), position_ids.to(pytorch.device),
      )
      holes, new_batch, new_attention = self.StepTrainingSeq(
        new_batch[0],
        new_prediction.detach().cpu()[0],
        new_attention,
      )
    compile_flag = self.checkIfBatchCompiles(new_batch[0].numpy())
    if compile_flag and masked_lm_label:
      for idx, t in enumerate(masked_lm_label):
        if t != -100:
          masked_lm_label[idx] = -100
    return new_batch[0], compile_flag, masked_lm_label

  def StepTrainingSeq(self,
                      batch             : np.array,
                      prediction_scores : np.array,
                      attention_mask    : np.array,
                      ) -> typing.Tuple[bool, torch.LongTensor, np.array]:
    """
    Applies step predictions to input sequence.
    Specifically optimized for training; does not compute sample indices for speed-up.
    """
    seq_length    = tuple(batch.shape)[0]
    allowed_incr = (seq_length - int(torch.where(batch==self.atomizer.padToken)[0][0])
                    if self.atomizer.padToken in batch
                    else 0)

    endTokens = [self.atomizer.endholeToken, self.atomizer.maskToken, self.atomizer.holeToken]
    closed_hole = np.zeros(seq_length, dtype=np.bool)
    new_hole = np.zeros(seq_length, dtype=np.bool)
    temp_batch = batch.cpu().detach().numpy().copy()

    for target_idx in torch.where((batch == self.atomizer.holeToken) | (batch == self.atomizer.maskToken))[0]:
      idx        = int(target_idx)
      prediction = int(self.argmax(prediction_scores[target_idx]))
      is_hole = temp_batch[idx] == self.atomizer.holeToken

      if prediction in endTokens:
        # Model predicted sth that will close the hole.
        closed_hole[idx] = True
        continue

      # We replace the hole with a prediction
      temp_batch[idx] = prediction
      rem_adds = allowed_incr + np.sum(closed_hole) - np.sum(new_hole)
      if is_hole and rem_adds:
        # if this was a hole and we have more empty space, reinsert the hole
        new_hole[idx] = True

    new_batch = np.full(seq_length, self.atomizer.padToken, dtype=np.int64)
    new_idx = 0
    for idx, t in enumerate(temp_batch):
      if closed_hole[idx]:
        continue
      new_batch[new_idx] = t
      new_idx += 1
      if new_hole[idx]:
        new_batch[new_idx] = self.atomizer.holeToken
        new_idx += 1
      if new_idx >= seq_length:
        break

    new_batch = torch.LongTensor([new_batch])
    attention_mask = (new_batch != self.atomizer.padToken)
    return np.any(new_hole), new_batch, attention_mask

  def iterSampleSeq(self,
                    batch           : np.array,
                    prediction      : np.array,
                    attention       : np.array,
                    position_ids    : np.array,
                    sample_indices  : np.array,
                    ) -> typing.Tuple[np.array, np.array, np.array]:
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
    holes, new_batch, new_attention, sample_indices = self.StepSampleSeq(
      batch, prediction, attention
    )
    while holes:
      new_prediction, new_seq_relationship_score = self.model.get_output(
        new_batch.to(pytorch.device), new_attention.to(pytorch.device), position_ids.to(pytorch.device),
      )
      holes, new_batch, new_attention, sample_indices = self.StepSampleSeq(
        new_batch[0],
        new_prediction.detach().cpu()[0],
        new_attention,
      )
    compile_flag = self.checkIfBatchCompiles(new_batch[0].numpy())
    return new_batch[0], compile_flag, sample_indices

  def StepSampleSeq(self,
                    input_ids      : np.array,
                    predictions    : np.array,
                    attention_mask : np.array,
                    sample_indices : np.array,
                    ):
    """
    Updates new_input_ids with the model's output prediction.
    The output, if still contains hole or mask tokens, is fed back
    to the model's input through the input_fn's sample_gen generator.
    """
    masked_lm_ids = [
                      [x for idx, x in enumerate(predictions[batch_idx])
                          if input_ids[batch_idx][idx] == self.atomizer.maskToken
                          or input_ids[batch_idx][idx] == self.atomizer.holeToken
                      ] for batch_idx in range(len(input_ids))
                    ]
    assert len(input_ids) == len(masked_lm_ids), "Inputs and predictions do not have the same batch size."

    updated_sequence = []
    there_is_target = False
    for batch_idx, _ in enumerate(input_ids):
      batch = []
      mask_id_index     = 0
      closed_hole_index = 0
      for idx, token in enumerate(input_ids[batch_idx]):
        if   token == self.atomizer.maskToken:
          there_is_target = True
          mt = masked_lm_ids[batch_idx][mask_id_index]
          if mt == self.atomizer.maskToken or mt == self.atomizer.holeToken:
            continue
          if len(sample_indices[batch_idx][mask_id_index]) > 0:
            while(sample_indices[batch_idx][mask_id_index + closed_hole_index][-1]) == self.atomizer.endholeToken:
              closed_hole_index += 1
          sample_indices[batch_idx][mask_id_index + closed_hole_index].append(int(mt.cpu().numpy()))
          mask_id_index += 1
          batch.append(mt)
        elif token == self.atomizer.holeToken:
          there_is_target = True
          mt = masked_lm_ids[batch_idx][mask_id_index]
          if mt == self.atomizer.maskToken or mt == self.atomizer.holeToken:
            continue
          if len(sample_indices[batch_idx][mask_id_index]) > 0:
            while(sample_indices[batch_idx][mask_id_index + closed_hole_index][-1]) == self.atomizer.endholeToken:
              closed_hole_index += 1
          sample_indices[batch_idx][mask_id_index + closed_hole_index].append(int(mt.cpu().numpy()))
          mask_id_index += 1
          if mt != self.atomizer.endholeToken:
            batch.append(mt)
            batch.append(self.atomizer.holeToken)
            # done = False
        else:
          batch.append(token)

      while len(batch) < len(input_ids[batch_idx]):
        batch.append(self.atomizer.padToken)
      batch = batch[:len(input_ids[batch_idx])]

      pad_idx = None
      if self.atomizer.padToken in batch:
        pad_idx = batch.index(self.atomizer.padToken)
      attention_mask[batch_idx] = (torch.full([len(input_ids[0])], 1, dtype = torch.int64)
                        if pad_idx is None else
                        torch.cat(
                            (torch.full([pad_idx], 1, dtype = torch.int64),
                             torch.full([len(input_ids[batch_idx]) - pad_idx], 0, dtype = torch.int64)
                            )
                          )
                        )

      batch = batch[:len(input_ids[0])]
      updated_sequence.append(batch)
    new_input_ids = torch.LongTensor(updated_sequence).to(pytorch.device)
    return there_is_target, new_input_ids, attention_mask, sample_indices
