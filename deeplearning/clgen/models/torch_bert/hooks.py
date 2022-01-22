import json
import numpy as np
import pathlib

from deeplearning.clgen.util import plotter
from deeplearning.clgen.samplers import validation_database
from deeplearning.clgen.util import logging as l

class tensorMonitorHook(object):
  def __init__(self, 
               cache_path: pathlib.Path, 
               current_step: int, 
               step_freq: int,
               flush_freq: int = None,
               average: bool = True,
               ):
    self.cache_path       = cache_path
    self.current_step     = current_step
    self.step_freq        = step_freq
    self.flush_freq       = flush_freq
    self.average          = average

    self.jsonfile         = cache_path / "training.json"
    self.tensors          = []
    self.plot_tensors     = {}
    self.epoch_tensors    = {}
    self.epch_loss        = []
    self.delay_checkpoint = True if current_step != 0 else False
    self._initTensors()

    self.monitor_func = [
      self._tensor2JSON,
      self._tensor2plot,
    ]
    return

  @property
  def epoch_loss(self):
    return sum(self.epch_loss) / len(self.epch_loss)
  
  def step(self, **tensors):
    for key, value in tensors.items():
      if value is None:
        continue
      if key in self.epoch_tensors and "num_" not in key and not "val_" in key:
        # "num_" means tensor registers accumulated number and won't average.
        # Therefore we are just going to take the last registed value.
        self.epoch_tensors[key] += value
      else:
        self.epoch_tensors[key] = value

    self.current_step += 1
    if self._step_triggered():
      self._logTensors()
      self.epoch_tensors = {}
    return

  def end_epoch(self, **tensors):
    for key, value in tensors.items():
      if value is None:
        continue
      self.epoch_tensors[key] = value
    # if self._step_triggered():
    self._logTensors()
    self.epoch_tensors = {}
    self.epch_loss = []
    return

  def _initTensors(self):
    if self.current_step > 0:
      if self.jsonfile.exists():
        with open(self.jsonfile, 'r') as js:
          loaded_tensors = json.load(js)
          if loaded_tensors[-1]['step'] > self.current_step:
            # If previous sessions have written beyond current step, overwrite them.
            back_index = -2
            while loaded_tensors[back_index]['step'] > self.current_step:
              back_index -= 1
            self.tensors = loaded_tensors[:back_index + 1]
          else:
            self.tensors = loaded_tensors

          for ch in self.tensors:
            for k, v in ch.items():
              if k == 'step':
                continue
              if k not in self.plot_tensors:
                self.plot_tensors[k] = {'value': [], 'step': []}
              self.plot_tensors[k]['value'].append(v)
              self.plot_tensors[k]['step'].append(ch['step'])
      else:
        l.logger().error("Training json log-file not found. Will keep track from this point on.")
    return

  def _step_triggered(self):
    if self.delay_checkpoint:
      self.delay_checkpoint = False
      return False
    if (self.current_step) % self.step_freq == 0 or self.current_step - 1 == 0:
      return True
    return False

  def _logTensors(self):

    effective_step = self.current_step if self.current_step - 1 != 0 else 0
  
    if self.average is True:
      epoch_tensors = (self.epoch_tensors if effective_step == 0
                     else {k: (v / self.step_freq if not "num_" in k and not "val_" in k else v) for k, v in self.epoch_tensors.items()})
    else:
      epoch_tensors = (self.epoch_tensors if effective_step == 0
                     else {k: v for k, v in self.epoch_tensors.items()})

    self.tensors.append(epoch_tensors)
    self.tensors[-1]['step'] = effective_step
    if 'total_loss' in epoch_tensors:
      self.epch_loss.append(epoch_tensors['total_loss'])
    
    for key, value in epoch_tensors.items():
      if key == 'step':
        continue
      if key not in self.plot_tensors:
        self.plot_tensors[key] = {'value': [], 'step': []}
      self.plot_tensors[key]['value'].append(value)
      self.plot_tensors[key]['step'].append(effective_step)

    for func in self.monitor_func:
      func()
    return

  def _tensor2JSON(self):
    with open(self.jsonfile, 'w') as js:
      json.dump(self.tensors, js, indent = 2, sort_keys = True)
    return

  def _tensor2plot(self):
    for (key, value) in self.plot_tensors.items():
      if key != "step":
        plotter.SingleScatterLine(
          x = value['step'],
          y = value['value'],
          title = key,
          x_name = "Training Step",
          y_name = key,
          plot_name = key,
          path = self.cache_path,
        )
    return

class validationSampleHook(object):
  """Real time storage hook for validation results"""

  def __init__(self,
               url,
               tokenizer,
               model_step,
               ):

    self.tokenizer  = tokenizer
    self.val_db     = validation_database.ValidationDatabase(url)
    self.val_files  = {}
    self.val_id     = self.val_db.count
    self.model_step = model_step
    self.mask_accuracy = [0, 0]
    self.nsp_accuracy  = [0, 0]
    return

  def step(self,
           inputs,
           outputs,
           ) -> None:
    """
      Requested tensors are evaluated and their values are available
    """

    seen_in_training      = inputs['seen_in_training'].numpy()
    original_input        = inputs['original_input'].numpy()
    masked_lm_lengths     = inputs['masked_lm_lengths'].numpy()
    input_ids             = inputs['input_ids'].cpu().numpy()
    input_mask            = inputs['input_mask'].cpu().numpy()
    next_sentence_labels  = inputs['next_sentence_labels'].cpu().numpy()
    mask_labels           = inputs['mask_labels'].cpu().numpy()
    pred_logits           = outputs['prediction_logits'].cpu().numpy()
    seq_rel_logits        = outputs['seq_relationship_logits'].cpu().numpy()

    batch_size = len(pred_logits)

    masked_lm_ids = [[x for x in batch if x != -100] for batch in mask_labels]
    masked_lm_positions = [[idx for idx, x in enumerate(batch) if x != -100] for batch in mask_labels]
    masked_lm_predictions = [
          [np.argmax(pred_logits[batch][x]) for x in masked_lm_positions[batch]]
          for batch in range(batch_size)
        ]
    next_sentence_predictions = [[np.argmax(x) for x in batch][-1] for batch in seq_rel_logits]

    for target, prediction in zip(masked_lm_ids, masked_lm_predictions):
      if target == prediction:
        self.mask_accuracy[0] += 1
      self.mask_accuracy[1] += 1

    for target, prediction in zip(next_sentence_labels, next_sentence_predictions):
      if target == prediction:
        self.nsp_accuracy[0] += 1
      self.nsp_accuracy[1] += 1

    for b in range(batch_size):
      f = validation_database.BERTValFile(
        **validation_database.BERTValFile.FromArgs(
          tokenizer = self.tokenizer,
          id        = self.val_id,
          train_step                = self.model_step,
          seen_in_training          = seen_in_training[b],
          original_input            = original_input[b],
          input_ids                 = input_ids[b],
          input_mask                = input_mask[b],
          masked_lm_positions       = masked_lm_positions[b],
          masked_lm_ids             = masked_lm_ids[b],
          masked_lm_weights         = [],
          masked_lm_lengths         = masked_lm_lengths[b],
          next_sentence_labels      = next_sentence_labels[b],
          masked_lm_predictions     = masked_lm_predictions[b],
          next_sentence_predictions = next_sentence_predictions[b],
        )
      )
      if f.sha256 not in self.val_files:
        self.val_files[f.sha256] = f
        self.val_id += 1

    # with self.val_db.Session(commit = True) as session:
    #   for b in range(batch_size):
    #     val_trace = validation_database.BERTValFile(
    #       **validation_database.BERTValFile.FromArgs(
    #         tokenizer = self.tokenizer,
    #         id       = self.val_id,
    #         train_step                = self.model_step,
    #         seen_in_training          = seen_in_training[b],
    #         original_input            = original_input[b],
    #         input_ids                 = input_ids[b],
    #         input_mask                = input_mask[b],
    #         masked_lm_positions       = masked_lm_positions[b],
    #         masked_lm_ids             = masked_lm_ids[b],
    #         masked_lm_weights         = [],
    #         masked_lm_lengths         = masked_lm_lengths[b],
    #         next_sentence_labels      = next_sentence_labels[b],
    #         masked_lm_predictions     = masked_lm_predictions[b],
    #         next_sentence_predictions = next_sentence_predictions[b],
    #       )
    #     )
    #     try:
    #       exists = session.query(validation_database.BERTValFile.sha256).filter_by(sha256 = val_trace.sha256).scalar() is not None
    #     except sqlalchemy.orm.exc.MultipleResultsFound as e:
    #       l.logger().error("Selected sha256 has been already found more than once.")
    #       raise e
    #     if not exists:
    #       session.add(val_trace)
    #       self.val_id += 1
    return

  def final(self,
            val_set: str,
            masked_lm_loss: float,
            next_sentence_loss: float,
           ) -> None:
    if self.mask_accuracy[1] == 0 or self.nsp_accuracy[1] == 0:
      return

    masked_lm_accuracy = self.mask_accuracy[0] / self.mask_accuracy[1]
    next_sentence_accuracy = self.nsp_accuracy[0] / self.nsp_accuracy[1]
    r = [
      "masked_lm_accuracy: {}".format(masked_lm_accuracy),
      "masked_lm_loss: {}".format(masked_lm_loss),
      "next_sentence_accuracy: {}".format(next_sentence_accuracy),
      "next_sentence_loss: {}".format(next_sentence_loss),
    ]
    with self.val_db.Session(commit = True) as session:

      for f in self.val_files.values():
        session.add(f)

      exists = session.query(validation_database.ValResults.key).filter_by(key = val_set).scalar() is not None
      if exists:
        entry = session.query(validation_database.ValResults).filter_by(key = val_set).first()
        entry.results = "\n".join(r)
      else:
        session.add(validation_database.ValResults(key = val_set, results = "\n".join(r)))
    l.logger().info("LM Accuracy: {}, LM Loss: {}, NSP Accuracy: {}, NSP Loss: {}".format(
        masked_lm_accuracy,
        masked_lm_loss,
        next_sentence_accuracy,
        next_sentence_loss
      )
    )
    return
