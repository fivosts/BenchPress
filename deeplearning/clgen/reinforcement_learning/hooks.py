"""
Hook class to monitor reinforcement learning agent's learning process.
"""
import json
import numpy as np
import pathlib

from deeplearning.clgen.util import plotter
from deeplearning.clgen.util import logging as l

class tensorMonitorHook(object):
  def __init__(self, 
               cache_path   : pathlib.Path, 
               current_step : int, 
               step_freq    : int,
               flush_freq   : int = None,
               average      : bool = True,
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
