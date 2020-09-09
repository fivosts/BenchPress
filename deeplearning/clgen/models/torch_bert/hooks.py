import json
import numpy as np
import pathlib

from deeplearning.clgen import validation_database
from eupy.native import logger as l
from eupy.native import plotter as plt

class torchTrainingHook(object):
  def __init__(self, 
               cache_path: pathlib.Path, 
               current_step: int, 
               step_freq: int
               ):
    self.cache_path    = cache_path
    self.jsonfile      = cache_path / "training.json"
    self.current_step  = current_step
    self.step_freq     = step_freq
    self.tensors       = []
    self.epoch_tensors = {}
    self._initTensors()

    self.monitor_func = [
      self._tensor2JSON,
      self._tensor2plot,
    ]
    return

  def step(self, **tensors):

    for key, value in tensors.items():
      if key in self.epoch_tensors:
        self.epoch_tensors[key] += value
      else:
        self.epoch_tensors[key] = value

    if self._step_triggered():
      self._logTensors()
    return

  def _initTensors(self):
    if current_step >= 0:
      if self.jsonfile.exists():
        with open(self.jsonfile, 'r') as js:
          self.tensors = json.load(js)
      else:
        raise FileNotFoundError(self.jsonfile)
    return

  def _step_triggered(self):
    self.current_step += 1
    if (self.current_step - 1) % self.step_freq == 0:
      return True
    return False

  def _logTensors(self):
    if self.current_step - 1 == 0:
      self.tensors.append(self.epoch_tensors)
    else:
      self.tensors.append(
        {k: v / self.step_freq for k, v in self.epoch_tensors.items()}
      )
    for func in self.monitor_func:
      func()
    return

  def _tensor2JSON(self):
    with open(self.jsonfile, 'w') as js:
      js.dump(self.tensors, indent = 2, sort_keys = True)
    return

  def _tensor2plot(self):
    for (key, value) in self.tensors.items():
      plt.linesSingleAxis(
        {key: {'y': value, 'x': np.arange(len(self.tensors), step = self.step_freq)}},
        y_label = (key, 13),
        x_label = ("Train step", 13),
        plot_title = (key, 20),
        x_lim   = [0, value['step'][-1] + 0.01 * value['step'][-1]],
        y_lim   = 1.1 * max(value['value']),
        legend  = False,
        showfig = False,
        savefig = str(self.cache_path / "{}.png".format(key)),
        force_init = True,
      )
    return