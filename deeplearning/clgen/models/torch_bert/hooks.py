import progressbar
import six 
import json
import humanize
import numpy as np
import glob
import pathlib

from deeplearning.clgen import validation_database
from eupy.native import logger as l
from eupy.native import plotter as plt

"""
1) Write json file
2) Plot averaged metrics.
"""
class torchTrainingHook(object):
  def __init__(self, 
               cache_path: pathlib.Path, 
               current_step: int, 
               step_freq: int
               ):
    self.cache_path   = cache_path
    self.current_step = current_step
    self.step_freq    = step_freq
    self.tensors      = []
    self._initTensors()
    return

  def _initTensors(self):
    if current_step >= 0:
      if (self.cache_path / "training.json").exists():
        with open(self.cache_path / "training.json", 'r') as js:
          self.tensors = json.load(js)
      else:
        raise FileNotFoundError(self.cache_path / "training.json")
    return

  def _step_triggered(self):
    self.current_step += 1
    if (self.current_step - 1) % self.step_freq == 0:
      return True
    return False

  def step(self, **tensors):
    self.tensors.append(tensors)
    if self._step_triggered():
      self._logTensors()
    return

  def _logTensors(self):
    if self.current_step - 1 == 0:
      # log me as it is
      pass
    else:
      # Average me by step_freq and then log me
      pass