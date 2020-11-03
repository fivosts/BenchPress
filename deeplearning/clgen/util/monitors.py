"""Statistical distributions used for sampling"""
import pathlib
import typing
import numpy as np

from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen.util import plotter

class Monitor():
  def __init__(self, 
               cache_path : typing.Union[pathlib.Path, str],
               set_name   : str
               ):
    self.cache_path     = cache_path if isinstance(cache_path, pathlib.Path) else pathlib.Path(cache_path)
    self.set_name       = set_name
    self.sample_counter = {}
    return

  def register(self, actual_sample):
    raise NotImplementedError("Abstract Class")

  def sample(self):
    raise NotImplementedError("Abstract Class")

  def plot(self):
    raise NotImplementedError("Abstract Class")

class FrequencyMonitor(Monitor):
  """
  Not an actual sampling distribution.
  This subclass is used to register values of a specific type, keep track of them
  and bar plot them. E.g. length distribution of a specific encoded corpus.
  """
  def __init__(self, 
               cache_path: typing.Union[pathlib.Path, str], 
               set_name  : str,
               ):
    super(FrequencyMonitor, self).__init__(cache_path, set_name)
    return

  def register(self, actual_sample):
    if isinstance(actual_sample, list):
      for s in actual_sample:
        self.register(s)
    else:
      if actual_sample not in self.sample_counter:
        self.sample_counter[actual_sample] =  1
      else:
        self.sample_counter[actual_sample] += 1
    return

  def plot(self):
    sorted_dict = sorted(self.sample_counter.items(), key = lambda x: x[0])
    plotter.FrequencyBars(
      x = [x for (x, _) in sorted_dict],
      y = [y for (_, y) in sorted_dict],
      title     = self.set_name,
      x_name    = self.set_name,
      plot_name = self.set_name,
      path = self.cache_path
    )
    return

class HistoryMonitor(Monitor):
  """
  Not an actual sampling distribution.
  In contrast to the rest, this subclass does not count frequency of a certain value.
  It registers values and plots them against time.
  """
  def __init__(self, 
               cache_path: typing.Union[pathlib.Path, str], 
               set_name: str,
               ):
    super(HistoryMonitor, self).__init__(cache_path, set_name)
    self.sample_list = []
    return

  def register(self, actual_sample: int) -> None:
    # if isinstance(actual_sample, str):
    #   actual_sample = int(actual_sample)
    self.sample_list.append(float(actual_sample))
    return

  def plot(self):
    plotter.SingleScatterLine(
      x = np.arange(len(self.sample_list)),
      y = self.sample_list,
      title = self.set_name,
      x_name = "",
      y_name = self.set_name,
      plot_name = self.set_name,
      path = self.cache_path,
    )
    return
