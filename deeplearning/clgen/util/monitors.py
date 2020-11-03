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
    return

  def register(self, actual_sample):
    raise NotImplementedError("Abstract Class")

  def sample(self):
    raise NotImplementedError("Abstract Class")

  def plot(self):
    raise NotImplementedError("Abstract Class")

class FrequencyMonitor(Monitor):
  """
  Keeps monitor of the occured frequency of a specific key.
  Key is provided through `actual_sample` in register method.
  Its frequency is incremented by one.

  Bar plots num of occurences VS keys.
  """
  def __init__(self, 
               cache_path: typing.Union[pathlib.Path, str], 
               set_name  : str,
               ):
    super(FrequencyMonitor, self).__init__(cache_path, set_name)
    self.sample_counter = {}
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
    """Plot bars of number of occurences."""
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
  Monitors values in an ordered timeline
  Plots a line of values against timesteps.
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
    """Plot line over timescale"""
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

class FeatureMonitor(Monitor):
  """
  Produces a Radar chart of normalized features.
  """
  def __init__(self,
               cache_path: typing.Union[pathlib.Path, str],
               set_name: str,
               ):
    super(FeatureMonitor, self).__init__(cache_path, set_name)
    self.features = {}
    self.instance_counter = 0
    return

  def register(self, actual_sample: typing.Dict[str, float]) -> None:
    """actual sample is a dict of features to their values."""
    if not isinstance(actual_sample, dict):
      raise TypeError("Feature sample must be dictionary of string features to float values.")

    self.instance_counter += 1
    for k, v in actual_sample.items():
      if k not in self.features:
        self.features[k] = v
      else:
        self.features[k] += v
    return

  def plot(self):
    """Plot averaged Radar chart"""
    r = [v / self.instance_counter for _, v in self.features.items()]
    theta = [k for k, _ in self.features.items()]
    plotter.NormalizedRadar(
      r = r,
      theta = theta,
      title = self.set_name,
      plot_name = self.set_name,
      path = self.cache_path,
    )
    return
