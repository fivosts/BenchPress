"""Statistical distributions used for sampling"""
import pathlib
import numpy as np

from deeplearning.clgen.proto import model_pb2
from eupy.native import plotter as plt

class Distribution():
  def __init__(self, sample_length, log_path, set_name):
    self.sample_length  = sample_length
    self.log_path       = log_path
    self.set_name       = set_name
    self.sample_counter = {}
    return

  @classmethod
  def FromHoleConfig(cls, 
                     config: model_pb2.Hole,
                     log_path: pathlib.Path,
                     set_name: str,
                     ):
    if config.HasField("uniform_distribution"):
      return UniformDistribution(config.hole_length,
                                 log_path,
                                 set_name,
                                 )
    elif config.HasField("normal_distribution"):
      return NormalDistribution(config.hole_length, 
                                config.normal_distribution.mean, 
                                config.normal_distribution.variance,
                                log_path,
                                set_name,
                                )
    else:
      raise NotImplementedError(config)

  def sample(self):
    raise NotImplementedError

  def register(self, actual_sample):
    if actual_sample not in self.sample_counter:
      self.sample_counter[actual_sample] =  1
    else:
      self.sample_counter[actual_sample] += 1
    return

  def plot(self):
    sorted_dict = sorted(self.sample_counter.items(), key = lambda x: x[0])
    point_set = [
      {
        'x': [[x for (x, _) in sorted_dict]],
        'y': [[y for (_, y) in sorted_dict]],
        'label': [str(y) for (_, y) in sorted_dict],
      }
    ]
    from eupy.native import logger as l
    plt.plotBars(
      point_set, save_file = True, file_path = str(self.log_path / self.set_name),
      show_xlabels = True, file_extension = ""
    )
    return

class UniformDistribution(Distribution):
  def __init__(self, sample_length, log_path, set_name):
    super(UniformDistribution, self).__init__(sample_length, log_path, set_name)
    self.sampler = lambda : np.random.randint(0, sample_length + 1)

  def sample(self):
    sample = self.sampler()
    return sample

class NormalDistribution(Distribution):
  def __init__(self, sample_length, mean, variance, log_path, set_name):
    super(NormalDistribution, self).__init__(sample_length, log_path, set_name)
    self.mean     = mean
    self.variance = variance
    self.sampler  = lambda : np.random.normal(loc = mean, scale = variance)

  def sample(self):
    sample = int(round(self.sampler()))
    while sample < 0 or sample > self.sample_length:
      sample = int(round(self.sampler()))
    return sample
