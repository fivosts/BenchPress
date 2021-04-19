"""Statistical distributions used for sampling"""
import pathlib
import typing
import numpy as np

from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen.util import plotter

class Distribution():
  def __init__(self, 
               sample_length: int, 
               log_path     : typing.Union[pathlib.Path, str],
               set_name     : str
               ):
    self.sample_length  = sample_length
    self.log_path       = log_path if isinstance(log_path, pathlib.Path) else pathlib.Path(log_path)
    self.set_name       = set_name
    self.sample_counter = {}
    return

  @classmethod
  def FromHoleConfig(cls, 
                     config: model_pb2.Hole,
                     log_path: typing.Union[pathlib.Path, str],
                     set_name: str,
                     ) -> typing.TypeVar("Distribution"):
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

  def sample(self, length = None):
    raise NotImplementedError

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
      path = self.log_path
    )
    return

class UniformDistribution(Distribution):
  """
  A uniform distribution sampler. Get a random number from distribution calling sample()
  Upper range of sampling is defined as [0, sample_length].
  """
  def __init__(self, 
               sample_length: int,
               log_path     : typing.Union[pathlib.Path, str],
               set_name     : str
               ):
    super(UniformDistribution, self).__init__(sample_length, log_path, set_name)

  def sample(self, length = None):
    return np.random.RandomState().randint(0, self.sample_length + 1)

class NormalDistribution(Distribution):
  """
  Normal distribution sampler. Initialized with mean, variance.
  Upper range of sampling is defined as [0, sample_length].
  """
  def __init__(self,
               sample_length: int,
               mean         : float,
               variance     : float,
               log_path     : typing.Union[pathlib.Path, str],
               set_name     : str,
               ):
    super(NormalDistribution, self).__init__(sample_length, log_path, set_name)
    self.mean     = mean
    self.variance = variance

  def sample(self, length = None):
    sample = int(round(np.random.RandomState().normal(loc = self.mean, scale = self.variance)))
    while sample < 0 or sample > self.sample_length:
      sample = int(round(np.random.RandomState().normal(loc = self.mean, scale = self.variance)))
    return sample

  class ProgLinearDistribution(Distribution):
    """
    A sampling distribution used in training per stage mode.
    Distribution starts with empty or tiny holes and
    gradually progresses into sampling bigger holes while still
    feeding small holes as well, until max hole length is met.
    Gradual increase is an interval based on number of stages
    and number of train steps.

    Cumulative stage distribution appears as negative linear.
    At any given moment, probability of selecting a hole
    length should be uniform.

    Parameters:
      number of stages
      number of training steps
      max hole length
    """
    def __init__(self,
                 num_train_steps : int,
                 max_hole_length : int,
                 log_path        : typing.Union[pathlib.Path, str],
                 set_name        : str,
                 ):
      super(ProgLinearDistribution, self).__init__(
        max_hole_length, log_path, set_name
      )
      self.num_train_steps = num_train_steps

    def sample(self):
      return
