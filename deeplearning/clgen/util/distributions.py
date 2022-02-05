"""Statistical distributions used for sampling"""
import pathlib
import typing
import numpy as np

from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen.util import plotter

class Distribution():
  def __init__(self, 
               sample_length  : int,
               relative_length: float,
               log_path       : typing.Union[pathlib.Path, str],
               set_name       : str
               ):
    self.sample_length   = sample_length
    self.relative_length = relative_length
    self.log_path        = log_path if isinstance(log_path, pathlib.Path) else pathlib.Path(log_path)
    self.set_name        = set_name
    self.sample_counter  = {}
    return

  @classmethod
  def FromHoleConfig(cls, 
                     config: model_pb2.Hole,
                     log_path: typing.Union[pathlib.Path, str],
                     set_name: str,
                     ) -> typing.TypeVar("Distribution"):
    if config.HasField("absolute_length"):
      abs_len = config.absolute_length
      rel_len = 1.0
    elif config.HasField("relative_length"):
      abs_len = None
      rel_len = float(config.relative_length)

    if config.HasField("uniform_distribution"):
      return UniformDistribution(abs_len,
                                 rel_len,
                                 log_path,
                                 set_name,
                                 )
    elif config.HasField("normal_distribution"):
      return NormalDistribution(abs_len,
                                rel_len,
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
      plot_name = self.set_name,
      path      = self.log_path,
      title     = self.set_name,
      x_name    = self.set_name,
    )
    return

class UniformDistribution(Distribution):
  """
  A uniform distribution sampler. Get a random number from distribution calling sample()
  Upper range of sampling is defined as [0, sample_length].
  """
  def __init__(self, 
               sample_length  : int,
               relative_length: float,
               log_path       : typing.Union[pathlib.Path, str],
               set_name       : str
               ):
    super(UniformDistribution, self).__init__(sample_length, relative_length, log_path, set_name)

  def sample(self, length = None):
    if self.sample_length:
      return np.random.RandomState().randint(0, self.sample_length + 1)
    elif length:
      return np.random.RandomState().randint(0, int(length * self.relative_length))
    else:
      raise ValueErrror("One of sample length and upper length must be specified.")

class NormalDistribution(Distribution):
  """
  Normal distribution sampler. Initialized with mean, variance.
  Upper range of sampling is defined as [0, sample_length].
  """
  def __init__(self,
               sample_length  : int,
               relative_length: float,
               mean           : float,
               variance       : float,
               log_path       : typing.Union[pathlib.Path, str],
               set_name       : str,
               ):
    super(NormalDistribution, self).__init__(sample_length, relative_length, log_path, set_name)
    self.mean     = mean
    self.variance = variance

  def sample(self, length = None):
    upper_length = self.sample_length or length * self.relative_length
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

class UnknownDistribution(Distribution):
  """
  A small sample distribution of datapoints
  that we don't know what distribution they follow. Used
  to perform statistics on small samples.
  """
  @classmethod
  def FromConvolution(d1: "UnknownDistribution", d2: "UnknownDistribution") -> "UnknownDistribution":
    ## If you take the raw samples, it makes more sense to use dot-dot-multiplication and then construct the distribution.
    ## If you use the dictionaries, maybe you should do convolution
    return

  def __init__(self, samples: typing.List[int]):
    super(UnknownDistribution, self).__init__(
      sample_length   = len(samples),
      relative_length = float('NaN'),
      log_path        = log_path,
      set_name        = set_name,
    )
    self.distribution = {}
    for s in samples:
      if s in self.distribution:
        self.distribution[s] = 1
      else:
        self.distribution[s] += 1
    return
