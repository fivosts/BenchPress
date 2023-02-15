# coding=utf-8
# Copyright 2022 Foivos Tsimpourlas.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Statistical distributions used for sampling"""
import pathlib
import sys
import copy
import typing
import math
import numpy as np

from deeplearning.benchpress.proto import model_pb2
from deeplearning.benchpress.util import plotter

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
               set_name       : str,
               seed           : int = None,
               ):
    super(UniformDistribution, self).__init__(sample_length, relative_length, log_path, set_name)
    if seed:
      self.seed = seed
      self.sample_gen = np.random
      self.sample_gen.seed(seed)
      if self.sample_length:
        self.sampler = self.sample_gen.randint
      else:
        self.sampler = self.sample_gen.randint
    else:
      if self.sample_length:
        self.sampler = np.random.RandomState().randint
      else:
        self.sampler = np.random.RandomState().randint
    return

  def sample(self, length = None):
    if not self.sample_length and not length:
      raise ValueErrror("One of sample length and upper length must be specified.")
    if self.sample_length:
      return self.sampler(0, self.sample_length + 1)
    else:
      return self.sampler(0, int(length * self.relative_length))

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

class GenericDistribution(Distribution):
  """
  A small sample distribution of datapoints
  that we don't know what distribution they follow. Used
  to perform statistics on small samples.
  """
  @property
  def population_size(self) -> int:
    """
    Size of distribution's population.
    """
    return self.sample_length

  @property
  def population(self) -> typing.List[int]:
    """
    Get population.
    """
    return self.samples

  @property
  def min(self) -> int:
    return self.min_idx

  @property
  def max(self) -> int:
    return self.max_idx

  @property
  def average(self) -> float:
    if self.avg is not None:
      return self.avg
    else:
      self.avg = sum(self.population) / self.population_size
      return self.avg
      """
      avg = 0.0
      for idx, p in enumerate(self.distribution):
        avg += p * (idx + self.min_idx)
      self.avg = avg
      return self.avg
      """

  @property
  def median(self) -> int:
    if self.med is not None:
      return self.med
    else:
      s = sorted(self.population)
      if self.population_size % 2 == 1:
        self.med = s[self.population_size // 2]
      else:
        self.med = 0.5 * (s[(self.population_size // 2) - 1] + s[self.population_size // 2])
      return self.med
      """
      l_idx, r_idx = 0, len(self.distribution)
      l,r = self.distribution[l_idx], None

      queue = copy.copy(self.distribution)
      cur = queue.pop(0)
      offset = -cur
      if cur != 0:
        l = cur
      while queue:
        if offset < 0:
          cur    = queue.pop()
          r_idx -= 1
          if cur != 0:
            r       = r_idx
            offset += cur
        else:
          cur    = queue.pop(0)
          l_idx += 1
          if cur != 0:
            l    = l_idx
            offset -= cur
      if offset > sys.float_info.epsilon:
        self.med = r + self.min_idx
      elif offset < -sys.float_info.epsilon:
        self.med = l + self.min_idx
      else:
        self.med = (l+r+2*self.min_idx) / 2
      return self.med
      """

  @property
  def variance(self) -> float:
    """
    Calculate variance of population.
    """
    if self.var is not None:
      return self.var
    else:
      self.var = sum([(x - self.average)**2 for x in self.population]) / self.population_size
      return self.var

  @property
  def standard_deviation(self) -> float:
    return math.sqrt(self.variance)

  def __init__(self, samples: typing.List[int], log_path: pathlib.Path, set_name: str):
    super(GenericDistribution, self).__init__(
      sample_length   = len(samples),
      relative_length = float('NaN'),
      log_path        = log_path,
      set_name        = set_name,
    )
    self.min_idx, self.max_idx = math.inf, -math.inf
    self.avg, self.med, self.var = None, None, None
    self.samples = samples
    total = len(samples)
    if len(samples) > 0:
      for s in samples:
        if s > self.max_idx:
          self.max_idx = s
        if s < self.min_idx:
          self.min_idx = s
      # If there is a large discrepancy in the min/max range, this array will be massive.
      # You could turn this to a dict for only non-zero keys.
      self.distribution = [0] * abs(1 + self.max_idx - self.min_idx)
      for s in samples:
        self.distribution[s - self.min_idx] += 1
      for idx, v in enumerate(self.distribution):
        self.distribution[idx] = v / total
      self.pmf_to_pdf()
    else:
      self.distribution = []
      self.pdf = []
    return

  def __add__(self, d: "GenericDistribution") -> "GenericDistribution":
    """
    The addition of two distribution happens with convolution.
    For two discrete distributions d1, d2:

    P[X1=X + X2=Y] = P[X1=X] ** P[X2=Y] = Σn Σk (Pd1[k] * Pd2[n-k])
    """
    if self.min_idx > d.min_idx:
      d1 = self.realign(self.min_idx - d.min_idx)
      d2 = d.distribution
    else:
      d1 = self.distribution
      d2 = d.realign(d.min_idx - self.min_idx)

    if len(d1) > len(d2):
      d2 = d2 + [0] * (len(d1) - len(d2))
    else:
      d1 = d1 + [0] * (len(d2) - len(d1))

    ret = GenericDistribution([], self.log_path, "{}+{}".format(self.set_name, d.set_name))
    summed  = list(np.convolve(d1, d2, mode = 'full'))

    while summed[0] == 0:
      summed.pop(0)
    while summed[-1] == 0:
      summed.pop()

    min_idx = self.min_idx + d.min_idx
    max_idx = len(summed) - 1 + min_idx

    ret.distribution = summed
    ret.min_idx = min_idx
    ret.max_idx = max_idx
    ret.pmf_to_pdf()
    return ret

  def __sub__(self, d: "GenericDistribution") -> "GenericDistribution":
    """
    Subtraction of distributions is equal to addition of inverted distribution.
    P[X - Y] = P[X + (-Y)]
    """
    neg = d.negate()
    sub = self + neg
    sub.set_name = "{}-{}".format(self.set_name, d.set_name)
    return sub

  def __mul__(self, d: "GenericDistribution") -> "GenericDistribution":
    """
    Multiplication of two independent random variables.
    P[X*Y = c] = Σx Σy P[X=x]*P[Y=y]
    """
    l_idx = self.min_idx*d.min_idx
    r_idx = self.max_idx*d.max_idx
    out_distr = [0] * (r_idx - l_idx)
    for x in range(self.min_idx, self.max_idx + 1):
      for y in range(d.min_idx, d.max_idx + 1):
        out_distr[x*y - l_idx] = self.distribution[x] * d.distribution[y]

    while out_distr[0] == 0:
      out_distr.pop(0)
      l_idx += 1
    while out_distr[-1] == 0:
      out_distr.pop()
      r_idx -= 1

    mul = GenericDistribution([], self.log_path, "{}*{}".format(self.set_name, d.set_name))
    mul.distribution = out_distr
    mul.min_idx = l_idx
    mul.max_idx = r_idx
    return mul

  def __ge__(self, v: int) -> float:
    """
    Probability of P[X >= v]
    """
    voffset = v - self.min_idx
    probs = 0.0
    for idx, s in enumerate(self.distribution):
      if idx >= voffset:
        probs += s
    return probs

  def __gt__(self, v: int) -> float:
    """
    Probability of P[X > v]
    """
    voffset = v - self.min_idx
    probs = 0.0
    for idx, s in enumerate(self.distribution):
      if idx > voffset:
        probs += s
    return probs

  def __le__(self, v: int) -> float:
    """
    Probability of P[X <= v]
    """
    voffset = v - self.min_idx
    probs = 0.0
    for idx, s in enumerate(self.distribution):
      if idx <= voffset:
        probs += s
    return probs

  def __lt__(self, v: int) -> float:
    """
    Probability of P[X < v]
    """
    voffset = v - self.min_idx
    probs = 0.0
    for idx, s in enumerate(self.distribution):
      if idx < voffset:
        probs += s
    return probs

  def __eq__(self, v: int) -> float:
    """
    Probability of P[X = v]
    """
    voffset = v - self.min_idx
    probs = 0.0
    for idx, s in enumerate(self.distribution):
      if idx == voffset:
        probs += s
    return probs

  def negate(self) -> "GenericDistribution":
    """
    Inverts distribution: P[Y] -> P[-Y]
    """
    neg = GenericDistribution([], self.log_path, "neg-{}".format(self.set_name))
    neg.distribution = self.distribution[::-1]
    neg.min_idx = -self.max_idx
    neg.max_idx = -self.min_idx
    neg.pmf_to_pdf()
    return neg

  def realign(self, offset: int) -> typing.List[int]:
    """
    When performing operations with distributions,
    both distributions must have the same reference as to
    what index does array's 0th index refers to.

    This function slides to the right, the index array
    that is leftmost, and aligns it with the others.
    """
    return [0] * offset + self.distribution

  def cov(self, d: "GenericDistribution") -> float:
    """
    Compute covariance of two distributions.
    """
    if self.population_size != d.population_size:
      raise ValueError("Covariance and correlation can only be computed to 1-1 equal-sized distributions. Or you could take two equal-sized samples.")
    return sum([(x - self.average)*(y - d.average) for (x, y) in zip(self.population, d.population)]) / self.population_size

  def corr(self, d: "GenericDistribution") -> float:
    """
    Compute correlation factor between two distributions.
    """
    try:
      return self.cov(d) / (self.standard_deviation * d.standard_deviation)
    except ZeroDivisionError:
      return math.inf

  def get_sorted_index(self, idx: int) -> int:
    """
    Get the smallest 'idx' sample in the population.
    """
    return sorted(self.samples)[min(len(self.samples) - 1, idx)]

  def pmf_to_pdf(self) -> None:
    """
    Compute pdf from pmf.
    """
    self.pdf = [0] * len(self.distribution)
    cur = 0.0
    for idx, prob in enumerate(self.distribution):
      cur += prob
      self.pdf[idx] = cur
    return

  def plot(self, with_avg: bool = False) -> None:
    """
    Plot distribution.
    """
    vlines = None
    if with_avg:
      vlines = [
        (self.average, "Average"),
        (self.median, "Median"),
      ]
    plotter.FrequencyBars(
      x = [idx + self.min_idx for idx, _ in enumerate(self.distribution)],
      y = [v for v in self.distribution],
      plot_name = "pmf_{}".format(self.set_name),
      path      = self.log_path,
      title     = "pmf_{}".format(self.set_name),
      x_name    = self.set_name,
      vlines = vlines,
    )
    plotter.FrequencyBars(
      x = [idx + self.min_idx for idx, _ in enumerate(self.pdf)],
      y = [v for v in self.pdf],
      plot_name = "pdf_{}".format(self.set_name),
      path      = self.log_path,
      title     = "pdf_{}".format(self.set_name),
      x_name    = self.set_name,
      vlines = vlines,
    )
    return
