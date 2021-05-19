"""
Feature space sampling of source code.
"""
import typing
import pathlib
import math

from deeplearning.clgen.features import extractor
from eupy.native import logger as l

from absl import flags
from eupy.native import logger as l

FLAGS = flags.FLAGS

class EuclideanSampler(object):
  """
  This is a shitty experimental class to work with benchmark comparison.
  Will be refactored obviously.
  """
  class Benchmark(typing.NamedTuple):
    path: pathlib.Path
    name: str
    contents: str
    feature_vector: typing.Dict[str, float]
    times_achieved: int

  def __init__(self):
    self.path = pathlib.Path("./rodinia_benchmarks").resolve()
    self.benchmarks = []
    for f in self.path.iterdir():
      with open(f, 'r') as file:
        contents = file.read()
        features = extractor.DictKernelFeatures(contents)
        if features:
          l.getLogger().warn(features)
          self.benchmarks.append(
            EuclideanSampler.Benchmark(
              f,
              f.name,
              contents,
              features,
              0
            )
          )

  def calculate_distance(self, infeat) -> float:
    """
    Comparing all benchmarks, return the minimum euclidan distance
    of the input feature vector, among any of the target feature vector.
    """
    d = 0
    for b in self.benchmarks:
      # cd = 0
      for key in b.feature_vector.keys():
        i = infeat[key]
        t = b.feature_vector[key]
        d += abs((t**2) - (i**2))
      # d += cd
    return math.sqrt(d)

  def topK_candidates(self,
                      candidates: typing.List[typing.TypeVar("ActiveSample")],
                      K : int,
                      ) -> typing.List[typing.TypeVar("ActiveSample")]:
    """
    Return top-K candidates.
    """
    return sorted(candidates, key = lambda x: x.score)[:K]

  def sample_from_set(self, 
                      candidates: typing.List[typing.TypeVar("ActiveSample")],
                      ) -> bool:
    """
    Find top K candidates by getting minimum
    euclidean distance from set of rodinia benchmarks.
    """
    for idx in range(len(candidates)):
      candidates[idx] = candidates[idx]._replace(
        score = self.calculate_distance(candidates[idx].features)
      )
    return self.topK_candidates(candidates, FLAGS.active_search_width)
