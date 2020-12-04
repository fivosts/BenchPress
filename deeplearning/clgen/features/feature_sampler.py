"""
Feature space sampling of source code.
"""
import typing
import pathlib

from deeplearning.clgen.features import extractor

from eupy.native import logger as l

class FeatureSampler(object):
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
          self.benchmarks.append(
            FeatureSampler.Benchmark(
              f,
              f.name,
              contents,
              features,
              0
            )
          )

    # for benchmark in self.benchmarks:
    #   l.getLogger().info(benchmark.feature_vector)

  def calculate_proximity(self, infeat, tarfeat):
    try:
      return abs((tarfeat - infeat) / tarfeat)
    except ZeroDivisionError:
      return abs( ((tarfeat + 0.01) - infeat) / (tarfeat + 0.01))

  def sample_from_set(self, input_feature: typing.Dict[str, float]) -> bool:
    min_proximity = None
    min_idx = None
    for idx, benchmark in enumerate(self.benchmarks):
      proximity_vector = [self.calculate_proximity(infeat, tarfeat) for (_, infeat), (_, tarfeat) in zip(input_feature.items(), benchmark.feature_vector.items())]
      avg_proximity = sum(proximity_vector) / len(proximity_vector)
      if min_proximity is None:
        min_proximity = avg_proximity
        min_idx = idx
      else:
        if avg_proximity < min_proximity:
          min_proximity = avg_proximity
          min_idx = idx

    threshold = 0.4
    if min_proximity <= threshold:
      self.benchmarks[min_idx] = self.benchmarks[min_idx]._replace(times_achieved = 1 + self.benchmarks[min_idx].times_achieved)
      return True
    return False
