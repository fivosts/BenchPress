"""
Feature space sampling of source code.
"""
import typing
import pathlib

from deeplearning.clgen.features import extractor

from eupy.native import logger as l

def is_kernel_bigger(input_feature: typing.Dict[str, float],
                     sample_feature: typing.Dict[str, float],
                     ) -> bool:
  """
  Checks if sample kernel is larger than original feed
  by comparing all numerical features.
  """
  return (sample_feature['comp']      + sample_feature['rational'] +
          sample_feature['mem']       + sample_feature['localmem'] +
          sample_feature['coalesced'] + sample_feature['atomic']
          > 
          input_feature['comp']      + input_feature['rational'] +
          input_feature['mem']       + input_feature['localmem'] +
          input_feature['coalesced'] + input_feature['atomic'])

def is_kernel_smaller(input_feature: typing.Dict[str, float],
                      sample_feature: typing.Dict[str, float],
                      ) -> bool:
  """
  Checks if sample kernel is smaller than original feed
  by comparing all numerical features.
  """
  return (sample_feature['comp']      + sample_feature['rational'] +
          sample_feature['mem']       + sample_feature['localmem'] +
          sample_feature['coalesced'] + sample_feature['atomic']
          < 
          input_feature['comp']      + input_feature['rational'] +
          input_feature['mem']       + input_feature['localmem'] +
          input_feature['coalesced'] + input_feature['atomic'])

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
        stdout, _ = extractor.kernel_features(contents)
        features = extractor.StrToDictFeatures(stdout)
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

    threshold = 0.1
    if min_proximity <= threshold:
      self.benchmarks[min_idx] = self.benchmarks[min_idx]._replace(times_achieved = 1 + self.benchmarks[min_idx].times_achieved)
      return True
    return False
