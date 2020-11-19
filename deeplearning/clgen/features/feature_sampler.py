"""
Feature space sampling of source code.
"""
import typing

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
    self.path = pathlib.Path("/home/fivosts/rodinia_benchmarks")
    self.benchmarks = []
    for f in self.path.iterdir():
      with open(f, 'r') as file:
        contents = file.readlines()
        self.benchmarks.append(
          Benchmark(
            f,
            f.name,
            contents,
            extractor.StrToDictFeatures(extractor.kernel_features(contents)),
            0
          )
        )

  def 