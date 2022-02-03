"""
Helper module the provides range of worker functions for experiments.
"""
import typing
from deeplearning.clgen.features import extractor
from deeplearning.clgen.features import feature_sampler

def ContentHash(db_feat: typing.Tuple[str, str]) -> typing.Tuple[str, typing.Dict[str, float]]:
  """
  Multiprocessing Worker calculates contentfile hash
  of file and returns it.
  """
  src, feats = db_feat
  try:
    return opencl.ContentHash(src), extractor.RawToDictFeats(feats)
  except Exception as e:
    l.logger().warn(e)
    return None

def ContentFeat(db_feat: typing.Tuple[str, str]) -> typing.Dict[str, float]:
  """
  Multiprocessing Worker calculates contentfile hash
  of file and returns it.
  """
  _, feats = db_feat
  try:
    return extractor.RawToDictFeats(feats)
  except Exception as e:
    l.logger().warn(e)
    return None

def ExtractAndCalculate(src             : str,
                        target_features : typing.Dict[str, float],
                        feature_space   : str
                        ) -> typing.Dict[str, float]:
  """
  Extract features for source code and calculate distance from target.

  Returns:
    Tuple of source code with distance.
  """
  f = extractor.ExtractFeatures(src, [feature_space])
  if feature_space in f and f[feature_space]:
    return src, feature_sampler.calculate_distance(f[feature_space], target_features, feature_space)
  return None

def FeatureExtractor(src: str) -> typing.Tuple[str, str]:
  """
  Extracts Raw features for all feat spaces and returns tuple of source and features.
  """
  try:
    return src, extractor.ExtractRawFeatures(src)
  except ValueError:
    return src, ""

def SortedDistances(data: typing.List[typing.Tuple[str, typing.Dict[str, float]]],
                    target_features: typing.Dict[str, float],
                    feature_space: str
                    ) -> typing.List[float]:
  """
  Return list of pairs of source with respective euclidean distances from target features in ascending order.
  """
  return sorted([feature_sampler.calculate_distance(dp, target_features, feature_space) for _, dp in data])

def SortedSrcDistances(data: typing.List[typing.Tuple[str, typing.Dict[str, float]]],
                       target_features: typing.Dict[str, float],
                       feature_space: str
                       ) -> typing.List[typing.Tuple[str, float]]:
  """
  Return list of euclidean distances from target features in ascending order.
  """
  return sorted([(src, feature_sampler.calculate_distance(dp, target_features, feature_space)) for src, dp in data], key = lambda x: x[1])
