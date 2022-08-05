"""
Helper module the provides range of worker functions for experiments.
"""
import typing
import pathlib

from deeplearning.benchpress.features import extractor
from deeplearning.benchpress.features import feature_sampler
from deeplearning.benchpress.preprocessors import opencl
from deeplearning.benchpress.util import environment
from deeplearning.benchpress.util import logging as l

def ContentHash(db_feat: typing.Tuple[str, str]) -> typing.Tuple[str, typing.Dict[str, float]]:
  """
  Multiprocessing Worker calculates contentfile hash
  of file and returns it.
  """
  if len(db_feat) == 2:
    src, feats = db_feat
    include = None
  else:
    src, include, feat = db_feat
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
  if len(db_feat) == 2:
    _, feats = db_feat
  else:
    _, _, feats = db_feat
  try:
    return extractor.RawToDictFeats(feats)
  except Exception as e:
    l.logger().warn(e)
    return None

def ExtractAndCalculate(src_incl        : typing.Tuple[str, str],
                        target_features : typing.Dict[str, float],
                        feature_space   : str
                        ) -> typing.Tuple[str, str, float]:
  """
  Extract features for source code and calculate distance from target.

  Returns:
    Tuple of source code with distance.
  """
  src, incl = src_incl
  f = extractor.ExtractFeatures(src, [feature_space], header_file = incl, extra_args = ["-include{}".format(pathlib.Path(environment.CLSMITH_INCLUDE) / "CLSmith.h")] if incl else [""])
  if feature_space in f and f[feature_space]:
    return src, incl, feature_sampler.calculate_distance(f[feature_space], target_features, feature_space)
  return None

def IRExtractAndCalculate(bytecode      : str,
                        target_features : typing.Dict[str, float],
                        feature_space   : str
                        ) -> typing.Tuple[str, str, float]:
  """
  Extract features for source code and calculate distance from target.

  Returns:
    Tuple of source code with distance.
  """
  f = extractor.ExtractIRFeatures(bytecode, [feature_space])
  if feature_space in f and f[feature_space]:
    return bytecode, "", feature_sampler.calculate_distance(f[feature_space], target_features, feature_space)
  return None

def FeatureExtractor(src_incl: typing.Tuple[str, str]) -> typing.Tuple[str, str, str]:
  """
  Extracts Raw features for all feat spaces and returns tuple of source and features.
  """
  src, incl = src_incl
  try:
    return src, incl, extractor.ExtractRawFeatures(src, header_file = incl, extra_args = ["-include{}".format(pathlib.Path(environment.CLSMITH_INCLUDE) / "CLSmith.h")] if incl else [""])
  except ValueError:
    return src, incl, ""

def IRFeatureExtractor(bytecode: str) -> typing.Tuple[str, str, str]:
  """
  Extracts Raw features for all feat spaces and returns tuple of source and features.
  """
  try:
    return bytecode, "", extractor.ExtractIRRawFeatures(bytecode)
  except ValueError:
    return bytecode, "", ""

def SortedDistances(data: typing.List[typing.Tuple[str, str, typing.Dict[str, float]]],
                    target_features: typing.Dict[str, float],
                    feature_space: str
                    ) -> typing.List[float]:
  """
  Return list of euclidean distances from target features in ascending order.
  """
  return sorted([feature_sampler.calculate_distance(dp, target_features, feature_space) for _, _, dp in data])

def SortedSrcDistances(data: typing.List[typing.Tuple[str, typing.Dict[str, float]]],
                       target_features: typing.Dict[str, float],
                       feature_space: str
                       ) -> typing.List[typing.Tuple[str, str, float]]:
  """
  Return list of pairs of euclidean distances from target features with source code in ascending order.
  """
  return sorted([(src, include, feature_sampler.calculate_distance(dp, target_features, feature_space)) for src, include, dp in data], key = lambda x: x[2])

def SortedSrcFeatsDistances(data: typing.List[typing.Tuple[str, typing.Dict[str, float]]],
                            target_features: typing.Dict[str, float],
                            feature_space: str
                            ) -> typing.List[typing.Tuple[str, str, typing.Dict[str, float], float]]:
  """
  Return list of pairs of euclidean distances from target features with source code and features in ascending order.
  """
  return sorted([(src, include, dp, feature_sampler.calculate_distance(dp, target_features, feature_space)) for src, include, dp in data], key = lambda x: x[3])
