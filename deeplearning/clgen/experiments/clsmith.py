"""
Evaluation script for clsmith mutation program.
"""
import typing
import tempfile
import subprocess
import pathlib
import json
import os
import tqdm
import math

from absl import flags

from deeplearning.clgen.features import extractor
from deeplearning.clgen.util import plotter
from deeplearning.clgen.util import environment

FLAGS = flags.FLAGS

CLSMITH = environment.CLSMITH

def ExtractAndCalculate_worker(src             : str,
                               target_features : typing.Dict[str, float],
                               feature_space   : str
                               ) -> typing.Dict[str, float]:
  """
  Extract features for source code and calculate distance from target.

  Returns:
    Tuple of source code with distance.
  """
  f = extractor.ExtractFeatures(src, [feat_space])
  if feature_space in f and f[feature_space]:
    return src, feature_sampler.calculate_distance(f[feature_space], target_features, feature_space)
  return None

def GenerateCLSmith(**kwargs) -> None:
  """
  Compare mutec mutation tool on github's database against BenchPress.
  Comparison is similar to KAverageScore comparison.
  """
  clsmith_path = kwargs.get('clsmith_path', '')

  if not pathlib.Path(CLSMITH).exists():
    raise FileNotFoundError("CLSmith executable not found: {}".format(CLSMITH))

  # Initialize clsmith database
  clsmith_db = samples_database.SamplesDatabase(url = "sqlite:///{}".format(str(pathlib.Path(clsmith_path).resolve())), must_exist = False)

  ## Somewhere in here you should execute a subprocess command in a while loop.
  raise NotImplementedError

  return
