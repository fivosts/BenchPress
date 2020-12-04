"""
Feature extraction tools for active learning.
"""
import subprocess
import tempfile
import typing

from deeplearning.clgen.util import environment
from deeplearning.clgen.util import crypto

CLGEN_FEATURES = environment.CLGEN_FEATURES

def StrToDictFeatures(str_features: str) -> typing.Dict[str, float]:
  """
  Converts clgen_features subprocess output from raw string
  to a mapped dictionary of feature -> value.
  """
  try:
    lines  = str_features.split('\n')
    header, values = lines[0].split(',')[2:], lines[-2].split(',')[2:]
    if len(header) != len(values):
      raise ValueError("Bad alignment of header-value list of features. This should never happen.")
    try:
      return {key: float(value) for key, value in zip(header, values)}
    except ValueError as e:
      raise ValueError("{}, {}".format(str(e), str_features))
  except Exception as e:
    # Kernel has a syntax error and feature line is empty.
    # Return an empty dict.
    return {}

def StrKernelFeatures(src: str, *extra_args) -> str:
  """
  Invokes clgen_features extractor on a single kernel.

  Params:
    src: (str) Kernel in string format.
    extra_args: Extra compiler arguments passed to feature extractor.
  Returns:
    Feature vector and diagnostics in str format.
  """
  file_hash = crypto.sha256_str(src)
  with tempfile.NamedTemporaryFile(
          'w', prefix = "feat_ext_{}_".format(file_hash), suffix = '.cl'
        ) as f:
    f.write(src)
    f.flush()
    cmd = [str(CLGEN_FEATURES), f.name]

    process = subprocess.Popen(
      cmd,
      stdout = subprocess.PIPE,
      stderr = subprocess.PIPE,
      universal_newlines = True,
    )
    stdout, stderr = process.communicate()
  return stdout, stderr

def DictKernelFeatures(src: str, *extra_args) -> typing.Dict[str, float]:
  """
  Invokes clgen_features extractor on source code and return feature mappings
  in dictionary format.

  If the code has syntax errors, features will not be obtained and empty dict
  is returned.
  """
  str_features, stderr = StrKernelFeatures(src, extra_args)
  return StrToDictFeatures(str_features)
