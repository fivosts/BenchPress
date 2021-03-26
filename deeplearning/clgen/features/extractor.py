"""
Feature extraction tools for active learning.
"""
import subprocess
import tempfile
import typing

from deeplearning.clgen.util import environment
from deeplearning.clgen.util import crypto

from eupy.hermes import client

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

def DBStrToDictFeatures(feat: str) -> typing.Dict[str, float]:
  """
  Convert string formatted features to dictionary.
  String is in the same format with DB entry.
  '
  comp:2.0
  mem:2.0
  ...
  '
  """
  features = {}
  try:
    for line in feat.split('\n'):
      delim = line.split(':')
      features[''.join(delim[0:-1])] = float(delim[-1])
  except ValueError:
    raise ValueError("{}".format(feat.split('\n')))
  return features

def DictKernelFeatures(src: str, *extra_args) -> typing.Dict[str, float]:
  """
  Invokes clgen_features extractor on source code and return feature mappings
  in dictionary format.

  If the code has syntax errors, features will not be obtained and empty dict
  is returned.
  """
  try:
    str_features, stderr = StrKernelFeatures(src, extra_args)
    return StrToDictFeatures(str_features)
  except OSError:
    import os
    import psutil
    mail = client.getClient()
    mail.send_message("extractor", src)
    main_process = psutil.Process(os.getpid())
    total_mem = (main_process.memory_info().rss +
                  sum([p.memory_info().rss 
                    for p in main_process.children(recursive = True)]
                  )
                )
    mail.send_message("extractor", str(total_mem))
    return {}
