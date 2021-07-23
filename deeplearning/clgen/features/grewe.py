"""
Feature Extraction module for Dominic Grewe features.
"""
import subprocess
import tempfile
import typing

from deeplearning.clgen.util import environment
from deeplearning.clgen.util import crypto

from eupy.native import logger as l

GREWE = environment.GREWE

class GreweFeatures(object):
  """
  Source code features as defined in paper
  "Portable Mapping of Data Parallel Programs to OpenCL for Heterogeneous Systems"
  by D.Grewe, Z.Wang and M.O'Boyle.
  """
  def __init__(self):
    return

  @classmethod
  def ExtractFeatures(cls, src: str, use_aux_headers: bool = True) -> typing.Dict[str, float]:
    """
    Invokes clgen_features extractor on source code and return feature mappings
    in dictionary format.

    If the code has syntax errors, features will not be obtained and empty dict
    is returned.
    """
    str_features = cls.ExtractRawFeatures(src, use_aux_headers)
    return cls.RawToDictFeats(str_features)

  @classmethod
  def ExtractRawFeatures(cls, src: str, use_aux_headers: bool = True) -> str:
    """
    Invokes clgen_features extractor on a single kernel.

    Params:
      src: (str) Kernel in string format.
    Returns:
      Feature vector and diagnostics in str format.
    """
    file_hash = crypto.sha256_str(src)
    with tempfile.NamedTemporaryFile(
            'w', prefix = "feat_ext_{}_".format(file_hash), suffix = '.cl'
          ) as f:
      f.write(src)
      f.flush()
      cmd = [str(GREWE), f.name]

      process = subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        universal_newlines = True,
      )
      stdout, stderr = process.communicate()
    return stdout

  @classmethod
  def RawToDictFeats(cls, str_feats: str) -> typing.Dict[str, float]:
    """
    Converts clgen_features subprocess output from raw string
    to a mapped dictionary of feature -> value.
    """
    try:
      lines  = str_feats.split('\n')
      header, values = lines[0].split(',')[2:], [l for l in lines[1:] if l != '' and l != '\n']
      cumvs  = [0] * 8
      try:
        for vv in values:
          for idx, el in enumerate(vv.split(',')[2:]):
            cumvs[idx] += float(el)
        if len(header) != len(cumvs):
          raise ValueError("Bad alignment of header-value list of features. This should never happen.")
        return {key: float(value) for key, value in zip(header, cumvs)}
      except ValueError as e:
        raise ValueError("{}, {}".format(str(e), str_feats))
    except Exception as e:
      print(e)
      # l.getLogger().warn("Grewe RawDict: {}".format(e))
      # Kernel has a syntax error and feature line is empty.
      # Return an empty dict.
      return {}