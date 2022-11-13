# coding=utf-8
# Copyright 2022 Foivos Tsimpourlas.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Feature Extraction module for Dominic Grewe features.
"""
import subprocess
import tempfile
import typing

from deeplearning.benchpress.util import environment
from deeplearning.benchpress.util import crypto

from deeplearning.benchpress.util import logging as l
from absl import flags

FLAGS = flags.FLAGS
GREWE = environment.GREWE

KEYS = [
  "comp",
  "rational",
  "mem",
  "localmem",
  "coalesced",
  "atomic",
  "F2:coalesced/mem",
  "F4:comp/mem",
]

class GreweFeatures(object):
  """
  Source code features as defined in paper
  "Portable Mapping of Data Parallel Programs to OpenCL for Heterogeneous Systems"
  by D.Grewe, Z.Wang and M.O'Boyle.
  """
  @classmethod
  def KEYS(cls: 'GreweFeatures') -> typing.List[str]:
    return KEYS
  
  def __init__(self):
    return

  @classmethod
  def ExtractFeatures(cls,
                      src: str,
                      header_file     : str = None,
                      use_aux_headers : bool = True,
                      extra_args      : typing.List[str] = [],
                      ) -> typing.Dict[str, float]:
    """
    Invokes clgen_features extractor on source code and return feature mappings
    in dictionary format.

    If the code has syntax errors, features will not be obtained and empty dict
    is returned.
    """
    str_features = cls.ExtractRawFeatures(src, header_file = header_file, use_aux_headers = use_aux_headers, extra_args = extra_args)
    return cls.RawToDictFeats(str_features)

  @classmethod
  def ExtractIRFeatures(cls, bytecode: str) -> typing.Dict[str, float]:
    """
    Bytecode input in text-level feature space makes no sense. Therefore this function is just a decoy.
    """
    return {}

  @classmethod
  def ExtractRawFeatures(cls,
                         src: str,
                         header_file     : str = None,
                         use_aux_headers : bool = True,
                         extra_args      : typing.List[str] = [],
                         ) -> str:
    """
    Invokes clgen_features extractor on a single kernel.

    Params:
      src: (str) Kernel in string format.
    Returns:
      Feature vector and diagnostics in str format.
    """
    try:
      tdir = FLAGS.local_filesystem
    except Exception:
      tdir = None
    with tempfile.NamedTemporaryFile('w', prefix = "feat_ext_", suffix = '.cl', dir = tdir) as f:
      f.write(src)
      f.flush()

      arguments = []
      if header_file:
        htf = tempfile.NamedTemporaryFile('w', prefix = "feat_ext_head_", suffix = '.h', dir = tdir)
        htf.write(header_file)
        htf.flush()
        arguments = ["-extra-arg=-include{}".format(htf.name)]
        if extra_args:
          for arg in extra_args:
            arguments.append("--extra-arg={}".format(arg)) 

      cmd = [str(GREWE)] + arguments + [f.name]

      process = subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        universal_newlines = True,
      )
      stdout, stderr = process.communicate()
    return stdout

  @classmethod
  def ExtractIRRawFeatures(cls, bytecode: str) -> str:
    """
    Bytecode input in text-level feature space makes no sense. Therefore this function is just a decoy.
    """
    return ""

  @classmethod
  def RawToDictFeats(cls, str_feats: str) -> typing.Dict[str, float]:
    """
    Converts clgen_features subprocess output from raw string
    to a mapped dictionary of feature -> value.
    """
    try:
      lines  = str_feats.split('\n')
      # header, cumvs = lines[0].split(',')[2:], lines[-2].split(',')[2:]
      header, values = lines[0].split(',')[2:], [l for l in lines[1:] if l != '' and l != '\n']
      cumvs  = [0] * 8
      try:
        for vv in values:
          for idx, el in enumerate(vv.split(',')[2:]):
            cumvs[idx] = float(el)
        if len(header) != len(cumvs):
          raise ValueError("Bad alignment of header-value list of features. This should never happen.")
        return {key: float(value) for key, value in zip(header, cumvs)}
      except ValueError as e:
        raise ValueError("{}, {}".format(str(e), str_feats))
    except Exception as e:
      return {}