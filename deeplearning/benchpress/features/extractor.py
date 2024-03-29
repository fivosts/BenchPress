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
Feature extraction tools for active learning.
"""
import typing

from deeplearning.benchpress.features import grewe
from deeplearning.benchpress.features import instcount
from deeplearning.benchpress.features import autophase
from deeplearning.benchpress.features import hidden_state
from deeplearning.benchpress.util import crypto

from eupy.hermes import client

extractors = {
  'GreweFeatures'     : grewe.GreweFeatures,
  'InstCountFeatures' : instcount.InstCountFeatures,
  'AutophaseFeatures' : autophase.AutophaseFeatures,
  'HiddenState'       : hidden_state.HiddenStateFeatures,
}

def ExtractFeatures(src: str,
                    ext: typing.List[str] = None,
                    header_file     : str = None,
                    use_aux_headers : bool = True,
                    extra_args      : typing.List[str] = [],
                     **kwargs,
                    ) -> typing.Dict[str, typing.Dict[str, float]]:
  """
  Wrapper method for core feature functions.
  Returns a mapping between extractor type(string format) and feature data collected.
  """
  if not ext:
    ext = [k for k in extractors.keys() if k != 'HiddenState']
  return {xt: extractors[xt].ExtractFeatures(src, header_file = header_file, use_aux_headers = use_aux_headers, extra_args = extra_args, **kwargs) for xt in ext}

def ExtractFeaturesIter(srcs: typing.List[str],
                        ext: typing.List[str] = None,
                        header_file     : str = None,
                        use_aux_headers : bool = True,
                        extra_args      : typing.List[str] = [],
                       **kwargs,
                        ) -> typing.Dict[str, typing.Iterator[typing.Dict[str, float]]]:
  """
  Wrapper method for core feature functions.
  Returns a mapping between extractor type(string format) and feature data collected.
  """
  if not ext:
    ext = [k for k in extractors.keys() if k != 'HiddenState']
  return {xt: extractors[xt].ExtractFeaturesIter(srcs, header_file = header_file, use_aux_headers = use_aux_headers, extra_args = extra_args, **kwargs) for xt in ext}

def ExtractIRFeatures(bytecode: str,
                      ext: typing.List[str] = None,
                      ) -> typing.Dict[str, typing.Dict[str, float]]:
  """
  Wrapper method for core feature functions.
  Returns a mapping between extractor type(string format) and feature data collected.

  Works for LLVM-IR as an input.
  """
  if not ext:
    ext = [k for k in extractors.keys() if k != 'HiddenState']
  return {xt: extractors[xt].ExtractIRFeatures(bytecode, **kwargs) for xt in ext}

def ExtractIRFeaturesIter(bytecodes: typing.List[str],
                          ext: typing.List[str] = None,
                          **kwargs,
                          ) -> typing.Dict[str, typing.Iterator[typing.Dict[str, float]]]:
  """
  Wrapper method for core feature functions.
  Returns a mapping between extractor type(string format) and feature data collected.

  Works for LLVM-IR as an input.
  """
  if not ext:
    ext = [k for k in extractors.keys() if k != 'HiddenState']
  return {xt: extractors[xt].ExtractIRFeaturesIter(bytecodes, **kwargs) for xt in ext}

def ExtractRawFeatures(src: str,
                       ext: typing.List[str] = None,
                       header_file     : str = None,
                       use_aux_headers : bool = True,
                       extra_args      : typing.List[str] = [],
                       **kwargs,
                       ) -> str:
  """
  Wrapper method for core feature functions.
  Returns a mapping between extractor type(string format) and feature data collected.
  """
  if not ext:
    ext = [k for k in extractors.keys() if k != 'HiddenState']
  if ext and not isinstance(ext, list):
    raise TypeError("Requested feature space extractors must be a list, {} received".format(type(ext)))
  return '\n'.join(["{}:\n{}".format(xt, extractors[xt].ExtractRawFeatures(src, header_file = header_file, use_aux_headers = use_aux_headers, extra_args = extra_args, **kwargs)) for xt in ext])

def ExtractRawFeaturesIter(srcs: typing.List[str],
                           ext: typing.List[str] = None,
                           header_file     : str = None,
                           use_aux_headers : bool = True,
                           extra_args      : typing.List[str] = [],
                           **kwargs,
                           ) -> typing.Iterator[str]:
  """
  Wrapper method for core feature functions.
  Returns a mapping between extractor type(string format) and feature data collected.
  """
  if not ext:
    ext = [k for k in extractors.keys() if k != 'HiddenState']
  if ext and not isinstance(ext, list):
    raise TypeError("Requested feature space extractors must be a list, {} received".format(type(ext)))
  return '\n'.join(["{}:\n{}".format(xt, extractors[xt].ExtractRawFeaturesIter(srcs, header_file = header_file, use_aux_headers = use_aux_headers, extra_args = extra_args, **kwargs)) for xt in ext])

def ExtractIRRawFeatures(bytecode: str,
                         ext: typing.List[str] = None,
                         **kwargs,
                         ) -> str:
  """
  Wrapper method for core feature functions.
  Returns a mapping between extractor type(string format) and feature data collected.

  Works for LLVM-IR as an input.
  """
  if not ext:
    ext = [k for k in extractors.keys() if k != 'HiddenState']
  if ext and not isinstance(ext, list):
    raise TypeError("Requested feature space extractors must be a list, {} received".format(type(ext)))
  return '\n'.join(["{}:\n{}".format(xt, extractors[xt].ExtractIRRawFeatures(bytecode, **kwargs)) for xt in ext])

def ExtractIRRawFeaturesIter(bytecodes: typing.List[str],
                             ext: typing.List[str] = None,
                             **kwargs,
                             ) -> typing.Iterator[str]:
  """
  Wrapper method for core feature functions.
  Returns a mapping between extractor type(string format) and feature data collected.

  Works for LLVM-IR as an input.
  """
  if not ext:
    ext = [k for k in extractors.keys() if k != 'HiddenState']
  if ext and not isinstance(ext, list):
    raise TypeError("Requested feature space extractors must be a list, {} received".format(type(ext)))
  return '\n'.join(["{}:\n{}".format(xt, extractors[xt].ExtractIRRawFeaturesIter(bytecodes, **kwargs)) for xt in ext])

def RawToDictFeats(str_feats: typing.Union[str, typing.List[float]],
                   ext: typing.List[str] = None,
                   **kwargs,
                   ) -> typing.Dict[str, typing.Dict[str, float]]:
  """
  Wrapper method for core feature functions.
  Returns a mapping between extractor type(string format) and feature data collected.
  """
  if not ext:
    ext = [k for k in extractors.keys() if k != 'HiddenState']
  if ext and not isinstance(ext, list):
    raise TypeError("Requested feature space extractors must be a list, {} received".format(type(ext)))
  if ext != ['HiddenState']:
    feats = {b.split(":\n")[0]: ''.join(b.split(':\n')[1:]) for b in str_feats.split('\n\n') if b.split(':\n')[1:]}
  else:
    feats = {ext[0]: str_feats}
  return {xt: extractors[xt].RawToDictFeats(feat, **kwargs) for xt, feat in feats.items()}
