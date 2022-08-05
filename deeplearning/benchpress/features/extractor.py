"""
Feature extraction tools for active learning.
"""
import typing

from deeplearning.benchpress.features import grewe
from deeplearning.benchpress.features import instcount
from deeplearning.benchpress.features import autophase
from deeplearning.benchpress.util import crypto

from eupy.hermes import client

extractors = {
  'GreweFeatures'     : grewe.GreweFeatures,
  'InstCountFeatures' : instcount.InstCountFeatures,
  'AutophaseFeatures' : autophase.AutophaseFeatures
}

def ExtractFeatures(src: str,
                    ext: typing.List[str] = None,
                    header_file     : str = None,
                    use_aux_headers : bool = True,
                    extra_args      : typing.List[str] = []
                    ) -> typing.Dict[str, typing.Dict[str, float]]:
  """
  Wrapper method for core feature functions.
  Returns a mapping between extractor type(string format) and feature data collected.
  """
  if not ext:
    ext = list(extractors.keys())
  return {xt: extractors[xt].ExtractFeatures(src, header_file = header_file, use_aux_headers = use_aux_headers, extra_args = extra_args) for xt in ext}

def ExtractIRFeatures(bytecode: str,
                      ext: typing.List[str] = None,
                      ) -> typing.Dict[str, typing.Dict[str, float]]:
  """
  Wrapper method for core feature functions.
  Returns a mapping between extractor type(string format) and feature data collected.

  Works for LLVM-IR as an input.
  """
  if not ext:
    ext = list(extractors.keys())
  return {xt: extractors[xt].ExtractIRFeatures(bytecode) for xt in ext}

def ExtractRawFeatures(src: str,
                       ext: typing.List[str] = None,
                       header_file     : str = None,
                       use_aux_headers : bool = True,
                       extra_args      : typing.List[str] = []
                       ) -> str:
  """
  Wrapper method for core feature functions.
  Returns a mapping between extractor type(string format) and feature data collected.
  """
  if not ext:
    ext = list(extractors.keys())
  if ext and not isinstance(ext, list):
    raise TypeError("Requested feature space extractors must be a list, {} received".format(type(ext)))
  return '\n'.join(["{}:\n{}".format(xt, extractors[xt].ExtractRawFeatures(src, header_file = header_file, use_aux_headers = use_aux_headers, extra_args = extra_args)) for xt in ext])

def ExtractIRRawFeatures(bytecode: str,
                         ext: typing.List[str] = None,
                         ) -> str:
  """
  Wrapper method for core feature functions.
  Returns a mapping between extractor type(string format) and feature data collected.

  Works for LLVM-IR as an input.
  """
  if not ext:
    ext = list(extractors.keys())
  if ext and not isinstance(ext, list):
    raise TypeError("Requested feature space extractors must be a list, {} received".format(type(ext)))
  return '\n'.join(["{}:\n{}".format(xt, extractors[xt].ExtractIRRawFeatures(bytecode)) for xt in ext])

def RawToDictFeats(str_feats: str) -> typing.Dict[str, typing.Dict[str, float]]:
  """
  Wrapper method for core feature functions.
  Returns a mapping between extractor type(string format) and feature data collected.
  """
  feats = {b.split(":\n")[0]: ''.join(b.split(':\n')[1:]) for b in str_feats.split('\n\n') if b.split(':\n')[1:]}
  return {xt: extractors[xt].RawToDictFeats(feat) for xt, feat in feats.items()}
