"""
Feature extraction tools for active learning.
"""
import subprocess
import tempfile
import typing

from deeplearning.clgen.features import grewe
from deeplearning.clgen.features import instcount
from deeplearning.clgen.features import autophase
from deeplearning.clgen.util import crypto

from eupy.hermes import client

extractors = {
  'GreweFeatures'     : grewe.GreweFeatures,
  'InstCountFeatures' : instcount.InstCountFeatures,
  'AutophaseFeatures' : autophase.AutophaseFeatures
}

def ExtractFeatures(src: str,
                    ext: typing.List[str] = None,
                    use_aux_headers: bool = True,
                    ) -> typing.Dict[str, typing.Dict[str, float]]:
  """
  Wrapper method for core feature functions.
  Returns a mapping between extractor type(string format) and feature data collected.
  """
  if not ext:
    ext = list(extractors.keys())
  return {xt: extractors[xt].ExtractFeatures(src, use_aux_headers = use_aux_headers) for xt in ext}

def ExtractRawFeatures(src: str,
                       ext: typing.List[str] = None,
                       use_aux_headers: bool = True,
                       ) -> str:
  """
  Wrapper method for core feature functions.
  Returns a mapping between extractor type(string format) and feature data collected.
  """
  if not ext:
    ext = list(extractors.keys())
  return '\n'.join(["{}:\n{}".format(xt, extractors[xt].ExtractRawFeatures(src, use_aux_headers = use_aux_headers)) for xt in ext])

def RawToDictFeats(str_feats: str) -> typing.Dict[str, typing.Dict[str, float]]:
  """
  Wrapper method for core feature functions.
  Returns a mapping between extractor type(string format) and feature data collected.
  """
  feats = {b.split(":\n")[0]: ''.join(b.split(':\n')[1:]) for b in str_feats.split('\n\n') if b.split(':\n')[1:]}
  return {xt: extractors[xt].RawToDictFeats(feat) for xt, feat in feats.items()}
