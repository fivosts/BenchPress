"""
Feature Extraction module for Autophase paper features.
"""
import subprocess
import tempfile
import typing

from deeplearning.clgen.util import environment
from deeplearning.clgen.util import crypto

from eupy.hermes import client

class AutophaseFeatures(object):
  """
  TODO write description.
  """
  def __init__(self):
    return

  @classmethod
  def ExtractFeatures(cls, src: str) -> typing.Dict[str, float]:
    raise NotImplementedError

  @classmethod
  def ExtractRawFeatures(cls, src: str) -> str:
    raise NotImplementedError

  @classmethod
  def RawToDictFeats(cls, str_feats: str) -> typing.Dict[str, float]:
    raise NotImplementedError
