"""
Feature Extraction module for LLVM InstCount pass.
"""
import subprocess
import tempfile
import typing

from deeplearning.clgen.util import environment
from deeplearning.clgen.util import crypto

from eupy.hermes import client

class InstCountFeatures(object):
  """
  70-dimensional LLVM-IR feature vector,
  describing Total Funcs, Total Basic Blocks, Total Instructions
  and count of all different LLVM-IR instruction types.
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
