"""
Feature Extraction module for Autophase paper features.
"""
import subprocess
import tempfile
import typing

from deeplearning.clgen.preprocessors import opencl
from deeplearning.clgen.util import environment

from eupy.hermes import client

AUTOPHASE = "-load {} -autophase".format(environment.AUTOPHASE)

class AutophaseFeatures(object):
  """
  TODO write description.
  """
  def __init__(self):
    return

  @classmethod
  def ExtractFeatures(cls, src: str) -> typing.Dict[str, float]:
    return cls.RawToDictFeats(cls.ExtractRawFeatures(src))

  @classmethod
  def ExtractRawFeatures(cls, src: str) -> str:
    return opencl.CompileOptimizer(src, AUTOPHASE)

  @classmethod
  def RawToDictFeats(cls, str_feats: str) -> typing.Dict[str, float]:
    return {feat.split(' : ')[0]: int(feat.splt(' : ')[1]) for feat in str_feats.split('\n') if ' : ' in feat}
