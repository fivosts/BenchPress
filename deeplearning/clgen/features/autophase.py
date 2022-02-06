"""
Feature Extraction module for Autophase paper features.
"""
import typing

from deeplearning.clgen.preprocessors import opencl
from deeplearning.clgen.util import environment

AUTOPHASE = ["-load", environment.AUTOPHASE, "-autophase"]

class AutophaseFeatures(object):
  """
  TODO write description.
  """
  def __init__(self):
    return

  @classmethod
  def ExtractFeatures(cls,
                      src: str,
                      header_file     : str = None,
                      use_aux_headers : bool = True,
                      extra_args      : typing.List[str] = [],
                      ) -> typing.Dict[str, float]:
    return cls.RawToDictFeats(cls.ExtractRawFeatures(src, header_file = header_file, use_aux_headers = use_aux_headers, extra_args = extra_args))

  @classmethod
  def ExtractIRFeatures(cls, bytecode: str) -> typing.Dict[str, float]:
    return cls.RawToDictFeats(cls.ExtractRawFeatures(src, header_file = header_file, use_aux_headers = use_aux_headers, extra_args = extra_args))

  @classmethod
  def ExtractRawFeatures(cls,
                         src: str,
                         header_file     : str = None,
                         use_aux_headers : bool = True,
                         extra_args      : typing.List[str] = [],
                         ) -> str:
    try:
      return opencl.CompileOptimizer(src, AUTOPHASE, header_file = header_file, use_aux_headers = use_aux_headers, extra_args = extra_args)
    except ValueError:
      return ""

  @classmethod
  def ExtractIRRawFeatures(cls, bytecode: str) -> str:
    try:
      return opencl.CompileOptimizerIR(src, AUTOPHASE)
    except ValueError:
      return ""

  @classmethod
  def RawToDictFeats(cls, str_feats: str) -> typing.Dict[str, float]:
    return {feat.split(' : ')[0]: int(feat.split(' : ')[1]) for feat in str_feats.split('\n') if ' : ' in feat}
