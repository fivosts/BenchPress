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

from deeplearning.benchpress.language_models import backends

from absl import flags

FLAGS = flags.FLAGS

KEYS = None
LANGUAGE_MODEL = None

def setup_lm(lm: backends.BackendBase) -> None:
  global LANGUAGE_MODEL
  global KEYS
  KEYS = ["f{}".format(x) for x in range(lm.hidden_state_size)]
  LANGUAGE_MODEL = lm
  return

class HiddenStateFeatures(object):
  """
  Source code features as defined in paper
  "Portable Mapping of Data Parallel Programs to OpenCL for Heterogeneous Systems"
  by D.Grewe, Z.Wang and M.O'Boyle.
  """
  def __init__(self):
    return

  @classmethod
  def ExtractFeatures(cls,
                      src: str,
                      header_file     : str  = None,
                      use_aux_headers : bool = True,
                      extra_args      : typing.List[str] = [],
                      ) -> typing.Dict[str, float]:
    """
    Invokes clgen_features extractor on source code and return feature mappings
    in dictionary format.

    If the code has syntax errors, features will not be obtained and empty dict
    is returned.
    """
    raw_features = cls.ExtractRawFeatures(src)
    return cls.RawToDictFeats(raw_features)

  @classmethod
  def ExtractIRFeatures(cls, bytecode: str) -> typing.Dict[str, float]:
    """
    Bytecode input in text-level feature space makes no sense. Therefore this function is just a decoy.
    """
    return {}

  @classmethod
  def ExtractRawFeatures(cls, src: str) -> typing.List[float]:
    """
    Invokes clgen_features extractor on a single kernel.

    Params:
      src: (str) Kernel in string format.
    Returns:
      Feature vector and diagnostics in str format.
    """
    encoded = LANGUAGE_MODEL.train.data_generator._padToMaxPosition(
      LANGUAGE_MODEL.train.data_generator._addStartEndToken(
        [int(x) for x in tokenized]
      )
    )[:LANGUAGE_MODEL.train.data_generator.sampler.sequence_length]
    input_ids = torch.LongTensor(encoded).unsqueeze(0).unsqueeze(0)
    workload = {
      'input_ids'         : input_ids,
      'input_mask'        : (input_ids != LANGUAGE_MODEL.tokenizer.padToken),
      'position_ids'      : torch.arange(len(encoded), dtype = torch.int64).unsqueeze(0).unsqueeze(0),
      'mask_labels'       : torch.full(tuple(input_ids.shape), -100, dtype = torch.int64),
      'masked_lm_lengths' : torch.full(tuple(1,1,1), -1, dtype = torch.int64)

    }
    return LANGUAGE_MODEL.sample_model_step(
      LANGUAGE_MODEL.sample.model,
      workload,
      iteration = 0,
      extract_hidden_states = True,
    )['hidden_state']

  @classmethod
  def ExtractIRRawFeatures(cls, bytecode: str) -> str:
    """
    Bytecode input in text-level feature space makes no sense. Therefore this function is just a decoy.
    """
    return ""

  @classmethod
  def RawToDictFeats(cls, raw_feats: typing.List[float]) -> typing.Dict[str, float]:
    """
    Converts clgen_features subprocess output from raw string
    to a mapped dictionary of feature -> value.
    """
    return {
      "f{}".format(k): v for k, v in zip(KEYS, raw_feats)
    }