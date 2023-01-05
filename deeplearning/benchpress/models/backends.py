# coding=utf-8
# Copyright 2022 Foivos Tsimpourlas and Chris Cummins.
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
"""Neural network backends for language models."""
import typing
import numpy as np

from deeplearning.benchpress.corpuses import tokenizers
from deeplearning.benchpress.corpuses import corpuses
from deeplearning.benchpress.proto import model_pb2
from deeplearning.benchpress.util import cache
from deeplearning.benchpress.util import pytorch

torch = pytorch.torch

class BackendBase(object):
  """The base class for a language model backend.

  A language model backend encapsulates all of the neural network logic.
  """

  def __init__(
    self,
    config: model_pb2.Model,
    fs_cache: cache.FSCache,
    hash: str,
    tokenizer: tokenizers.TokenizerBase = None,
    **kwargs,
  ):
    self.config = config
    self.cache = fs_cache
    self.hash = hash
    self.tokenizer = tokenizer

  ## Legacy function to support lazy creation of corpus
  def Create(self, tokenizer: tokenizers.TokenizerBase) -> None:
    self.tokenizer = tokenizer

  def PreTrain(self, corpus: corpuses.Corpus, **extra_kwargs) -> None:
    """Pre-train the backend"""
    raise NotImplementedError("pre-training is only supported in PyTorch BERT.")

  def Train(self, corpus: corpuses.Corpus, **extra_kwargs) -> None:
    """Train the backend."""
    raise NotImplementedError("Abstract Class.")

  def TrainBatch(self, batch) -> None:
    """Incrementally train language model on a batch of data."""
    raise NotImplementedError("Abstract Class.")

  def InitSampling(
    self, sampler: 'samplers.Sampler', seed: typing.Optional[int] = None
  ) -> None:
    """Initialize backend for sampling."""
    raise NotImplementedError("Abstract Class.")

  def InitSampleBatch(self, sampler: 'samplers.Sampler') -> None:
    """Begin a new sampling batch. Only called after InitSampling()."""
    raise NotImplementedError("Abstract Class.")

  def SampleNextIndices(
    self, sampler: 'samplers.Sampler', done: np.ndarray, tokenizer = None
  ) -> np.ndarray:
    """Sample the next indices for the current sample batch.

    Returns:
      A numpy array of int32 values with shape (batch_size,).
    """
    raise NotImplementedError("Abstract Class.")

  def SampleBatch(self, batch) -> np.ndarray:
    """Specifically sample a requested batch of data."""
    raise NotImplementedError("Abstract Class.")

  def EncodeInputs(self, src: typing.List[str]) -> np.array:
    """Encode text inputs to numpy arrays."""
    raise NotImplementedError("Abstract Class.")
  
  def ExtractHidden(self, encoded: typing.List[np.array]) -> np.array:
    """Extract Hidden State from Language Model"""
    raise NotImplementedError("Abstract Class")

  def GetEncoderModule(self, **kwargs) -> torch.nn.Module:
    """Return the internal torch module of an architecture."""
    raise NotImplementedError("Abstract class")

  def GetDecoderModule(self, **kwargs) -> torch.nn.Module:
    """Return a decoder version of LM's decoder."""
    raise NotImplementedError("Abstract class")
