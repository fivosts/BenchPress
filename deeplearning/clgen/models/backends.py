"""Neural network backends for language models."""
import typing
import numpy as np

from deeplearning.clgen.corpuses import tokenizers
from deeplearning.clgen.corpuses import corpuses
from deeplearning.clgen.samplers import samplers
from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen.util import cache
from deeplearning.clgen.util import pytorch

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
    self, sampler: samplers.Sampler, seed: typing.Optional[int] = None
  ) -> None:
    """Initialize backend for sampling."""
    raise NotImplementedError("Abstract Class.")

  def InitSampleBatch(self, sampler: samplers.Sampler) -> None:
    """Begin a new sampling batch. Only called after InitSampling()."""
    raise NotImplementedError("Abstract Class.")

  def SampleNextIndices(
    self, sampler: samplers.Sampler, done: np.ndarray, tokenizer = None
  ) -> np.ndarray:
    """Sample the next indices for the current sample batch.

    Returns:
      A numpy array of int32 values with shape (batch_size,).
    """
    raise NotImplementedError("Abstract Class.")

  def SampleBatch(self, batch) -> np.ndarray:
    """Specifically sample a requested batch of data."""
    raise NotImplementedError("Abstract Class.")

  def GetSamplingModule(self, temperature: int, **kwargs) -> torch.nn.Module:
    """Return the internal torch module of an architecture."""
    raise NotImplementedError("Abstract class")
