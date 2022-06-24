"""
Memory replay buffer for reinforcement learning training.
"""
import pathlib
import typing
import pickle

from deeplearning.clgen.reinforcement_learning import interactions
from deeplearning.clgen.corpuses import tokenizers
from deeplearning.clgen.models import language_models
from deeplearning.clgen.proto import reinforcement_learning_pb2
from deeplearning.clgen.util import environment
from deeplearning.clgen.util import distrib
from deeplearning.clgen.util import pytorch

torch = pytorch.torch

def from_config(config: reinforcement_learning_pb2.RLModel,
                feature_tokenizer: tokenizers.FeatureTokenizer,
                language_model: language_models.Model
                ) -> "FeatureLoader":
  """
  Return the right torch dataloader based on configuration.
  """
  if config.HasField("train_set"):
    return CorpusFeatureLoader(language_model.corpus, feature_tokenizer)
  return

class FeatureLoader(torch.utils.data.Dataset):
  """
  Torch-based dataloading class for target feature vectors.
  """
  def __init__(self, config: reinforcement_learning_pb2.RLModel):
    self.config = config
    return
