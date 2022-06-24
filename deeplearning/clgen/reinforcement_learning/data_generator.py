"""
Memory replay buffer for reinforcement learning training.
"""
import pathlib
import typing
import pickle

from deeplearning.clgen.reinforcement_learning import interactions
from deeplearning.clgen.proto import reinforcement_learning_pb2
from deeplearning.clgen.util import environment
from deeplearning.clgen.util import distrib
from deeplearning.clgen.util import pytorch

torch = pytorch.torch

class FeatureLoader(torch.utils.data.Dataset):
  """
  Torch-based dataloading class for target feature vectors.
  """
  def __init__(self, config: reinforcement_learning_pb2.RLModel):
    self.config = config
    return