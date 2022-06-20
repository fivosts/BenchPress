"""
Modeling for reinforcement learning program synthesis.
"""
import pathlib
import typing
from deeplearning.clgen.reinforcement_learning.interactions import Action

from deeplearning.clgen.util import environment
from deeplearning.clgen.util import distrib
from deeplearning.clgen.util import pytorch

torch = pytorch.torch

class ActionTypeQV(torch.nn.Module):
  """Deep Q-Values for Action type prediction."""
  def __init__(self):
    super(ActionTypeQV, self).__init__()
    return
  
  def forward(self, input_ids: typing.Dict[str, torch.Tensor]) -> typing.Dict[str, torch.Tensor]:
    """Action type forward function."""
    raise NotImplementedError
    return

class QValuesModel(object):
  """
  Handler of Deep-QNMs for program synthesis.
  """
  def __init__(self, cache_path: pathlib.Path) -> None:
    self.cache_path = cache_path / "DQ_model"
    if environment.WORLD_RANK == 0:
      self.cache_path.mkdir(exists_ok = True, parents = True)
    return
  
  def saveCheckpoint(self) -> None:
    """Checkpoint Deep Q-Nets."""
    raise NotImplementedError
    return

  def loadCheckpoint(self) -> None:
    """Load Deep Q-Nets."""
    raise NotImplementedError
    return
