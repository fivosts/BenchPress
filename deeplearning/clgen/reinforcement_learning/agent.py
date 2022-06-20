"""
Agents module for reinforcement learning.
"""
import pathlib
import typing
import numpy as np

from deeplearning.clgen.reinforcement_learning import interactions
from deeplearning.clgen.reinforcement_learning import model
from deeplearning.clgen.util import pytorch
from deeplearning.clgen.util import environment

torch = pytorch.torch

class Policy(object):
  """
  The policy selected over Q-Values
  """
  def __init__(self, cache_path: pathlib.Path):
    
    self.cache_path = cache_path / "agent"
    if environment.WORLD_RANK == 0:
      self.cache_path.mkdir(exists_ok = True, parents = True)
    return

  def predict_action_type(self, action_type_logits: torch.FloatTensor) -> int:
    """
    Get the Q-Values for action types and apply policy on it.
    """
    raise NotImplementedError
    return action_type

  def predict_action_index(self, action_index_logits: torch.FloatTensor) -> int:
    """
    Get the Q-Values for action index and apply policy on it.
    """
    raise NotImplementedError
    return action_index

  def select_token(self, token_logits: torch.FloatTensor) -> int:
    """
    Get logit predictions for token and apply policy on it.
    """
    raise NotImplementedError
    return token

class Agent(object):
  """
  Benchmark generation RL-Agent.
  """
  def __init__(self, cache_path: pathlib.Path):

    self.cache_path = cache_path
    if environment.WORLD_RANK == 0:
      self.cache_path.mkdir(exists_ok = True, parents = True)

    self.q_values = model.QValuesModel()
    self.policy   = Policy()

    self.loadCheckpoint()
    return

  def make_action(self, state: interactions.State) -> interactions.Action:
    """
    Agent collects the current state by the environment
    and picks the right action.
    """
    action_type_logits  = self.q_values.predict_action_type(state)
    action_type         = self.policy.select_action_type(action_type_logits)

    if action_type != interactions.ACTION_SPACE['COMP']:
      action_index_logits = self.q_values.predict_action_index(state, action_type)
      action_index        = self.policy.select_action_index(action_index_logits)
      if action_type == interactions.ACTION_SPACE['ADD']:
        token_logits        = self.q_values.predict_token(state)
        token               = self.policy.select_token(token_logits)
      else:
        token_logits      = None
        token             = None
    else:
      action_index        = None
      action_index_logits = None
      token_logits        = None
      token               = None

    return interactions.Action(
      action_type         = action_type,
      action_type_logits  = action_type_logits,
      action_index        = action_index,
      action_index_logits = action_index_logits,
      token_type          = token,
      token_type_logits   = token_logits,
    )

  def update_agent(self, input_ids) -> None:
    """
    Train the agent on the new episodes.
    """
    raise NotImplementedError
    return

  def saveCheckpoint(self) -> None:
    """
    Save agent state.
    """
    return
  
  def loadCheckpoint(self) -> None:
    """
    Load agent state.
    """
    return
