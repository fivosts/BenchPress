"""
RL Environment for the task of targeted benchmark generation.
"""
import pathlib
import numpy as np

from deeplearning.clgen.preprocessors import opencl
from deeplearning.clgen.reinforcement_learning import agent

from absl import flags

class Reward(typing.NamedTuple):
  action   : agent.Action
  value    : float
  distance : float
  comment  : str

class State(typing.NamedTuple):
  target_features : typing.Dict[str, float]
  code            : np.array

class Environment(object):
  """
  Environment representation for RL Agents.
  """
  def __init__(self) -> None:
    self.feature_space = feature_space
    return
  
  def compute_reward(self, action) -> Reward:
    """Compute an action's reward."""
    if action.action_type == agent.ACTION_TYPE_SPACE['ADD'] or action.action_type == agent.ACTION_TYPE_SPACE['REM']:
      return Reward(
        action   = action,
        value    = 0.0,
        distance = None,
        comment  = "[ADD] action, reward is +0."
      )
    elif action.action_type == agent.ACTION_TYPE_SPACE['COMP']:
      source = self.tokenizer.ArrayToCode(self.current_state.code)
      try:
        _ = opencl.Compile(source)
        feats = extractors.ExtractFeatures(source, [self.feature_space])
        compiles = True
      except ValueError:
        compiles = False
      if not compiles:
        return Reward(
          action   = action,
          value    = -1,
          distance = None,
          comment  = "[COMPILE] action failed and reward is -1.",
        )
      else:
        dist = feature_sample.euclidean_distance(feats, self.current_state.target_features)
        if dist == 0:
          return Reward(
            action   = action,
            value    = 1.0,
            distance = dist,
            comment  = "[COMPILE] led to dropping distance to 0, reward is +1!"
          )
        else:
          return Reward(
            action = action,
            value  = 1.0 / (dist),
            distance = dist,
            comment = "[COMPILE] succeeded, reward is {}, new distance is {}".format(1.0 / dist, dist)
          )
    else:
      raise ValueError("Action type {} does not exist.".format(action.action_type))

  def step(self, action) -> Reward:
    """
    Collect an action from an agent and compute its reward.
    """
    raise NotImplementedError
  
  def reset(self) -> None:
    """
    Reset the state of the environment.
    """
    raise NotImplementedError
  
  def get(self) -> State:
    """
    Get the current state of the environment.
    """
    raise NotImplementedError