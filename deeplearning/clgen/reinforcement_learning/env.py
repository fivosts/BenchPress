"""
RL Environment for the task of targeted benchmark generation.
"""
import numpy as np

from deeplearning.clgen.features import extractor
from deeplearning.clgen.features import feature_sampler
from deeplearning.clgen.preprocessors import opencl
from deeplearning.clgen.reinforcement_learning import interactions

from absl import flags

class Environment(object):
  """
  Environment representation for RL Agents.
  """
  def __init__(self) -> None:
    self.feature_space = feature_space
    return
  
  def compute_reward(self, action) -> interactions.Reward:
    """Compute an action's reward."""
    if action.action_type == interactions.ACTION_TYPE_SPACE['ADD'] or action.action_type == interactions.ACTION_TYPE_SPACE['REM']:
      return interactions.Reward(
        action   = action,
        value    = 0.0,
        distance = None,
        comment  = "[ADD] action, reward is +0."
      )
    elif action.action_type == interactions.ACTION_TYPE_SPACE['COMP']:
      source = self.tokenizer.ArrayToCode(self.current_state.code)
      try:
        _ = opencl.Compile(source)
        feats = extractor.ExtractFeatures(source, [self.feature_space])
        compiles = True
      except ValueError:
        compiles = False
      if not compiles:
        return interactions.Reward(
          action   = action,
          value    = -1,
          distance = None,
          comment  = "[COMPILE] action failed and reward is -1.",
        )
      else:
        dist = feature_sampler.euclidean_distance(feats, self.current_state.target_features)
        if dist == 0:
          return interactions.Reward(
            action   = action,
            value    = 1.0,
            distance = dist,
            comment  = "[COMPILE] led to dropping distance to 0, reward is +1!"
          )
        else:
          return interactions.Reward(
            action = action,
            value  = 1.0 / (dist),
            distance = dist,
            comment = "[COMPILE] succeeded, reward is {}, new distance is {}".format(1.0 / dist, dist)
          )
    else:
      raise ValueError("Action type {} does not exist.".format(action.action_type))

  def step(self, action) -> interactions.Reward:
    """
    Collect an action from an agent and compute its reward.
    """
    raise NotImplementedError
  
  def reset(self) -> None:
    """
    Reset the state of the environment.
    """
    raise NotImplementedError
  
  def get(self) -> interactions.State:
    """
    Get the current state of the environment.
    """
    return self.current_state

class Memory(object):
  """
  Replay buffer of previous states and actions.
  """
  def __init__(self):
    self.action_buffer = []
    self.state_buffer  = []
    self.reward_buffer = []
    self.loadCheckpoint()
    return

  def add(self, action, state, reward) -> None:
    """Add single step to memory buffers."""
    self.action_buffer.append(action)
    self.state_buffer.append(state)
    self.reward_buffer.append(reward)
    return
  
  def loadCheckpoint(self) -> None:
    """Fetch memory's latest state."""
    return
  
  def saveCheckpoint(self) -> None:
    """Save Checkpoint state."""
    return