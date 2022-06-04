"""
Agents module for reinforcement learning.
"""
import typing
import numpy as np

ACTION_TYPE_SPACE = {
  'ADD' : 0,
  'REM' : 1,
  'COMP': 2, 
}

class Action(typing.NamedTuple):
  """
  Agent action representation.
  """
  action_type         : int      # Your policy function picks the best action type.
  action_type_logits  : np.array # This must be a distribution vector over action space.
  action_index        : int      # Your policy function picks the best index to apply the policy.
  action_index_logits : np.array # Distribution vector over size of input code (or max length).
  token_type          : int      # Your policy function picks the best token.
  token_type_logits   : np.array # Distribution vector over possible tokens.

class Agent(object):
  """
  Benchmark generation RL-Agent.
  """
  def __init__(self):
    raise NotImplementedError("TODO")
    return
