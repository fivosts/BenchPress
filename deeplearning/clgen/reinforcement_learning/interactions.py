"""
A module containing all possible interactions between
the environment and an agent.
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
  action_type         : int        # Your policy function picks the best action type.
  action_type_logits  : np.array   # This must be a distribution vector over action space.
  action_index        : int        # Your policy function picks the best index to apply the policy.
  action_index_logits : np.array   # Distribution vector over size of input code (or max length).
  token_type          : int        # Your policy function picks the best token.
  token_type_logits   : np.array   # Distribution vector over possible tokens.

class State(typing.NamedTuple):
  """
  Environment's state representation.
  """
  target_features  : typing.Dict[str, float]
  feature_space    : str
  encoded_features : np.array
  code             : str
  encoded_code     : np.array

class Reward(typing.NamedTuple):
  """
  Reward provided to agent as feedback.
  """
  action   : Action
  value    : float
  distance : float
  comment  : str
