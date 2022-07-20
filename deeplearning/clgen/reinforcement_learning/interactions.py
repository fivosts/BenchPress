"""
A module containing all possible interactions between
the environment and an agent.
"""
import typing
import numpy as np

ACTION_TYPE_SPACE = {
  'ADD'     : 0,
  'REM'     : 1,
  'COMP'    : 2,
  'REPLACE' : 3,
}

ACTION_TYPE_MAP = {
  v: k for k, v in ACTION_TYPE_SPACE.items()
}

class Action(typing.NamedTuple):
  """
  Agent action representation.
  """
  action         : int        # Selected action
  index          : int        # At selected index
  indexed_action : int        # Action/Index perplexed id in action head's output.
  action_logits  : np.array   # ACTION_SPACE * SEQ_LEN logits array.
  action_probs   : np.array   # ACTION_SPACE * SEQ_LEN probability array.
  token          : int        # Your policy function picks the best token.
  token_logits   : np.array   # Distribution logits over possible tokens.
  token_probs    : np.array   # Distribution probs over possible tokens.
  comment        : str        # Add action description.

class State(typing.NamedTuple):
  """
  Environment's state representation.
  """
  target_features  : typing.Dict[str, float]
  feature_space    : str
  encoded_features : np.array
  code             : str
  encoded_code     : np.array
  comment          : str

class Reward(typing.NamedTuple):
  """
  Reward provided to agent as feedback.
  """
  action   : Action
  value    : float
  distance : float
  comment  : str

class Memory(typing.NamedTuple):
  """
  A memory representation used for agent training.
  """
  state  : State   # Input state for memory.
  action : Action  # Action taken by agent.
  reward : Reward  # Isolated reward of that action.
  rtg    : float   # Reward-to-go from trajectory.
  length : int     # Current index within the trajectory.
