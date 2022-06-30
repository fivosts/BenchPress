"""
Agents module for reinforcement learning.
"""
import pathlib
import typing
import numpy as np

from deeplearning.clgen.reinforcement_learning import interactions
from deeplearning.clgen.reinforcement_learning import model
from deeplearning.clgen.reinforcement_learning.config import QValuesConfig
from deeplearning.clgen.models import language_models
from deeplearning.clgen.proto import reinforcement_learning_pb2
from deeplearning.clgen.corpuses import tokenizers
from deeplearning.clgen.util import pytorch
from deeplearning.clgen.util import environment

torch = pytorch.torch

class Policy(object):
  """
  The policy selected over Q-Values
  """
  def __init__(self):
    return

  def SelectAction(self, type_logits: torch.FloatTensor, index_logits: torch.Tensor) -> typing.Tuple[int, int]:
    """
    Get the Q-Values for action and apply policy on it.
    """
    return 0, 0
    raise NotImplementedError
    return action_type

  def SelectToken(self, token_logits: torch.FloatTensor) -> int:
    """
    Get logit predictions for token and apply policy on it.
    """
    return 54
    raise NotImplementedError
    return token

class Agent(object):
  """
  Benchmark generation RL-Agent.
  """
  def __init__(self,
              config            : reinforcement_learning_pb2.RLModel,
              language_model    : language_models.Model,
              tokenizer         : tokenizers.TokenizerBase,
              feature_tokenizer : tokenizers.FeatureTokenizer,
              cache_path: pathlib.Path
              ):

    self.cache_path = cache_path / "agent"
    if environment.WORLD_RANK == 0:
      self.cache_path.mkdir(exist_ok = True, parents = True)

    self.config            = config
    self.language_model    = language_model
    self.tokenizer         = tokenizer
    self.feature_tokenizer = feature_tokenizer
    self.qv_config = QValuesConfig.from_config(self.config, self.tokenizer, self.feature_tokenizer)
    self.q_model = model.QValuesModel(
      self.language_model, self.feature_tokenizer, self.qv_config, self.cache_path
    )
    self.policy  = Policy()

    self.loadCheckpoint()
    return

  def make_action(self, state: interactions.State) -> interactions.Action:
    """
    Agent collects the current state by the environment
    and picks the right action.
    """
    type_logits, index_logits  = self.q_model.SampleAction(state)
    action_type, action_index  = self.policy.SelectAction(type_logits, index_logits)

    if action_type == interactions.ACTION_TYPE_SPACE['ADD']:
      token_logits = self.q_model.SampleToken(
        state, action_index, self.tokenizer, self.feature_tokenizer
      )
      token        = self.policy.SelectToken(token_logits)
    else:
      token_logits, token = None, None

    return interactions.Action(
      action_type         = action_type,
      action_type_logits  = type_logits,
      action_index        = action_index,
      action_index_logits = index_logits,
      token_type          = token,
      token_type_logits   = token_logits,
    )

  def update_agent(self, input_ids: typing.Dict[str, torch.Tensor]) -> None:
    """
    Train the agent on the new episodes.
    """
    self.q_model.Train(input_ids)
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
