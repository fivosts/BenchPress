"""
Agents module for reinforcement learning.
"""
import pathlib
import typing
import numpy as np

from deeplearning.clgen.reinforcement_learning import interactions
from deeplearning.clgen.reinforcement_learning import model
from deeplearning.clgen.reinforcement_learning import env
from deeplearning.clgen.reinforcement_learning.config import QValuesConfig
from deeplearning.clgen.models import language_models
from deeplearning.clgen.proto import reinforcement_learning_pb2
from deeplearning.clgen.corpuses import tokenizers
from deeplearning.clgen.util import pytorch
from deeplearning.clgen.util import environment
from deeplearning.clgen.util import logging as l

torch = pytorch.torch

class Policy(object):
  """
  The policy selected over Q-Values
  """
  def __init__(self, action_temp: float, idx_temp: float, token_temp: float):
    self.action_temperature = action_temp
    self.index_temperature  = idx_temp
    self.token_temperature  = token_temp
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
    ct = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(
        temperature = self.temperature if self.temperature is not None else 1.0,
        logits = t,
        validate_args = False if "1.9." in torch.__version__ else None,
      ).sample()

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
               cache_path        : pathlib.Path
               ):

    self.cache_path = cache_path / "agent"
    if environment.WORLD_RANK == 0:
      self.cache_path.mkdir(exist_ok = True, parents = True)

    self.config            = config
    self.language_model    = language_model
    self.tokenizer         = tokenizer
    self.feature_tokenizer = feature_tokenizer
    self.qv_config = QValuesConfig.from_config(
      self.config,
      self.language_model.backend.config.architecture.max_position_embeddings,
      self.tokenizer,
      self.feature_tokenizer,
    )
    self.q_model = model.QValuesModel(
      self.language_model, self.feature_tokenizer, self.qv_config, self.cache_path
    )
    self.critic_model = model.QValuesModel(
      self.language_model, self.feature_tokenizer, self.qv_config, self.cache_path, is_critic = True
    )
    self.policy  = Policy(
      action_temp = self.config.agent.action_qv.action_type_temperature_micros / 10e6,
      idx_temp    = self.config.agent.action_qv.action_index_temperature_micros / 10e6,
      token_temp  = self.config.agent.action_lm.token_temperature_micros / 10e6,
    )
    self.loadCheckpoint()
    return

  def Train(self, env: env.Environment, num_epochs: int) -> None:
    """
    Run PPO over policy and train the agent.
    """
    for ep in range(num_epochs):
      batch_states, batch_actions, batch_logits, batch_rtgs, batch_lens = self.rollout()
    return

  def rollout(self) -> typing.Tuple:
    """
    Play a number of episodes and collect batched data.
    """
    batch_states, batch_actions, batch_logits, batch_rtgs, batch_lens = [], [], [], [], []
    raise NotImplementedError
    return batch_states, batch_actions, batch_logits, batch_rtgs, batch_lens

  def make_action(self, state: interactions.State) -> interactions.Action:
    """
    Agent collects the current state by the environment
    and picks the right action.
    """
    logits = self.q_model.SampleAction(state)
    action_logits = logits['action_logits'].cpu().numpy()
    index_logits  = logits['index_logits'].cpu().numpy()
    action_type, action_index  = self.policy.SelectAction(action_logits, index_logits)

    comment = "Action: {}".format(interactions.ACTION_TYPE_MAP[action_type])

    if action_type == interactions.ACTION_TYPE_SPACE['ADD']:
      logits = self.q_model.SampleToken(
        state, action_index, self.tokenizer, self.feature_tokenizer
      )
      token_logits = logits['prediction_logits'].cpu().numpy()
      token        = self.policy.SelectToken(token_logits)
      comment      += ", index: {}, token: '{}'".format(action_index, self.tokenizer.decoder[token])
    elif action_type == interactions.ACTION_TYPE_SPACE['REM']:
      token_logits, token = None, None
      comment += ", index: {}".format(action_index)
    elif action_type == interactions.ACTION_TYPE_SPACE['COMP']:
      token_logits, token = None, None
    else:
      raise ValueError("Invalid action_type: {}".format(action_type))

    return interactions.Action(
      action_type         = action_type,
      action_type_logits  = action_logits,
      action_index        = action_index,
      action_index_logits = index_logits,
      token_type          = token,
      token_type_logits   = token_logits,
      comment             = comment,
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
