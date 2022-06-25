"""
RL Environment for the task of targeted benchmark generation.
"""
import pathlib
import pickle
import numpy as np

from deeplearning.clgen.corpuses import tokenizers
from deeplearning.clgen.corpuses import corpuses
from deeplearning.clgen.features import extractor
from deeplearning.clgen.features import feature_sampler
from deeplearning.clgen.preprocessors import opencl
from deeplearning.clgen.reinforcement_learning import interactions
from deeplearning.clgen.reinforcement_learning import memory
from deeplearning.clgen.reinforcement_learning import data_generator
from deeplearning.clgen.proto import reinforcement_learning_pb2
from deeplearning.clgen.util import environment
from deeplearning.clgen.util import distrib
from deeplearning.clgen.util import pytorch

torch = pytorch.torch

from absl import flags

class Environment(object):
  """
  Environment representation for RL Agents.
  """
  def __init__(self,
               config            : reinforcement_learning_pb2.RLModel,
               corpus            : corpuses.Corpus,
               tokenizer         : tokenizers.TokenizerBase,
               feature_tokenizer : tokenizers.FeatureTokenizer,
               cache_path        : pathlib.Path,
               ) -> None:
    self.tokenizer = tokenizer
    self.feature_tokenizer = feature_tokenizer
    self.config = config
    self.cache_path = cache_path / "environment"
    if environment.WORLD_RANK == 0:
      self.cache_path.mkdir(exist_ok = True, parents = True)

    self.feature_dataset = []  
    if self.config.HasField("train_set"):
      data = corpus.GetTrainingFeatures()
      for dp in data:
        for k, v in dp.items():
          if v:
            self.feature_dataset.append((k, v))
    elif self.config.HasField("random"):
      self.feature_dataset = []


    self.loadCheckpoint()
    return

  def step(self, action: interactions.Action) -> interactions.Reward:
    """
    Collect an action from an agent and compute its reward.
    """
    self.make_action(action) # This is where you apply the action. e.g. compile, add token, remove token etc.
    raise NotImplementedError
  
  def reset(self) -> None:
    """
    Reset the state of the environment.
    """
    next = self.feature_dataset.pop(0)
    self.current_state = interactions.State(
      feature_space   = next[0],
      target_features = next[1],
      code            = self.tokenizer.TokenizeString("")
    )
    return
  
  def get_state(self) -> interactions.State:
    """
    Get the current state of the environment.
    """
    return self.current_state

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
            value  = 1.0 / (1 + dist),
            distance = dist,
            comment = "[COMPILE] succeeded, reward is {}, new distance is {}".format(1.0 / dist, dist)
          )
    else:
      raise ValueError("Action type {} does not exist.".format(action.action_type))

  def make_action(self, action: interactions.Action) -> None:
    """
    Collect an agent's action and proceed with it into the current state.
    """
    return

  def loadCheckpoint(self) -> None:
    """
    Load environment checkpoint.
    """
    if (self.cache_path / "environment.pkl").exists():
      distrib.lock()
      with open(self.cache_path / "environment.pkl", 'rb') as inf:
        self.current_state = pickle.load(inf)
      distrib.unlock()
      distrib.barrier()
    return

  def saveCheckpoint(self) -> None:
    """
    Save environment state.
    """
    if environment.WORLD_RANK == 0:
      with open(self.cache_path / "environment.pkl", 'wb') as outf:
        pickle.dump(self.current_state, outf)
    distrib.barrier()
    return
