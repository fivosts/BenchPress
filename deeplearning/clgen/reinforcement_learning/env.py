"""
RL Environment for the task of targeted benchmark generation.
"""
import gym
import typing
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
from deeplearning.clgen.util import logging as l

torch = pytorch.torch

from absl import flags

class Environment(gym.Env):
  """
  Environment representation for RL Agents.
  """
  metadata = {
    'render_modes' : ['human'],
    'render_fps'   : 4,
  }
  @property
  def init_code_state(self) -> np.array:
    return np.array(
      [self.tokenizer.startToken, self.tokenizer.endToken]
      + ([self.tokenizer.padToken] * (self.max_position_embeddings - 2))
    )

  def __init__(self,
               config                  : reinforcement_learning_pb2.RLModel,
               max_position_embeddings : int,
               corpus                  : corpuses.Corpus,
               tokenizer               : tokenizers.TokenizerBase,
               feature_tokenizer       : tokenizers.FeatureTokenizer,
               cache_path              : pathlib.Path,
               ) -> None:
    self.config            = config
    self.tokenizer         = tokenizer
    self.feature_tokenizer = feature_tokenizer
    self.max_position_embeddings = max_position_embeddings
    self.feature_sequence_length = self.config.agent.feature_tokenizer.feature_sequence_length

    self.cache_path = cache_path / "environment"
    self.ckpt_path  = cache_path / "checkpoint"
    if environment.WORLD_RANK == 0:
      self.cache_path.mkdir(exist_ok = True, parents = True)
      self.ckpt_path.mkdir(exist_ok = True, parents = True)

    self.current_state = None
    self.feature_dataset = None
    self.loadCheckpoint()

    if self.feature_dataset is None:
      self.feature_dataset = []  
      if self.config.HasField("train_set"):
        data = corpus.GetTrainingFeatures()
        for dp in data:
          for k, v in dp.items():
            if v:
              self.feature_dataset.append((k, v))
      elif self.config.HasField("random"):
        self.feature_dataset = []
    return

  def intermediate_step(self,
                        state_code   : torch.LongTensor,
                        step_actions : torch.LongTensor,
                        ) -> typing.Tuple[torch.Tensor]:
    """
    The environment reads the predicted index, and makes
    necessary transformations to the input ids so they can be
    fed into the language model, if need be.
    """
    num_episodes = step_actions.shape[0]
    lm_input_ids = torch.zeros(state_code.shape, dtype = torch.long)
    use_lm       = torch.zeros((num_episodes), dtype = torch.bool)
    for idx, (code, action) in enumerate(zip(state_code, step_actions)):
      act_type  = int(action) % len(interactions.ACTION_TYPE_SPACE)
      act_index = int(action) // len(interactions.ACTION_TYPE_SPACE)
      if act_type == interactions.ACTION_TYPE_SPACE['ADD']:
        new_code = torch.cat((code[:act_index + 1], torch.LongTensor([self.tokenizer.holeToken]), code[act_index + 1:]))
        new_code = new_code[:code.shape[0]]
        lm_input_ids[idx] = new_code
        use_lm[idx]       = True
      elif act_type == interactions.ACTION_TYPE_SPACE['REPLACE']:
        new_code            = torch.clone(code)
        new_code[act_index] = self.tokenizer.holeToken
        lm_input_ids[idx]   = new_code
        use_lm[idx]         = True
    return use_lm, lm_input_ids

  def new_step(self,
               state_code        : torch.LongTensor,
               step_actions      : torch.LongTensor,
               step_tokens       : torch.LongTensor,
               traj_disc_rewards : torch.FloatTensor,
               use_lm            : torch.BoolTensor,
               gamma             : float,
              ) -> typing.Tuple[torch.Tensor]:
    """
    Step the environment, compute the reward.
    """
    super().reset()
    num_episodes      = step_actions.shape[0]
    reward            = torch.zeros((num_episodes), dtype = torch.float32)
    discounted_reward = torch.zeros((num_episodes), dtype = torch.float32)
    done              = torch.zeros((num_episodes), dtype = torch.bool)

    for idx, (code, act, tok, dr, lm) in enumerate(zip(state_code, step_actions, step_tokens, discounted_reward, use_lm)):
      act_type  = int(act) % len(interactions.ACTION_TYPE_SPACE)
      act_index = int(act) // len(interactions.ACTION_TYPE_SPACE)
      token_id  = int(tok)
      lm        = bool(lm)
      real_len  = torch.where(code == self.tokenizer.endToken)[0][0]
      if act_index >= real_len and act_type != interactions.ACTION_TYPE_SPACE['COMP']:
        reward[idx] = -1.0

      if act_type == interactions.ACTION_TYPE_SPACE['ADD']:
        new_code = torch.cat((code[:act_index + 1], torch.LongTensor([token_id]), code[act_index + 1:]))
        new_code = new_code[:code.shape[0]]
        state_code[idx] = new_code
      elif act_type == interactions.ACTION_TYPE_SPACE['REM']:
        new_code = torch.cat((code[:act_index], code[act_index + 1:], torch.LongTensor([self.tokenizer.padToken])))
        state_code[idx] = new_code
      elif act_type == interactions.ACTION_TYPE_SPACE['REPLACE']:
        state_code[idx][act_index] = token_id
      elif act_type == interactions.ACTION_TYPE_SPACE['COMP']:
        src = self.tokenizer.ArrayToCode([int(x) for x in code])
        try:
          _ = opencl.Compile(src)
          features = extractor.ExtractFeatures(code, ext = [self.current_state.feature_space])
          compiles = True
        except ValueError:
          compiles = False
          features = None
        if compiles and len(src) > 0:
          cur_dist = feature_sampler.calculate_distance(
            features[self.current_state.feature_space],
            self.current_state.target_features,
          )
          if cur_dist == 0:
            done[idx] = True
            reward[idx] = 1.0
          else:
            reward[idx] = 1 / cur_dist
      else:
        raise ValueError("Invalid action type: {}".format(act_type))
    discounted_reward = traj_disc_rewards * gamma + reward
    return state_code, reward, discounted_reward, done

  def reset(self, recycle: bool = True) -> interactions.State:
    """
    Reset the state of the environment.
    """
    if recycle and self.current_state:
      self.feature_dataset.append(
        (self.current_state.feature_space, self.current_state.target_features)
      )
    next = self.feature_dataset.pop(0)
    self.current_state = interactions.State(
      target_features  = next[1],
      feature_space    = next[0],
      encoded_features = self.feature_tokenizer.TokenizeFeatureVector(next[1], next[0], self.feature_sequence_length),
      code             = "",
      encoded_code     = self.init_code_state,
      comment          = "State: \nCode:\n\nFeatures:\n{}".format(next[1]),
    )
    return self.current_state
  
  def get_state(self) -> interactions.State:
    """
    Get the current state of the environment.
    """
    return self.current_state

  def loadCheckpoint(self) -> None:
    """
    Load environment checkpoint.
    """
    if (self.ckpt_path / "environment.pkl").exists():
      distrib.lock()
      with open(self.ckpt_path / "environment.pkl", 'rb') as inf:
        self.current_state = pickle.load(inf)
      distrib.unlock()
      distrib.barrier()

    if (self.ckpt_path / "feature_loader.pkl").exists():
      distrib.lock()
      with open(self.ckpt_path / "feature_loader.pkl", 'rb') as inf:
        self.feature_loader = pickle.load(inf)
      distrib.unlock()
      distrib.barrier()
    return

  def saveCheckpoint(self) -> None:
    """
    Save environment state.
    """
    if environment.WORLD_RANK == 0:
      with open(self.ckpt_path / "environment.pkl", 'wb') as outf:
        pickle.dump(self.current_state, outf)
      with open(self.ckpt_path / "feature_loader.pkl", 'wb') as outf:
        pickle.dump(self.feature_loader, outf)
    distrib.barrier()
    return
