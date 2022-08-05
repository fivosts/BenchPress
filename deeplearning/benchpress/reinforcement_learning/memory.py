# coding=utf-8
# Copyright 2022 Foivos Tsimpourlas.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Memory replay buffer for reinforcement learning training.
"""
import pathlib
import typing
import pickle

from deeplearning.benchpress.reinforcement_learning import interactions
from deeplearning.benchpress.util import environment
from deeplearning.benchpress.util import distrib
from deeplearning.benchpress.util import pytorch

torch = pytorch.torch

class Memory(object):
  """
  Replay buffer of previous states and actions.
  """
  def __init__(self, cache_path: pathlib.Path):

    self.cache_path = cache_path / "memory"
    if environment.WORLD_RANK == 0:
      self.cache_path.mkdir(exist_ok = True, parents = True)

    self.action_buffer = []
    self.state_buffer  = []
    self.reward_buffer = []
    self.done_buffer   = []
    self.info_buffer   = []

    self.loadCheckpoint()
    return

  def add(self,
          action : interactions.Action,
          state  : interactions.State,
          reward : interactions.Reward,
          done   : bool,
          info   : str,
          ) -> None:
    """Add single step to memory buffers."""
    self.action_buffer.append(action)
    self.state_buffer.append(state)
    self.reward_buffer.append(reward)
    self.done_buffer.append(done)
    self.info_buffer.append(info)
    return

  def sample(self) -> typing.Dict[str, torch.Tensor]:
    """
    Sample memories to update the RL agent.
    """
    return

  def loadCheckpoint(self) -> None:
    """Fetch memory's latest state."""
    if (self.cache_path / "memory.pkl").exists():
      distrib.lock()
      with open(self.cache_path / "memory.pkl", 'rb') as inf:
        checkpoint = pickle.load(inf)
      distrib.unlock()
      self.action_buffer = checkpoint['action_buffer']
      self.action_buffer = checkpoint['state_buffer']
      self.action_buffer = checkpoint['reward_buffer']
    return
  
  def saveCheckpoint(self) -> None:
    """Save Checkpoint state."""
    if environment.WORLD_RANK == 0:
      checkpoint = {
        'action_buffer' : self.action_buffer,
        'reward_buffer' : self.reward_buffer,
        'state_buffer'  : self.state_buffer,
      }
      with open(self.cache_path / "memory.pkl", 'wb') as outf:
        pickle.dump(checkpoint, outf)
    distrib.barrier()
    return
