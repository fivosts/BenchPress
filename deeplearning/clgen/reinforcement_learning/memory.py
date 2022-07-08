"""
Memory replay buffer for reinforcement learning training.
"""
import pathlib
import typing
import pickle

from deeplearning.clgen.reinforcement_learning import interactions
from deeplearning.clgen.util import environment
from deeplearning.clgen.util import distrib
from deeplearning.clgen.util import pytorch

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
