"""
Memory replay buffer for reinforcement learning training.
"""
import pathlib
import pickle

from deeplearning.clgen.reinforcement_learning import interactions
from deeplearning.clgen.util import environment
from deeplearning.clgen.util import distrib

class Memory(object):
  """
  Replay buffer of previous states and actions.
  """
  def __init__(self, cache_path: pathlib.Path):

    self.cache_path = cache_path / "memory"
    if environment.WORLD_RANK == 0:
      self.cache_path.mkdir(exists_ok = True, parents = True)

    self.action_buffer = []
    self.state_buffer  = []
    self.reward_buffer = []

    self.loadCheckpoint()
    return

  def add(self,
          action : interactions.Action,
          state  : interactions.State,
          reward : interactions.Reward
          ) -> None:
    """Add single step to memory buffers."""
    self.action_buffer.append(action)
    self.state_buffer.append(state)
    self.reward_buffer.append(reward)
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
