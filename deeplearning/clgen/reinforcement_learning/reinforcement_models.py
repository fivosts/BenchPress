"""
RL Environment for the task of targeted benchmark generation.
"""
import pathlib

from deeplearning.clgen.models import backends
from deeplearning.clgen.proto import reinforcement_learning_pb2
from deeplearning.clgen.util import logging as l
from deeplearning.clgen.util import distrib
from deeplearning.clgen.util import commit
from deeplearning.clgen.util import environment

from absl import flags

def AssertConfigIsValid(config: reinforcement_learning_pb2.RLModel) -> reinforcement_learning_pb2.RLModel:
  """
  Check validity of RL Model config.
  """
  raise NotImplementedError("TODO")
  return config

class RLModel(object):
  """
  Manager class of Reinforcement Learning pipeline for benchmark generation.
  """
  def __init__(self, config: reinforcement_learning_pb2.RLModel, cache_path: pathlib.Path):
    """
    Initialize RL manager.
    """
    # Error early, so that a cache isn't created.
    if not isinstance(config, reinforcement_learning_pb2.RLModel):
      t = type(config).__name__
      raise TypeError(f"Config must be an RLModel proto. Received: '{t}'")

    self.config = AssertConfigIsValid(config)
    self.cache_path = cache_path / "reinforcement_learning"

    if environment.WORLD_RANK == 0:
      self.cache_path.mkdir(exist_ok = True, parents = True)
    distrib.barrier()

    if environment.WORLD_RANK == 0:
      ## Store current commit
      commit.saveCommit(self.cache_path)

    """
    How do you target features during training ?
    1) Active learner - downstream task <- Sampler
    2) Random feasible vectors (collected from OpenCL corpus ?) <- Sampler ?
    3) Got from benchmark suites ? <- Sampler
    """

    self.env = env.Environment()
    self.agent = agent.Agent()
    return

  def Train(self) -> None:
    """
    Train the RL-Agent.
    """
    for ep in range(num_episodes):
      self.env.reset()
      target_features = self.feature_sampler.sample()
      self.env.init_state(target_features)
      is_term = False
      while not is_term:
        state  = self.env.get_state()
        action = self.agent.make_action(state)
        reward = self.env.step(action)
        self.memory.add(state, action, reward)
        self.agent.update(self.memory.sample())
    return
  
  def Sample(self, backend: backends.BackendBase) -> None:
    """
    Instead of calling Model's sample, this sample will be called, acting as a backend (BERT) wrapper.
    """
    return
