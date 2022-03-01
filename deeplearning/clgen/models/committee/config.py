""" Configuration base class for committee models."""
import typing

from deeplearning.clgen.models.committee import downstream_tasks
from deeplearning.clgen.models.proto import active_learning_pb2

def AssertConfigIsValid(config: active_learning_pb2:CommitteeConfig) -> active_learning_pb2:CommitteeConfig:
  """
  Parse proto description and check for validity.
  """

  return

class CommitteeConfig(object):

  model_type = "committee"

  @classmethod
  def FromConfig(cls,
                 config: active_learning_pb2.Committee,
                 downstream_task: downstream_tasks.DownstreamTask
                 ) -> "CommitteeConfig":
    config = CommitteeConfig(config, downstream_task)
    return config

  @property
  def num_labels(self) -> int:
    """
    The number of output labels for classification models.
    """
    return self.task.output_size

  @property
  def num_features(self) -> int:
    """
    The number of input features to model committee.
    """
    return self.input_size

  def __init__(self,
               config: active_learning_pb2.Committee,
               downstream_task: downstream_tasks.DownstreamTask
               ) -> "CommitteeConfig":
    self.config = config
    self.task   = downstream_task
    """
    The config must look like this:
    
    """
    return
