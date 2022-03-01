""" Configuration base class for committee models."""
import typing

from deeplearning.clgen.models.committee import downstream_tasks
from deeplearning.clgen.models.proto import active_learning_pb2

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
    :obj:`int`: The number of labels for classification models.
    """
    return len(self.id2label)

  def __init__(self,
               config: active_learning_pb2.Committee,
               downstream_task: downstream_tasks.DownstreamTask
               ) -> "CommitteeConfig":

    return