""" Configuration base class for committee models."""
import typing

from deeplearning.clgen.util import pbutil
from deeplearning.clgen.models.committee import downstream_tasks
from deeplearning.clgen.proto import active_learning_pb2

def AssertConfigIsValid(config: active_learning_pb2:ActiveLearner) -> active_learning_pb2:ActiveLearner:
  """
  Parse proto description and check for validity.
  """
  pbutil.AssertFieldConstraint(
    config,
    "downstream_task",
    lambda x: x in downstream_tasks.TASKS,
    "Downstream task has to be one of {}".format(', '.join([str(x) for x in downstream_tasks.TASKS]))
  )
  pbutil.AssertFieldIsSet(config, "committee")
  tm = 0
  for nn in config.committee.mlp:
    tl = 0
    for ln in nn.linear:
      pbutil.AssertFieldIsSet(ln, "in_features")
      pbutil.AssertFieldIsSet(ln, "out_features")
      tl += 1
    for ln in nn.dropout:
      pbutil.AssertFieldIsSet(ln, "dropout_prob")
      tl += 1
    for ln in nn.layer_norm:
      pbutil.AssertFieldIsSet(ln, "layer_norm_eps")
      tl += 1
    for ln in nn.act_fn:
      pbutil.AssertFieldIsSet(ln, "act_fn")
      tl += 1
    assert tl > 0, "Model is empty. No layers found."
    tm += 1
  ## Extend above for loop for other model architectures.
  assert tm > 0, "Committee is empty. No models found."
  return

class CommitteeConfig(object):

  model_type = "committee"

  @classmethod
  def FromConfig(cls, config: active_learning_pb2.ActiveLearner) -> "CommitteeConfig":
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

  def __init__(self, config: active_learning_pb2.ActiveLearner) -> "CommitteeConfig":
    self.config = config
    self.task   = config.downstream_task
    return
