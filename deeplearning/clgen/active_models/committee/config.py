""" Configuration base class for committee models."""
import typing

from deeplearning.clgen.util import pbutil
from deeplearning.clgen.active_models import downstream_tasks
from deeplearning.clgen.proto import active_learning_pb2

def AssertConfigIsValid(config: active_learning_pb2.ActiveLearner) -> active_learning_pb2.ActiveLearner:
  """
  Parse proto description and check for validity.
  """
  tm = 0
  pbutil.AssertFieldIsSet(config.committee.random_seed)
  for nn in config.committee.mlp:
    tl = 0
    for l in nn.layer:
      if l.HasField("linear"):
        pbutil.AssertFieldIsSet(l.linear, "in_features")
        pbutil.AssertFieldIsSet(l.linear, "out_features")
      elif l.HasField("dropout"):
        pbutil.AssertFieldIsSet(l.dropout, "dropout_prob")
      elif l.HasField("layer_norm"):
        pbutil.AssertFieldIsSet(l.layer_norm, "layer_norm_eps")
      elif l.HasField("act_fn"):
        pbutil.AssertFieldIsSet(l.act_fn, "act_fn")
      else:
        raise AttributeError(l)
      tl += 1
    assert tl > 0, "Model is empty. No layers found."
    tm += 1
  ## Add another for loop here if more committee model types are added.
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
