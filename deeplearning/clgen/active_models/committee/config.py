""" Configuration base class for committee models."""
import typing
import pathlib

from deeplearning.clgen.active_models import downstream_tasks
from deeplearning.clgen.proto import active_learning_pb2
from deeplearning.clgen.util import pbutil
from deeplearning.clgen.util import crypto

def AssertConfigIsValid(config: active_learning_pb2.ActiveLearner) -> active_learning_pb2.ActiveLearner:
  """
  Parse proto description and check for validity.
  """
  tm = 0
  pbutil.AssertFieldIsSet(config, "training_corpus")
  p = pathlib.Path(config.training_corpus).resolve()
  if not p.exists():
    raise FileNotFoundError(p)
  pbutil.AssertFieldIsSet(config.committee, "random_seed")
  for nn in config.committee.mlp:
    tl = 0
    pbutil.AssertFieldIsSet(nn, "initial_learning_rate_micros")
    pbutil.AssertFieldIsSet(nn, "batch_size")
    pbutil.AssertFieldIsSet(nn, "num_train_steps")
    pbutil.AssertFieldIsSet(nn, "num_warmup_steps")
    for l in nn.layer:
      if l.HasField("embedding"):
        pbutil.AssertFieldIsSet(l.embedding, "num_embeddings")
        pbutil.AssertFieldIsSet(l.embedding, "embedding_dim")
      elif l.HasField("linear"):
        pbutil.AssertFieldIsSet(l.linear, "in_features")
        pbutil.AssertFieldIsSet(l.linear, "out_features")
      elif l.HasField("dropout"):
        pbutil.AssertFieldIsSet(l.dropout, "dropout_prob")
      elif l.HasField("layer_norm"):
        pbutil.AssertFieldIsSet(l.layer_norm, "layer_norm_eps")
      elif l.HasField("act_fn"):
        pbutil.AssertFieldIsSet(l.act_fn, "fn")
      else:
        raise AttributeError(l)
      tl += 1
    assert tl > 0, "Model is empty. No layers found."
    tm += 1
  ## Add another for loop here if more committee model types are added.
  assert tm > 0, "Committee is empty. No models found."
  return

class ModelConfig(object):

  model_type = "committee"

  @classmethod
  def FromConfig(cls,
                 config: active_learning_pb2.Committee,
                 downstream_task: downstream_tasks.DownstreamTask
                 ) -> typing.List["ModelConfig"]:
    return [ModelConfig(m, downstream_task) for m in config.mlp] # Extend for more model types.

  @property
  def num_labels(self) -> int:
    """
    The number of output labels for classification models.
    """
    return self.downstream_task.output_size

  @property
  def num_features(self) -> int:
    """
    The number of input features to model committee.
    """
    return self.downstream_task.input_size

  def __init__(self,
               config          : typing.Union[active_learning_pb2.MLP],
               downstream_task : downstream_tasks.DownstreamTask,
               ) -> "ModelConfig":
    if isinstance(config, active_learning_pb2.MLP):
      self.name = "MLP"
    self.config           = config
    self.downstream_task  = downstream_task
    self.sha256           = crypto.sha256_str(str(config))

    self.num_train_steps  = config.num_train_steps
    self.num_warmup_steps = config.num_warmup_steps
    self.num_epochs       = 1
    self.steps_per_epoch  = config.num_train_steps
    self.batch_size       = config.batch_size

    self.learning_rate    = config.initial_learning_rate_micros / 1e6
    self.max_grad_norm    = 1.0

    if len(self.config.layer) == 0:
      raise ValueError("Layer list is empty for committee model")
    if self.config.layer[0].HasField("linear"):
      if self.config.layer[0].linear.in_features != self.downstream_task.input_size:
        raise ValueError("Mismatch between committee member's input size {} and downstream task's input size {}".format(
            self.config.layer[0].linear.in_features,
            self.downstream_task.input_size
          )
        )
    if self.config.layer[-1].HasField("linear"):
      if self.config.layer[-1].linear.out_features != self.downstream_task.output_size:
        raise ValueError("Mismatch between committee member's output size {} and downstream task's output size {}".format(
            self.config.layer[-1].linear.out_features,
            self.downstream_task.output_size
          )
        )
    self.layer_config = []
    for l in self.config.layer:
      if l.HasField("embedding"):
        self.layer_config.append((
            'Embedding', {
              'num_embeddings': l.embedding.num_embeddings,
              'embedding_dim' : l.embedding.embedding_dim,
              'padding_idx'   : l.embedding.padding_idx if l.embedding.HasField("padding_idx") else None
            }
          )
        )
      elif l.HasField("linear"):
        self.layer_config.append((
            'Linear', {
              'in_features': l.linear.in_features,
              'out_features': l.linear.out_features,
            }
          )
        )
      elif l.HasField("dropout"):
        self.layer_config.append((
            'Dropout', {
              'dropout_prob': l.dropout.dropout_prob
            }
          )
        )
      elif l.HasField("layer_norm"):
        self.layer_config.append((
            'LayerNorm', {
              'layer_norm_eps': l.dropout.layer_norm_eps
            }
          )
        )
      elif l.HasField("act_fn"):
        self.layer_config.append((
            l.act_fn.fn, {}
          )
        )
    return
