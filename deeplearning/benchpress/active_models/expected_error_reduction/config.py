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
Config setup for expected error reduction active learner.
"""
import typing
import pathlib

from deeplearning.benchpress.active_models import downstream_tasks
from deeplearning.benchpress.proto import active_learning_pb2
from deeplearning.benchpress.util import pbutil
from deeplearning.benchpress.util import crypto

def AssertConfigIsValid(config: active_learning_pb2.ExpectedErrorReduction) -> None:
  """
  Parse proto description and check for validity.
  """
  if config.HasField("mlp"):
    tl = 0
    pbutil.AssertFieldIsSet(config.mlp, "initial_learning_rate_micros")
    pbutil.AssertFieldIsSet(config.mlp, "batch_size")
    pbutil.AssertFieldIsSet(config.mlp, "num_warmup_steps")
    for l in config.mlp.layer:
      if l.HasField("embedding"):
        pbutil.AssertFieldIsSet(l.embedding, "num_embeddings")
        pbutil.AssertFieldIsSet(l.embedding, "embedding_dim")
      elif l.HasField("linear"):
        pbutil.AssertFieldIsSet(l.linear, "in_features")
        pbutil.AssertFieldIsSet(l.linear, "out_features")
      elif l.HasField("dropout"):
        pbutil.AssertFieldIsSet(l.dropout, "p")
      elif l.HasField("layer_norm"):
        pbutil.AssertFieldIsSet(l.layer_norm, "normalized_shape")
        pbutil.AssertFieldIsSet(l.layer_norm, "eps")
      elif l.HasField("act_fn"):
        pbutil.AssertFieldIsSet(l.act_fn, "fn")
      else:
        raise AttributeError(l)
      tl += 1
    assert tl > 0, "Model is empty. No layers found."
  return

class ModelConfig(object):

  model_type = "expected_error_reduction"

  @classmethod
  def FromConfig(cls,
                 config: active_learning_pb2.ExpectedErrorReduction,
                 downstream_task: downstream_tasks.DownstreamTask,
                 num_train_steps: int,
                 ) -> typing.List["ModelConfig"]:
    return NNModelConfig(config.mlp, downstream_task, num_train_steps)

  @property
  def num_labels(self) -> int:
    """
    The number of output labels for classification models.
    """
    return self.downstream_task.output_size

  @property
  def num_features(self) -> int:
    """
    The number of input features to model.
    """
    return self.downstream_task.input_size

  def __init__(self,
               name        : str,
               config          : typing.Union[active_learning_pb2.MLP, active_learning_pb2.KMeans],
               downstream_task : downstream_tasks.DownstreamTask
               ) -> "ModelConfig":
    self.name            = name
    self.config          = config
    self.downstream_task = downstream_task
    self.sha256          = crypto.sha256_str(str(config))

    ## Placeholding initialization
    self.num_train_steps  = None
    self.num_warmup_steps = None
    self.num_epochs       = None
    self.steps_per_epoch  = None
    self.batch_size       = None
    self.learning_rate    = None
    self.max_grad_norm    = None
    self.layer_config     = None
    return

class NNModelConfig(ModelConfig):
  """
  NeuralNetwork-based architectural config.
  """
  def __init__(self,
               config          : active_learning_pb2.MLP,
               downstream_task : downstream_tasks.DownstreamTask,
               num_train_steps : int
               ) -> "ModelConfig":
    super(NNModelConfig, self).__init__("MLP", config, downstream_task)

    ## NN-specific attributes
    self.num_train_steps  = (num_train_steps + config.batch_size) // config.batch_size
    self.num_warmup_steps = config.num_warmup_steps
    self.num_epochs       = 1
    self.steps_per_epoch  = self.num_train_steps
    self.batch_size       = config.batch_size

    self.learning_rate    = config.initial_learning_rate_micros / 1e6
    self.max_grad_norm    = 1.0

    if len(self.config.layer) == 0:
      raise ValueError("Layer list is empty for model")
    if self.config.layer[0].HasField("linear"):
      if self.config.layer[0].linear.in_features != self.downstream_task.input_size:
        raise ValueError("Mismatch between model's input size {} and downstream task's input size {}".format(
            self.config.layer[0].linear.in_features,
            self.downstream_task.input_size
          )
        )
    if self.config.layer[-1].HasField("linear"):
      if self.config.layer[-1].linear.out_features != self.downstream_task.output_size:
        raise ValueError("Mismatch between model's output size {} and downstream task's output size {}".format(
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
              'p': l.dropout.p,
            }
          )
        )
      elif l.HasField("layer_norm"):
        self.layer_config.append((
            'LayerNorm', {
              'normalized_shape': l.layer_norm.normalized_shape,
              'eps': l.layer_norm.eps,
            }
          )
        )
      elif l.HasField("act_fn"):
        self.layer_config.append((
            l.act_fn.fn, {}
          )
        )
    return
