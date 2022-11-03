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
""" Configuration base class for committee models."""
import typing
import pathlib

from deeplearning.benchpress.active_models import downstream_tasks
from deeplearning.benchpress.proto import active_learning_pb2
from deeplearning.benchpress.util import pbutil
from deeplearning.benchpress.util import crypto

def AssertConfigIsValid(config: active_learning_pb2.ActiveLearner.QueryByCommittee) -> None:
  """
  Parse proto description and check for validity.
  """
  tm = 0
  ## Parse all MLPs.
  for nn in config.mlp:
    tl = 0
    pbutil.AssertFieldIsSet(nn, "initial_learning_rate_micros")
    pbutil.AssertFieldIsSet(nn, "batch_size")
    pbutil.AssertFieldIsSet(nn, "num_warmup_steps")
    for l in nn.layer:
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
    tm += 1
  ## Parse all KMeans algos.
  for km in config.k_means:
    pbutil.AssertFieldIsSet(km, "n_clusters")
    pbutil.AssertFieldConstraint(
      km,
      "init",
      lambda x: x in {"k-means++", "random"},
      "KMeans algorithm can only be 'k-means++' or 'random'."
    )
    pbutil.AssertFieldIsSet(km, "n_init")
    pbutil.AssertFieldIsSet(km, "max_iter")
    pbutil.AssertFieldIsSet(km, "tol")
    pbutil.AssertFieldConstraint(
      km,
      "algorithm",
      lambda x : x in {"auto", "full", "elkan"},
      "KMeans algorithm can only be 'auto', 'full' or 'elkan'."
    )
    tm += 1
  ## Parse KNN algos.
  for k in config.knn:
    pbutil.AssertFieldIsSet(k, "n_neighbors")
    pbutil.AssertFieldConstraint(
      k,
      "weights",
      lambda x: x in {"uniform", "distance"},
      "KNN weights can only be 'uniform' or 'distance'."
    )
    pbutil.AssertFieldConstraint(
      k,
      "algorithm",
      lambda x: x in {"auto", "ball_tree", "kd_tree", "brute"},
      "KNN algorithm can only be 'auto', 'ball_tree', 'kd_tree' or 'brute'."
    )
    pbutil.AssertFieldIsSet(k, "leaf_size")
    pbutil.AssertFieldIsSet(k, "p")
  ## Add another for loop here if more committee model types are added.
  assert tm > 0, "Committee is empty. No models found."
  return

class ModelConfig(object):

  model_type = "committee"

  @classmethod
  def FromConfig(cls,
                 config: active_learning_pb2.Committee,
                 downstream_task: downstream_tasks.DownstreamTask,
                 num_train_steps: int,
                 ) -> typing.List["ModelConfig"]:
    model_configs = []
    nts = num_train_steps
    for m in config.mlp:
      model_configs.append(NNModelConfig(m, downstream_task, nts))
    for m in config.k_means:
      model_configs.append(KMeansModelConfig(m, downstream_task, nts))
    for m in config.knn:
      model_configs.append(KNNModelConfig(m, downstream_task, nts))
    return model_configs

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
               name: str,
               config : typing.Union[active_learning_pb2.MLP, active_learning_pb2.KMeans],
               downstream_task: downstream_tasks.DownstreamTask
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
    self.n_clusters       = None
    self.init             = None
    self.n_init           = None
    self.max_iter         = None
    self.tol              = None
    self.algorithm        = None
    self.n_neighbors      = None
    self.weights          = None
    self.algorithm        = None
    self.leaf_size        = None
    self.p                = None
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

class KMeansModelConfig(ModelConfig):
  """
  KMeans config subclass.
  """
  def __init__(self,
               config          : active_learning_pb2.KMeans,
               downstream_task : downstream_tasks.DownstreamTask,
               num_train_steps : int
               ) -> "ModelConfig":
    super(KMeansModelConfig, self).__init__("KMeans", config, downstream_task)

    ## KMeans-specific attributes.
    self.n_clusters      = self.config.n_clusters
    self.init            = self.config.init
    self.n_init          = self.config.n_init
    self.max_iter        = self.config.max_iter
    self.tol             = self.config.tol
    self.algorithm       = self.config.algorithm
    self.num_train_steps = num_train_steps
    return

class KNNModelConfig(ModelConfig):
  """
  KNN config subclass.
  """
  def __init__(self,
               config          : active_learning_pb2.KMeans,
               downstream_task : downstream_tasks.DownstreamTask,
               num_train_steps : int
               ) -> "ModelConfig":
    super(KNNModelConfig, self).__init__("KNN", config, downstream_task)

    ## KMeans-specific attributes.
    self.n_neighbors     = self.config.n_neighbors
    self.weights         = self.config.weights
    self.algorithm       = self.config.algorithm
    self.leaf_size       = self.config.leaf_size
    self.p               = self.config.p
    self.num_train_steps = num_train_steps
    return
