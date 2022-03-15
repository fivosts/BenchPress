"""
Here all the committee members are defined.
"""
import math
import typing
import numpy as np

from deeplearning.clgen.active_models.committee import config
from deeplearning.clgen.models.torch_bert import activations
from deeplearning.clgen.util import pytorch
from deeplearning.clgen.util.pytorch import torch

from deeplearning.clgen.util import logging as l

def mish(x):
  return x * torch.tanh(torch.nn.functional.softplus(x))

ACT2FN = {
  "gelu"     : activations.gelu,
  "relu"     : torch.nn.functional.relu,
  "swish"    : activations.swish,
  "gelu_new" : activations.gelu_new,
  "mish"     : mish,
  "softmax"  : torch.nn.Softmax
}

class CommitteeModels(object):
  """
  Abstract representation of model committee.
  """
  @classmethod
  def FromConfig(cls, id: int, config: config.ModelConfig) -> "CommitteeModels":
    return {
      'MLP'    : MLP,
      'KMeans' : KMeans,
      'KNN'    : KNN,
    }[config.name](id, config)

  def __init__(self, id: int):
    super(CommitteeModels, self).__init__()
    self.id = id
    return

  def forward(self, *args, **kwargs) -> typing.Dict[str, torch.Tensor]:
    raise NotImplementedError("Abstract class.")

  def get_checkpoint_state(self) -> typing.Dict[typing.Any]:
    raise NotImplementedError("Only for non-NN modules")

  def load_checkpoint_state(self) -> typing.Dict[typing.Any]:
    raise NotImplementedError("Only for non-NN modules")

class MLP(CommitteeModels, torch.nn.Module):
  """
  A modular MLP model that supports Linear, Dropout, LayerNorm and activations.
  """
  def __init__(self, id: int, config: config.ModelConfig):
    super(MLP, self).__init__(id)
    self.config = config.layer_config
    self.layers = []

    layers = {
      'Embedding' : torch.nn.Embedding,
      'Linear'    : torch.nn.Linear,
      'Dropout'   : torch.nn.Dropout,
      'LayerNorm' : torch.nn.LayerNorm,
    }
    layers.update(ACT2FN)
    self.layers = torch.nn.ModuleList([layers[layer[0]](**layer[1]) for layer in config])
    return

  def calculate_loss(self,
                     outputs: torch.Tensor,
                     target_ids: torch.Tensor,
                     ) -> torch.Tensor:
    """
    Categorical cross-entropy function.
    """
    ## Calculate categorical label loss.
    loss_fn = torch.nn.CrossEntropyLoss()
    label_loss = loss_fn(outputs.to(torch.float32), target_ids.squeeze(1))

    ## Calculate top-1 accuracy of predictions across batch.
    hits, total = 0, int(outputs.size(0))
    for out, target in zip(torch.argmax(outputs, dim = 1), target_ids):
      if out == target:
        hits += 1
    return label_loss, torch.FloatTensor([hits / total])

  def forward(self,
              input_ids       : torch.Tensor,
              target_ids      : torch.Tensor = None,
              is_sampling     : bool = False
              ) -> torch.Tensor:
    """
    Args:
      input_ids: Input features for training or prediction.
      target_ids: Target tokens to predict during training.
      static_features: List of static input features of respective sample to predict.
      is_sampling: Select between training and sampling method.
    """
    device = input_ids.get_device()
    device = device if device >= 0 else 'cpu'

    out = input_ids
    for layer in self.layers:
      out = layer(out)

    if not is_sampling:
      total_loss, batch_accuracy = self.calculate_loss(out, target_ids)
      return {
        'total_loss'   : total_loss,
        'accuracy'     : batch_accuracy.to(device),
        'output_probs' : out,
        'output_label' : torch.argmax(out)
      }
    else:
      return {
        'output_probs' : out,
        'output_label' : torch.argmax(out, dim = 1),
      }

class KMeans(CommitteeModels):
  """
  Wrapper class to manage, fit and predict KMeans clusters.
  """
  def __init__(self, id: int, config: ModelConfig):
    super(KMeans, self).__init__(id)
    self.config     = config
    self.target_ids = self.config.downstream_task.output_ids
    self.kmeans = sklearn.cluster.KMeans(
      n_clusters = self.config.n_clusters,
      init       = self.config.init,
      n_init     = self.config.n_init,
      max_iter   = self.config.max_iter,
      tol        = self.config.tol,
      algorithm  = self.config.algorithm,
    )
    ## The following two variables are the model's attributes.
    self.classifier  = None
    self.cluster_map = {}
    return

  def __call__(self,
               input_ids   : np.array,
               target_ids  : np.array = None,
               is_sampling : bool = False
               ) -> None:
    if not is_sampling:
      ## Create a map for labels from target ids, and cluster IDS.
      self.cluster_map = {}
      self.classifier  = self.kmeans.fit(input_ids)
      for cluster_id, target_id in zip(self.classifier.labels_, self.target_ids):
        if cluster_id not in self.cluster_map:
          self.cluster_map[cluster_id] = [0] * self.config.num_labels
        self.cluster_map[cluster_id][target_id] += 1
      return {
        'cluster_map'    : self.cluster_map,
        'cluster_labels' : self.classifier.labels_,
      }
    else:
      cluster_labels = self.classifier.predict(input_ids)
      target_labels  = [np.argmax(self.cluster_map[x]) for x in cluster_labels]
      return {
        'cluster_labels'   : cluster_labels,
        'predicted_labels' : target_labels,
      }

  def get_checkpoint_state(self) -> typing.Dict[typing.Any]:
    """
    Return the blob that is to be checkpointed.
    """
    return {
      'kmeans'      : self.classifier,
      'cluster_map' : self.cluster_map,
    }

  def load_checkpoint_state(self, checkpoint_state: typing.Dict[typing.Any]) -> None:
    """
    Load the checkpoints to the class states.
    """
    self.classifier  = checkpoint_state['kmeans']
    self.cluster_map = checkpoint_state['cluster_map']
    return

class KNN(CommitteeModels):
  """
  Wrapper class to manage, fit and predict KNN algorithm.
  """
  def __init__(self, id: int, config: ModelConfig):
    super(KNN, self).__init__(id)
    self.config     = config
    self.knn = sklearn.neighbors.KNeighborsRegressor(
      n_neighbors = self.config.n_neighbors,
      weights     = self.config.weights,
      algorithm   = self.config.algorithm,
      leaf_size   = self.config.leaf_size,
      p           = self.config.p,
      n_jobs      = -1,
    )
    ## The model's attributes
    self.classifier  = None
    return

  def __call__(self,
               input_ids   : np.array,
               target_ids  : np.array = None,
               is_sampling : bool = False,
               ) -> typing.Dict[str, np.array]:
    if not is_sampling:
      self.classifier = self.knn.fit(input_ids, target_ids)
      return {}
    else:
      labels = self.classifier.predict(input_ids)
      return {
        'predicted_labels' : labels
      }

  def get_checkpoint_state(self) -> typing.Dict[typing.Any]:
    """
    Return the blob that is to be checkpointed.
    """
    return {'knn' : self.classifier,}

  def load_checkpoint_state(self, checkpoint_state: typing.Dict[typing.Any]) -> None:
    """
    Load the checkpoints to the class states.
    """
    self.classifier = checkpoint_state['knn']
    return