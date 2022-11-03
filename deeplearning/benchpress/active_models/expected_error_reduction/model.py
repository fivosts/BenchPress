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
Here all the committee members are defined.
"""
import math
import sys
import typing
import numpy as np
from sklearn import cluster as sklearn_cluster
from sklearn import neighbors as sklearn_neighbors

from deeplearning.benchpress.active_models.expected_error_reduction import config
from deeplearning.benchpress.models.torch_bert import activations
from deeplearning.benchpress.util import pytorch
from deeplearning.benchpress.util.pytorch import torch

from deeplearning.benchpress.util import logging as l

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
    self.layers = torch.nn.ModuleList([layers[layer[0]](**layer[1]) for layer in self.config])
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
              input_ids   : torch.Tensor,
              target_ids  : torch.Tensor = None,
              is_sampling : bool = False
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
