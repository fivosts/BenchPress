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
  "gelu": activations.gelu,
  "relu": torch.nn.functional.relu,
  "swish": activations.swish,
  "gelu_new": activations.gelu_new,
  "mish": mish
}

class CommitteeModels(torch.nn.Module):
  """
  Abstract representation of model committee.
  """
  @classmethod
  def FromConfig(cls, id: int, config: config.ModelConfig) -> "CommitteeModels":
    print(config)
    return {
      'MLP': MLP,
    }[config.name](id, config.layer_config)

  def __init__(self, id: int):
    super(CommitteeModels, self).__init__()
    self.id = id
    return

class MLP(CommitteeModels):
  """
  A modular MLP model that supports Linear, Dropout, LayerNorm and activations.
  """
  def __init__(self, id: int, config: typing.List):
    super(MLP, self).__init__(id)
    self.config = config
    self.layers = []

    layers = {
      'Linear'    : torch.nn.Linear,
      'Dropout'   : torch.nn.Dropout,
      'LayerNorm' : torch.nn.LayerNorm,
    }
    layers.update(ACT2FN)
    self.layers = torch.nn.ModuleList([layers[layer[0]](**layer[1]) for layer in config])
    return

  def forward(self,
              inp: torch.Tensor,
              target: torch.Tensor,
              is_sampling: bool = False
              ) -> torch.Tensor:
    out = inp
    for layer in self.layers:
      out = layer(out)

    if not is_sampling:
      total_loss = self.calculate_loss(out, target)
      return {
        'total_loss'   : total_loss,
        'output_label' : torch.argmax(out)
      }
    else:
      return {
        'output_label' : torch.argmax(out)
      }
