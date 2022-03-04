"""
Here all the committee members are defined.
"""
import math
import typing

import numpy as np

from deeplearning.clgen.active_models.committee import config
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
  def FromConfig(cls, config: config.ModelConfig) -> "CommitteeModels":
    return {
      'MLP': MLP,
    }[config.name](config)

  def __init__(self, id: int):
    self.id = id
    return

class MLP(CommitteeModels):
  """
  A modular MLP model that supports Linear, Dropout, LayerNorm and activations.
  """
  def __init__(self, id: int, config: config.MLPConfig):
    super(self, MLP).__init__(id)
    self.config = config
    self.layers = []

    layers = {
      'Linear'    : torch.nn.Linear,
      'Dropout'   : torch.nn.Dropout,
      'LayerNorm' : torch.nn.LayerNorm,
    }
    layers.update(ACT2FN)
    self.layers = torch.nn.ModuleList([layers[name](**params) for name, params in config.layers.items()])
    return

  def forward(self, inp: torch.Tensor) -> torch.Tensor:
    out = inp
    for layer in self.layers:
      out = layer(out)
    return out
