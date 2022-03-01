"""
Here all the committee members are defined.
"""
import math
import typing

import numpy as np

from deeplearning.clgen.models.committee import config
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

class Committee(torch.nn.Module):
  """
  Abstract representation of model committee.
  """
  @classmethod
  def FromConfig(cls, config: config.CommitteeConfig) -> "Committee":
    return

class MLP(torch.nn.Module):
  def __init__(self, config: config.CommitteeConfig):

    return

  def forward(self, inp: torch.Tensor) -> torch.Tensor:
    return