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