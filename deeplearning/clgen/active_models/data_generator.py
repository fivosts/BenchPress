"""
Data generators for active learning committee.
"""
import typing
import pathlib

from deeplearning.clgen.util import pytorch
from deeplearning.clgen.util.pytorch import torch

class Dataloader(torch.utils.data.Dataset):
  """
  Abstract torch dataloading class.
  Inherit for this for multiple downstream tasks.
  """
  def __init__(self):
    super(Dataloader, self).__init__()
    return

class GrewePredictiveLoader(Dataloader):
  """
  Specified dataloader for Grewe predictive model.
  """
  def __init__(self):
    super(GrewePredictiveLoader, self).__init__()
    return
