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
  def __init__(self, dataset: typing.List[typing.Tuple[typing.List[int], typing.List[int]]]):
    super(Dataloader, self).__init__()
    return
