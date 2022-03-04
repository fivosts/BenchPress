"""
Data generators for active learning committee.
"""
import typing
import pathlib

from deeplearning.clgen.util import pytorch
from deeplearning.clgen.util.pytorch import torch

class Dataloader(torch.utils.data.Dataset):
  """
  Modular dataloading class for downstream tasks.
  """
  def __init__(self, dataset):
    super(Dataloader, self).__init__()
    ## The dataset here should be a list, and each entry
    ## must be a tuple containing the input and the target vector.
    if len(dataset) <= 0:
      raise ValuError("Predictive model dataset seems empty.")
    self.compute_dataset(dataset)
    return

  def compute_dataset(self, dataset) -> None:
    """
    Convert list dataset to torch tensors.
    """
    for dp in dataset:
      inp, targ = dp
      self.dataset.append(
        {
          'input_ids' : torch.FloatTensor(inp),
          'target_ids': torch.LongTensor(targ),
        }
      )
    return
