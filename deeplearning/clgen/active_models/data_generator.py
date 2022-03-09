"""
Data generators for active learning committee.
"""
import typing
import pathlib

from deeplearning.clgen.util import pytorch
from deeplearning.clgen.util.pytorch import torch

class ListTrainDataloader(torch.utils.data.Dataset):
  """
  Modular dataloading class for downstream tasks.
  """
  def __init__(self, dataset: typing.List[typing.Tuple[typing.List, typing.List]]):
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
    self.dataset = []
    for dp in dataset:
      inp, targ = dp
      self.dataset.append(
        {
          'input_ids' : torch.FloatTensor(inp),
          'target_ids': torch.LongTensor(targ),
        }
      )
    return

  def __len__(self) -> int:
    return len(self.dataset)

  def __getitem__(self, idx: int) -> typing.Dict[str, torch.Tensor]:

    if idx < 0:
      if -idx > len(self):
        raise ValueError("absolute value of index should not exceed dataset length")
      idx = len(self) + idx

    return self.dataset[idx]

class DictPredictionDataloader(torch.utils.data.Dataset):
  """
  Dataloading class that takes datapoint dictionary.
  """
  def __init__(self, dataset: typing.):
    super(Dataloader, self).__init__()
    if len(dataset) <= 0:
      raise ValuError("Sample dataset is empty.")
    self.compute_dataset(dataset)
    return

  def compute_dataset(self,
                      dataset: typing.List[typing.Dict[str, typing.Union[typing.Dict, typing.List]]]
                      ) -> None:
    self.dataset = []
    for dp in dataset:
      self.dataset.append(
        {
          'static_features' : torch.FloatTensor(dp['static_features']),
          'input_ids'       : torch.FloatTensor(dp['input_ids']),
          'predictions'     : torch.LongTensor([-1, -1]),
        }
      )
    return

  def __len__(self) -> int:
    return len(self.dataset)

  def __getitem__(self, idx: int) -> typing.Dict[str, torch.Tensor]:

    if idx < 0:
      if -idx > len(self):
        raise ValueError("absolute value of index should not exceed dataset length")
      idx = len(self) + idx

    return self.dataset[idx]
