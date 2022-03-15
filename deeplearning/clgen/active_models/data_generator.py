"""
Data generators for active learning committee.
"""
import typing
import pathlib
import numpy as np

from deeplearning.clgen.util import pytorch
from deeplearning.clgen.util.pytorch import torch

class ListTrainDataloader(torch.utils.data.Dataset):
  """
  Modular dataloading class for downstream tasks.
  """
  def __init__(self, 
               dataset : typing.List[typing.Tuple[typing.List, typing.List]],
               lazy    : bool = False,
               ):
    super(ListTrainDataloader, self).__init__()
    ## The dataset here should be a list, and each entry
    ## must be a tuple containing the input and the target vector.
    if len(dataset) <= 0 and not lazy:
      raise ValueError("Predictive model dataset seems empty.")
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

  def get_batched_dataset(self) -> typing.Dict[str, np.array]:
    """
    Batch the whole dataset by keys and return it.
    """
    return {
      'input_ids'  : np.as_array([x['input_ids'].numpy() for x in self.dataset]),
      'target_ids' : np.as_array([x['target_ids'].numpy() for x in self.dataset]),
    }

  def __len__(self) -> int:
    return len(self.dataset)

  def __getitem__(self, idx: int) -> typing.Dict[str, torch.Tensor]:

    if idx < 0:
      if -idx > len(self):
        raise ValueError("absolute value of index should not exceed dataset length")
      idx = len(self) + idx

    return self.dataset[idx]

  def __add__(self, dl: 'ListTrainDataloader') -> 'ListTrainDataloader':
    ret = ListTrainDataloader([], lazy = True)
    ret.dataset = self.dataset
    if dl:
      ret.dataset += dl.dataset
    return ret

class DictPredictionDataloader(torch.utils.data.Dataset):
  """
  Dataloading class that takes datapoint dictionary.
  """
  def __init__(self, 
               dataset: typing.List[typing.Dict[str, typing.List]],
               lazy    : bool = False,
               ):
    super(DictPredictionDataloader, self).__init__()
    if len(dataset) <= 0 and not lazy:
      raise ValuError("Sample dataset is empty.")
    self.compute_dataset(dataset)
    return

  def compute_dataset(self,
                      dataset: typing.List[typing.Dict[str, typing.List]]
                      ) -> None:
    """
    Batch the whole dataset by keys and return it.
    """
    self.dataset = []
    for dp in dataset:
      self.dataset.append(
        {
          'static_features'  : torch.FloatTensor(dp['static_features']),
          'runtime_features' : torch.LongTensor(dp['runtime_features']),
          'input_ids'        : torch.FloatTensor(dp['input_ids']),
        }
      )
    return

  def get_batched_dataset(self) -> typing.Dict[str, np.array]:
    """
    Batch the whole dataset by keys and return it.
    """
    return {
      'static_features'  : np.as_array([x['static_features'].numpy() for x in self.dataset]),
      'runtime_features' : np.as_array([x['runtime_features'].numpy() for x in self.dataset]),
      'input_ids'        : np.as_array([x['input_ids'].numpy() for x in self.dataset]),
    }

  def __len__(self) -> int:
    return len(self.dataset)

  def __getitem__(self, idx: int) -> typing.Dict[str, torch.Tensor]:

    if idx < 0:
      if -idx > len(self):
        raise ValueError("absolute value of index should not exceed dataset length")
      idx = len(self) + idx

    return self.dataset[idx]

  def __add__(self, dl: 'DictPredictionDataloader') -> 'DictPredictionDataloader':
    ret = DictPredictionDataloader([], lazy = True)
    ret.dataset = self.dataset
    if dl:
      ret.dataset += dl.dataset
    return ret
