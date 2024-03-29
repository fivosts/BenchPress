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
Data generators for active learning committee.
"""
import typing
import copy
import pathlib
import numpy as np

from deeplearning.benchpress.util import pytorch
from deeplearning.benchpress.util.pytorch import torch
from deeplearning.benchpress.util import logging as l

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
      l.logger().warn("Active learning committee dataset is empty. Make sure this is expected behavior.")
    self.compute_dataset(dataset)
    return

  def compute_dataset(self, dataset) -> None:
    """
    Convert list dataset to torch tensors.
    """
    self.dataset = []
    for dp in dataset:
      if len(dp) == 2:
        inp, targ = dp
        self.dataset.append(
          {
            'input_ids' : torch.FloatTensor(inp),
            'target_ids': torch.LongTensor(targ),
          }
        )
      elif len(dp) == 3:
        inp, targ, idx = dp
        self.dataset.append(
          {
            'input_ids' : torch.FloatTensor(inp),
            'target_ids': torch.LongTensor(targ),
            'idx'       : torch.LongTensor(idx),
          }
        )
    return

  def get_batched_dataset(self) -> typing.Dict[str, np.array]:
    """
    Batch the whole dataset by keys and return it.
    """
    return {
      'input_ids'  : np.asarray([x['input_ids'].numpy() for x in self.dataset]),
      'target_ids' : np.asarray([x['target_ids'].numpy() for x in self.dataset]),
    }

  def get_random_subset(self, num: int, seed: int = None) -> 'ListTrainDataloader':
    """
    Get a sample of num random samples from dataset.
    """
    ret  = ListTrainDataloader([], lazy = True)
    num  = min(num, len(self.dataset))
    if seed:
      generator = torch.Generator()
      generator.manual_seed(seed)
    else:
      generator = None
    rand = set(torch.randperm(len(self.dataset), generator = None).tolist()[:num])
    ret.dataset = [x for idx, x in enumerate(self.dataset) if idx in rand]
    return ret

  def get_sliced_subset(self, l: int = None, r: int = None) -> 'ListTrainDataloader':
    """
    Implement slice operation of current List Dataset.
    """
    ret = ListTrainDataloader([], lazy = True)
    if l is None:
      l = 0
    if r is None:
      r = len(self.dataset)
    ret.dataset = self.dataset[l:r]
    return ret

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
    ret.dataset = copy.copy(self.dataset)
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
    for idx, dp in enumerate(dataset):
      self.dataset.append(
        {
          'idx'              : torch.LongTensor([idx]),
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
      'idx'              : np.asarray([x['idx'].numpy() for x in self.dataset]),
      'static_features'  : np.asarray([x['static_features'].numpy() for x in self.dataset]),
      'runtime_features' : np.asarray([x['runtime_features'].numpy() for x in self.dataset]),
      'input_ids'        : np.asarray([x['input_ids'].numpy() for x in self.dataset]),
    }

  def get_random_subset(self, num: int, seed: int = None) -> 'DictPredictionDataloader':
    """
    Get a sample of num random samples from dataset.
    """
    ret  = DictPredictionDataloader([], lazy = True)
    num  = min(num, len(self.dataset))
    if seed:
      generator = torch.Generator()
      generator.manual_seed(seed)
    else:
      generator = None
    rand = set(torch.randperm(len(self.dataset), generator = generator).tolist()[:num])
    ret.dataset = [x for idx, x in enumerate(self.dataset) if idx in rand]
    return ret

  def get_sliced_subset(self, l: int = None, r: int = None) -> 'DictPredictionDataloader':
    """
    Implement slice operation of current List Dataset.
    """
    ret = DictPredictionDataloader([], lazy = True)
    if l is None:
      l = 0
    if r is None:
      r = len(self.dataset)
    ret.dataset = self.dataset[l:r]
    return ret

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
    ret.dataset = copy.copy(self.dataset)
    if dl:
      ret.dataset += dl.dataset
    return ret
