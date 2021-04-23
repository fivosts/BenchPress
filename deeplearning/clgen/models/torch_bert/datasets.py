"""This file defines the streaming generators for model training data.

We train models on overlapping one-hot encoded text sequences. For a corpus of
a reasonable size, the full training data may not fit in memory. This modules
provides Python Generator classes for use by a sequential Keras model's
fit_generator() method to stream batches of training data.
"""
import typing
import pickle
import functools
import numpy as np
import pathlib

from deeplearning.clgen.util import pytorch
from deeplearning.clgen.util.pytorch import torch
from deeplearning.clgen.util import distributions
from deeplearning.clgen.models import sequence_masking
from absl import flags
from eupy.native import logger as l

FLAGS = flags.FLAGS

class OnlineDataset(torch.utils.data.Dataset):
  r"""Online pre-processing dataset of raw corpus.

  This dataset holds path to raw corpus and yields
  pre-processed instances on the fly.

  Arguments:
    dataset (path): Path for raw dataset
    func (callable): Function called to pre-process sequence.
  """
  def __init__(self, dg: torchLMDataGenerator, is_train: bool):
    super(OnlineDataset, self).__init__()
    self.dataset         = self.load_data(dg.cache.path / "corpus.pkl")
    self.cache_path      = dg.cache.path
    self.size            = len(self.dataset)
    self.cur_step        = 0
    self.steps_per_epoch = dg.steps_per_epoch
    if is_train:
      self.hlen_monitor  = monitors.NormalizedFrequencyMonitor(self.cache_path, "online_hole_length")
    """
    TODO, add custom config just like in lm_data_generator
    for val sets / sample sets etc.
    """

    if dg.config.HasField("mask"):
      self.func = functools.partial(sequence_masking.MaskSequence,
                                    train_set         = is_train,
                                    max_predictions   = dg.training_opts.max_predictions_per_seq,
                                    pickled_tokenizer = dg.tokenizer,
                                    training_opts     = dg.training_opts,
                                    is_torch          = True,
                                    config            = dg.config,
      )
    elif dg.config.HasField("hole"):
      distribution = distributions.Distribution.FromHoleConfig(
        dg.config.hole, dg.cache.path, "hole_length_online"
      )
      self.func = functools.partial(sequence_masking.HoleSequence,
                                    train_set       = is_train,
                                    max_predictions = dg.training_opts.max_predictions_per_seq,
                                    distribution    = distribution,
                                    tokenizer       = dg.tokenizer,
                                    training_opts   = dg.training_opts,
        )
    return

  def __len__(self):
    return self.size

  def __getitem__(self, idx):

    self.cur_step += 1
    if idx < 0:
      if -idx > len(self):
        raise ValueError("absolute value of index should not exceed dataset length")
      idx = len(self) + idx
    k = self.func(self.dataset[idx])

    if is_train:
      self.hlen_monitor.register(k['masked_lm_lengths'])
      if self.cur_step % self.steps_per_epoch == 0:
        self.hlen_monitor.plot()
        with open(self.cache_path / "hole_length_mon.pkl", 'wb') as outf:
          pickle.dump(self.hlen_monitor, outf)

    # raise NotImplementedError("Fix a) init state of rngen)
    return k

  def load_data(self, dataset: pathlib.Path) -> typing.List[np.array]:
    if dataset.exists():
      with open(dataset, 'rb') as infile:
        return pickle.load(infile)

class LazyConcatDataset(torch.utils.data.Dataset):
  r"""Dataset as a concatenation of multiple datasets.

  This class is useful to assemble different existing datasets
  and instantiate them lazily, to avoid loading them all in
  memory at the same time/

  Arguments:
    datasets (sequence): List of paths for datasets to be concatenated
  """

  @staticmethod
  def cumsum(sequence: typing.List[pathlib.Path]):
    r, s = [], 0
    for e in sequence:
      lt = len(torch.load(e))
      assert lt > 0, "Dataset {} is empty".format(e)
      r.append(lt + s)
      s += lt
    return r

  @property
  def num_datasets(self):
    return len(self.datasets)

  def __init__(self, datasets: typing.List[pathlib.Path]):
    super(LazyConcatDataset, self).__init__()
    assert len(datasets) > 0, 'Empty list of datasets provided.'
    self.datasets = datasets
    self.cumulative_sizes = self.cumsum(self.datasets)

    self.curr_dset_idx = None
    self.dataset       = None

  def __len__(self):
    return self.cumulative_sizes[-1]

  def __getitem__(self, idx):

    import bisect
    if idx < 0:
      if -idx > len(self):
        raise ValueError("absolute value of index should not exceed dataset length")
      idx = len(self) + idx
    dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
    
    if self.curr_dset_idx != dataset_idx:
      self.curr_dset_idx = dataset_idx
      self.dataset = torch.load(self.datasets[dataset_idx])

    if dataset_idx == 0:
      sample_idx = idx
    else:
      sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
    return self.dataset[sample_idx]

class LazyRandomSampler(torch.utils.data.Sampler):
  r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
  If with replacement, then user can specify :attr:`num_samples` to draw.

  Arguments:
    data_source (Dataset): dataset to sample from
    replacement (bool): samples are drawn with replacement if ``True``, default=``False``
    num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
      is supposed to be specified only when `replacement` is ``True``.
    generator (Generator): Generator used in sampling.
  """

  def __init__(self, data_source, replacement=False, num_samples=None, generator=None):
    self.data_source = data_source
    self.replacement = replacement
    self._num_samples = num_samples
    self.generator = generator
    self.dataset_idx = self.__datasetIdx_iter__
    if not isinstance(self.replacement, bool):
      raise TypeError("replacement should be a boolean value, but got "
                      "replacement={}".format(self.replacement))

    if self._num_samples is not None and not replacement:
      raise ValueError("With replacement=False, num_samples should not be specified, "
                       "since a random permute will be performed.")

    if not isinstance(self.num_samples, int) or self.num_samples <= 0:
      raise ValueError("num_samples should be a positive integer "
                       "value, but got num_samples={}".format(self.num_samples))

  @property
  def num_samples(self):
    # dataset size might change at runtime
    if self._num_samples is None:
      return len(self.data_source)
    return self._num_samples

  @property
  def num_datasets(self):
    if isinstance(self.data_source, LazyConcatDataset):
      return self.data_source.num_datasets
    else:
      return 1

  @property
  def __datasetIdx_iter__(self):
    dataset_idx = torch.randperm(self.num_datasets, generator = self.generator).tolist()
    self.dataset_tensor = iter(dataset_idx)
    return self.dataset_tensor

  def __iter__(self):
    try:
      dataset_idx = next(self.dataset_tensor)
    except StopIteration:
      dataset_idx = next(self.__datasetIdx_iter__)
    bounds = (self.data_source.cumulative_sizes[dataset_idx - 1] if dataset_idx else 0, self.data_source.cumulative_sizes[dataset_idx])

    if self.replacement:
      if self._num_samples is None:
        size = bounds[1] - bounds[0]
      else:
        size = self._num_samples // self.num_datasets
      rand_tensor = torch.randint(low = bounds[0], high = bounds[1], size = (size,), generator = self.generator).tolist()
    else:
      rand_tensor = [x + bounds[0] for x in torch.randperm(bounds[1] - bounds[0], generator = self.generator).tolist()]
    return iter(rand_tensor)

  def __len__(self):
    return self.num_samples
