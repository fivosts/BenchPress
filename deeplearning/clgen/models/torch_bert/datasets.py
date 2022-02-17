"""This file defines the streaming generators for model training data.

We train models on overlapping one-hot encoded text sequences. For a corpus of
a reasonable size, the full training data may not fit in memory. This modules
provides Python Generator classes for use by a sequential Keras model's
fit_generator() method to stream batches of training data.
"""
import typing
import pickle
import functools
import json
import numpy as np
import pathlib
import glob

from deeplearning.clgen.util import pytorch
from deeplearning.clgen.util.pytorch import torch
from deeplearning.clgen.util import distributions
from deeplearning.clgen.util import monitors
from deeplearning.clgen.util import environment
from deeplearning.clgen.models import sequence_masking
from deeplearning.clgen.models import lm_data_generator
from absl import flags
from deeplearning.clgen.util import logging as l

FLAGS = flags.FLAGS

class OnlineDataset(torch.utils.data.Dataset):
  r"""Online pre-processing dataset of raw corpus.

  This dataset holds path to raw corpus and yields
  pre-processed instances on the fly.

  Arguments:
    dataset (path): Path for raw dataset
    func (callable): Function called to pre-process sequence.
  """
  def __init__(self, dg: lm_data_generator.MaskLMDataGenerator, is_train: bool):
    super(OnlineDataset, self).__init__()
    full_dataset         = self.load_data(dg.cache.path / "{}corpus.pkl".format("pre_" if dg.pre_train else ""))
    """
    TODO you've better change is_train check to something more generic.
    """
    if is_train:
      self.dataset       = full_dataset[:int(len(full_dataset) * (1 - (dg.config.validation_split / 100)))]
    else:
      self.dataset       = full_dataset[int(len(full_dataset) * (1 - (dg.config.validation_split / 100))):]
    self.cache_path      = dg.cache.path
    self.size            = len(self.dataset)
    self.cur_step        = 0
    self.steps_per_epoch = dg.steps_per_epoch * dg.training_opts.batch_size

    self.hlen_monitor = None
    if is_train:
      if (self.cache_path / "{}hole_length_mon{}.pkl".format("pre_" if dg.pre_train else "", "_{}".format(environment.WORLD_RANK) if environment.WORLD_SIZE > 1 else "")).exists():
        with open(self.cache_path / "{}hole_length_mon{}.pkl".format("pre_" if dg.pre_train else "", "_{}".format(environment.WORLD_RANK) if environment.WORLD_SIZE > 1 else ""), 'rb') as infile:
          self.hlen_monitor = pickle.load(infile)
      else:
        self.hlen_monitor = monitors.NormalizedFrequencyMonitor(self.cache_path, "{}online_hole_length{}".format("pre_" if dg.pre_train else "", "_{}".format(environment.WORLD_RANK) if environment.WORLD_SIZE > 1 else ""))

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
                                    masked_lm_prob  = dg.training_opts.masked_lm_prob,
                                    distribution    = distribution,
                                    tokenizer       = dg.tokenizer,
        )
    elif dg.config.HasField("mask_seq"):
      distribution = distributions.Distribution.FromHoleConfig(
        dg.config.mask_seq, dg.cache.path, "mask_seq_length_online"
      )
      self.func = functools.partial(sequence_masking.HoleSequenceSeqMasks,
                                    train_set       = is_train,
                                    max_predictions = dg.training_opts.max_predictions_per_seq,
                                    masked_lm_prob  = dg.training_opts.masked_lm_prob,
                                    distribution    = distribution,
                                    tokenizer       = dg.tokenizer,
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

    if self.hlen_monitor:
      self.hlen_monitor.register([x for x in k['masked_lm_lengths'] if x >= 0])
      if self.cur_step % self.steps_per_epoch == 0:
        self.hlen_monitor.plot()
        with open(self.cache_path / "hole_length_mon{}.pkl".format("_{}".format(environment.WORLD_RANK) if environment.WORLD_SIZE > 1 else ""), 'wb') as outf:
          pickle.dump(self.hlen_monitor, outf)

    # raise NotImplementedError("Fix a) init state of rngen)
    return k

  def load_data(self, dataset: pathlib.Path) -> typing.List[np.array]:
    if dataset.exists():
      with open(dataset, 'rb') as infile:
        return pickle.load(infile)
    else:
      raise FileNotFoundError(dataset)

class LazyOnlineDataset(torch.utils.data.Dataset):
  r"""Dataset as a concatenation of multiple datasets.

  This class is useful to assemble different existing datasets
  and instantiate them lazily, to avoid loading them all in
  memory at the same time/

  Arguments:
    datasets (sequence): List of paths for datasets to be concatenated
  """

  @staticmethod
  def cumsum(sequence: typing.List[pathlib.Path], length_cache: pathlib.Path):
    lts, r, s = None, [], 0 # Cached lengths list, cumulative lengths, current max length.

    ## If lengths cache exists, just load the dictionary.
    if length_cache.exists():
      with open(length_cache, 'r') as inf:
        lts = json.load(inf)

    ## Iterate every dataset chunk, and fix the cumulative length distribution.
    for e in sequence:
      if lts:
        lt = lts[pathlib.Path(e).name]
      else:
        with open(e, 'rb') as infile:
          length = len(pickle.load(infile))
          lt = length
      assert lt > 0, "Dataset {} is empty".format(e)
      r.append(lt + s)
      s += lt

    ## If lengths cache had not been created, fix it now.
    if not lts and environment.WORLD_RANK == 0:
      lts = {}
      s = 0
      for e, rx in zip(sequence, r):
        lts[pathlib.Path(e).name] = rx - s
        s = rx
      with open(length_cache, 'w') as outf:
        json.dump(lts, outf)
    return r

  @property
  def num_datasets(self):
    return len(self.datasets)

  def __init__(self, dg: lm_data_generator.MaskLMDataGenerator, is_train: bool):
    super(LazyOnlineDataset, self).__init__()

    self.datasets = glob.glob(str(dg.cache.path / "{}corpus_*.pkl".format("pre_" if dg.pre_train else "")))
    self.cumulative_sizes = self.cumsum(self.datasets, dg.cache.path / "pre_lengths_cache.json")

    self.curr_dset_idx = None
    self.dataset       = None
    self.is_train      = is_train
    """
    TODO you've better change is_train check to something more generic.
    """
    self.vfactor = lambda l: int(l * (1 - (dg.config.validation_split / 100)))
    self.cache_path      = dg.cache.path
    self.cur_step        = 0
    self.steps_per_epoch = dg.steps_per_epoch * dg.training_opts.batch_size

    self.hlen_monitor    = None
    if is_train:
      if (self.cache_path / "{}hole_length_mon{}.pkl".format("pre_" if dg.pre_train else "", "_{}".format(environment.WORLD_RANK) if environment.WORLD_SIZE > 1 else "")).exists():
        with open(self.cache_path / "{}hole_length_mon{}.pkl".format("pre_" if dg.pre_train else "", "_{}".format(environment.WORLD_RANK) if environment.WORLD_SIZE > 1 else ""), 'rb') as infile:
          self.hlen_monitor = pickle.load(infile)
      else:
        self.hlen_monitor = monitors.NormalizedFrequencyMonitor(self.cache_path, "{}online_hole_length{}".format("pre_" if dg.pre_train else "", "_{}".format(environment.WORLD_RANK) if environment.WORLD_SIZE > 1 else ""))

    """
    TODO, add custom config just like in lm_data_generator
    for val sets / sample sets etc.
    """
    self.tokenizer = dg.tokenizer
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
                                    masked_lm_prob  = dg.training_opts.masked_lm_prob,
                                    distribution    = distribution,
                                    tokenizer       = dg.tokenizer,
        )
    return

  def __len__(self):
    return self.cumulative_sizes[-1]

  def __getitem__(self, idx):

    self.cur_step += 1
    if idx < 0:
      if -idx > len(self):
        raise ValueError("absolute value of index should not exceed dataset length")
      idx = len(self) + idx
    import bisect
    dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
    
    if self.curr_dset_idx != dataset_idx:
      self.curr_dset_idx = dataset_idx
      with open(self.datasets[dataset_idx], 'rb') as infile:
        self.dataset = pickle.load(infile)

    if dataset_idx == 0:
      sample_idx = idx
    else:
      sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
    k = self.func(self.dataset[sample_idx])

    if self.hlen_monitor:
      self.hlen_monitor.register([x for x in k['masked_lm_lengths'] if x >= 0])
      if self.cur_step % self.steps_per_epoch == 0:
        self.hlen_monitor.plot()
        with open(self.cache_path / "hole_length_mon{}.pkl".format("_{}".format(environment.WORLD_RANK) if environment.WORLD_SIZE > 1 else "", 'wb') as outf:
          pickle.dump(self.hlen_monitor, outf)
    return k

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

  def __init__(self, data_source, replacement = False, num_samples = None, generator = None):
    self.data_source  = data_source
    self.replacement  = replacement
    self._num_samples = num_samples
    self.generator    = generator
    self.distributed  = True if environment.WORLD_SIZE > 1 else False
    self.dataset_idx  = self.__datasetIdx_iter__

    self.epoch = None
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
    if isinstance(self.data_source, LazyConcatDataset) or isinstance(self.data_source, LazyOnlineDataset):
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

    lb, ub = self.data_source.cumulative_sizes[dataset_idx - 1] if dataset_idx else 0, self.data_source.cumulative_sizes[dataset_idx]
    if isinstance(self.data_source, LazyOnlineDataset):
      clen = ub - lb
      if self.data_source.is_train:
        bounds = (lb, lb + self.data_source.vfactor(clen))
      else:
        bounds = (lb + self.data_source.vfactor(clen), ub)
    else:
      bounds = (lb, ub)

    if self.distributed:
      self.generator = torch.Generator()
      self.generator.manual_seed(self.epoch)

    if self.replacement:
      if self._num_samples is None:
        size = bounds[1] - bounds[0]
      else:
        size = self._num_samples // self.num_datasets
      rand_tensor = torch.randint(low = bounds[0], high = bounds[1], size = (size,), generator = self.generator).tolist()
    else:
      rand_tensor = [x + bounds[0] for x in torch.randperm(bounds[1] - bounds[0], generator = self.generator).tolist()]

    if self.distributed:
      rounded_total = (len(rand_tensor) // environment.WORLD_SIZE) * environment.WORLD_SIZE
      rand_tensor   = rand_tensor[environment.WORLD_RANK:rounded_total:environment.WORLD_SIZE]
    return iter(rand_tensor)

  def __len__(self):
    return self.num_samples

  def set_epoch(self, epoch: int) -> None:
    """
    Sets epoch for deterministic runs across DDP.
    """
    self.epoch = epoch
    return
