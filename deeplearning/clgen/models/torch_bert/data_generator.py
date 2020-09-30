"""This file defines the streaming generators for model training data.

We train models on overlapping one-hot encoded text sequences. For a corpus of
a reasonable size, the full training data may not fit in memory. This modules
provides Python Generator classes for use by a sequential Keras model's
fit_generator() method to stream batches of training data.
"""
import os
import typing
import glob
import humanize
import numpy as np
import pathlib

from deeplearning.clgen.util import pytorch
from deeplearning.clgen.util.pytorch import torch
from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen.models import lm_data_generator
from absl import flags
from eupy.native import logger as l

FLAGS = flags.FLAGS

class torchLMDataGenerator(lm_data_generator.MaskLMDataGenerator):
  def __init__(self):
    super(torchLMDataGenerator, self).__init__("pt_record")
    self.dataloader = None
    return

  @classmethod
  def TrainMaskLMBatchGenerator(cls,
                               corpus: "corpuses.Corpus",
                               training_opts: model_pb2.TrainingOptions,
                               cache_path,
                               ) -> "data_generator.MaskLMBatchGenerator":
    """Initializes data generator for training."""
    d = super(torchLMDataGenerator, torchLMDataGenerator()).TrainMaskLMBatchGenerator(
                corpus, training_opts, cache_path
        )
    d.dataloader = d.train_dataloader()
    return d

  @classmethod
  def SampleMaskLMBatchGenerator(cls,
                                model_opts,
                                sampler,
                                atomizer,
                                seed: int,
                                max_position_embeddings: int,
                                cache_path,
                                ) -> "data_generator.MaskLMBatchGenerator":
    """Initializes data generator for inference."""
    d = super(torchLMDataGenerator, torchLMDataGenerator()).SampleMaskLMBatchGenerator(
              model_opts, sampler, atomizer, seed, max_position_embeddings, cache_path
        )
    d.dataloader = d.predict_dataloader()
    return d

  def train_dataloader(self, set_name = 'train_dataset') -> None:
    """
    Pytorch dataloader used for training.
  
    set_name defaults to train_dataset, and that way this function
    this dataloader's function is used for training.

    eval_dataloaders sets set_name to reuse the function for all different sets.
    """
    dataset = LazyConcatDataset(
                [x for x in self.dataset[set_name]['file']]
              )
    dataloader = torch.utils.data.dataloader.DataLoader(
      dataset    = dataset,
      batch_size = self.training_opts.batch_size,
      sampler    = (
            LazyRandomSampler(dataset, replacement = False)
            if not pytorch.torch_tpu_available or pytorch.torch_xla.xrt_world_size() <= 1
            else torch.utils.data.distributed.DistributedSampler(
                  dataset, 
                  num_replicas = pytorch.torch_xla.xrt_world_size(),
                  rank = pytorch.torch_xla.get_ordinal()
                 )
            ),
      num_workers = 1,
      drop_last   = True,
    )
    return dataloader

  def eval_dataloaders(self):
    """Pytorch dataloader used for validation."""
    for set_name in self.dataset:
      yield set_name, self.train_dataloader(set_name)

  def predict_dataloader(self):
    """
    Pytorch dataloader used for inference.
    
    isFixedStr == True means there is a fixed sample feed, e.g. 'kernel void [HOLE]'
    Otherwise, a set has been given to provide random samples from it.
    """
    if self.sampler.isFixedStr:

      input_sample = self.sampler.encoded_start_text
      target_idx   = np.where(np.in1d(input_sample, [self.atomizer.maskToken, self.atomizer.holeToken]))[0]
      num_targets  = (np.count_nonzero(input_sample == self.atomizer.maskToken) + 
                     np.count_nonzero(input_sample == self.atomizer.holeToken))

      assert np.ndim(input_sample) == 1, "Input samples have to be one-dimensional. {} given.".format(input_sample.shape)
      assert len(target_idx)       != 0, "No target prediction in sample text"

      seen_in_training     = 0
      original_input       = np.full((self.sampler.sequence_length), 0, dtype = np.int64)
      input_ids            = self._padToMaxPosition(input_sample)[:self.sampler.sequence_length]
      input_mask           = np.concatenate([
                                  np.ones(len(input_sample), dtype = np.int64),
                                  np.zeros(len(input_ids) - len(input_sample), dtype = np.int64)
                                ])      
      position_ids         = np.arange(self.sampler.sequence_length, dtype = np.int64)
      mask_labels          = np.full((self.sampler.sequence_length), -100, dtype = np.int64)
      masked_lm_lengths    = np.full((self.sampler.sequence_length), -1, dtype = np.int64)
      next_sentence_labels = 0
      raise NotImplementedError("Check here that the metrics are correct.")
      sample_element = {
        'seen_in_training'    : seen_in_training,
        'original_input'      : original_input,
        'input_ids'           : input_ids,
        'input_mask'          : input_mask,
        'position_ids'        : position_ids,
        'mask_labels'         : mask_labels,
        'masked_lm_lengths'   : masked_lm_lengths,
        'next_sentence_labels': next_sentence_labels,
      }
      dataset = [{k: torch.from_numpy(v) for (k, v) in sample_element.items()}]
    else:
      path_list = self.configSampleSets()
      dataset = LazyConcatDataset(
                  [x for x in path_list]
                )
    dataloader = torch.utils.data.dataloader.DataLoader(
      dataset    = dataset,
      # You are wondering why batch_size == 1 and not sampler.batch_size:
      # By design, an inference batch contains the same feed.
      # If dataloader's batch_size > 1, a batch will contain different feeds and so, is set to 1.
      # Model will ask for a batch (of 1) and then replicate it.
      batch_size = 1,
      sampler    = (
            LazyRandomSampler(dataset, replacement = False)
            if not pytorch.torch_tpu_available or pytorch.torch_xla.xrt_world_size() <= 1
            else torch.utils.data.distributed.DistributedSampler(
                  dataset, 
                  num_replicas = pytorch.torch_xla.xrt_world_size(), 
                  rank = pytorch.torch_xla.get_ordinal()
                 )
            ),
      num_workers = 1,
      drop_last   = True,
      )
    return dataloader

  def _saveCorpusRecord(self, masked_corpus: typing.Dict) -> None:
    """Converts corpus nparrays to torch tensors and stores corpus to pt_record"""

    torch.save(
      [{k: torch.from_numpy(v) for (k, v) in inst.items()} for inst in masked_corpus['corpus']],
      masked_corpus['file']
    )
    if FLAGS.write_text_dataset:
      with open(masked_corpus['txt'], 'w') as file_writer:
        for instance in masked_corpus['corpus']:
          file_writer.write("'seen_in_training': {}\n'original_input': {}\n'input_ids': {}\n'input_mask': {}\n'position_ids': {}\n'mask_labels': {}\n'masked_lm_lengths': {}\n'next_sentence_labels': {}\n\n"
                              .format((True if instance['seen_in_training'] == 1 else False),
                                      self.atomizer.DeatomizeIndices(instance['original_input'], ignore_token = self.atomizer.padToken),
                                      self.atomizer.DeatomizeIndices(instance['input_ids'], ignore_token = self.atomizer.padToken),
                                      instance['input_mask'],
                                      instance['position_ids'],
                                      instance['mask_labels'],
                                      instance['masked_lm_lengths'],
                                      instance['next_sentence_labels']
                                    )
                              )
    l.getLogger().info("Wrote {} instances ({} batches of {} datapoints) to {}"
                 .format(len(masked_corpus['corpus']), self.steps_per_epoch, self.training_opts.batch_size, masked_corpus['file']))
    return

class LazyConcatDataset(torch.utils.data.Dataset):
  r"""Dataset as a concatenation of multiple datasets.

  This class is useful to assemble different existing datasets
  and instantiate them lazily, to avoid loading them all in
  memory at the same time/

  Arguments:
    datasets (sequence): List of paths for datasets to be concatenated
  """

  @staticmethod
  def cumsum(sequence):
    r, s = [], 0
    for e in sequence:
      l = len(torch.load(e))
      r.append(l + s)
      s += l
    return r

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
      l.getLogger().info("Loading dataset: {}, {}".format(self.curr_dset_idx, self.datasets[dataset_idx]))
      self.dataset = torch.load(self.datasets[dataset_idx])

    if dataset_idx == 0:
      sample_idx = idx
    else:
      sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
    # self.monitor.plot()
    return self.dataset[sample_idx]

  @property
  def cummulative_sizes(self):
    warnings.warn("cummulative_sizes attribute is renamed to "
                  "cumulative_sizes", DeprecationWarning, stacklevel=2)
    return self.cumulative_sizes

  @property
  def num_datasets(self):
    return len(self.datasets)

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

  def __iter__(self):
    num_datasets     = self.data_source.num_datasets
    cumulative_sizes = self.data_source.cumulative_sizes
    l.getLogger().warn(num_datasets)
    l.getLogger().warn(cumulative_sizes)
    ds_tensors, curr_len = [], 0
    for idx in range(num_datasets):
      
      bounds  = (curr_len, cumulative_sizes[idx] - 1)
      curr_len = cumulative_sizes[idx]

      if self.replacement:
        if self._num_samples is None:
          size = 1 + bounds[1] - bounds[0]
        else:
          size = self._num_samples // num_datasets
        ds_tensors.append(torch.randint(low = bounds[0], high = bounds[1], size = (size,), generator = self.generator).tolist())
      else:
        ds_tensors.append([x + bounds[0] for x in torch.randperm(bounds[1] - bounds[0], generator = self.generator).tolist()])

    shuffled_idx = torch.randperm(num_datasets, generator = self.generator).tolist()
    rand_tensor = []
    for idx in shuffled_idx:
      rand_tensor += ds_tensors[idx]

    l.getLogger().error(rand_tensor)

    return iter(rand_tensor)

  def __len__(self):
    return self.num_samples
