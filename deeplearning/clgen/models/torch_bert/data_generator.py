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
import pickle
import functools
import numpy as np
import pathlib

from deeplearning.clgen.util import pytorch
from deeplearning.clgen.util.pytorch import torch
from deeplearning.clgen.util import distributions
from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen.models import lm_data_generator
from deeplearning.clgen.models import sequence_masking
from deeplearning.clgen.models import online_generator
from deeplearning.clgen.models.torch_bert import datasets
from deeplearning.clgen.features import active_generator
from absl import flags
from eupy.native import logger as l

FLAGS = flags.FLAGS

class torchLMDataGenerator(lm_data_generator.MaskLMDataGenerator):
  """Data generator subclass designed for PyTorch BERT model."""
  @classmethod
  def TrainMaskLMBatchGenerator(cls,
                               corpus: "corpuses.Corpus",
                               training_opts: model_pb2.TrainingOptions,
                               cache_path,
                               num_train_steps: int = None,
                               pre_train: bool = False,
                               ) -> "data_generator.MaskLMBatchGenerator":
    """Initializes data generator for training."""
    d = super(torchLMDataGenerator, torchLMDataGenerator()).TrainMaskLMBatchGenerator(
                corpus, training_opts, cache_path, num_train_steps, pre_train
        )
    d.dataloader = d.train_dataloader()
    return d

  @classmethod
  def SampleMaskLMBatchGenerator(cls,
                                 model_opts,
                                 sampler,
                                 tokenizer,
                                 seed: int,
                                 sample_batch_size: int,
                                 max_position_embeddings: int,
                                 cache_path,
                                 ) -> "data_generator.MaskLMBatchGenerator":
    """Initializes data generator for inference."""
    d = super(torchLMDataGenerator, torchLMDataGenerator()).SampleMaskLMBatchGenerator(
              model_opts, sampler, tokenizer, seed,
              sample_batch_size, max_position_embeddings, cache_path
        )
    # if sampler.is_active:
    #   return active_generator.ActiveSamplingGenerator.FromDataGenerator(d)
    # elif sampler.is_online:
    #   return online_generator.OnlineSamplingGenerator.FromDataGenerator(d)
    # else:
    d.dataloader = d.predict_dataloader()
    return d

  def __init__(self):
    super(torchLMDataGenerator, self).__init__("pt_record")
    self.dataloader = None
    return

  def train_dataloader(self, set_name = 'train_dataset', is_train = True) -> None:
    """
    Pytorch dataloader used for training.
  
    set_name defaults to train_dataset, and that way this function
    this dataloader's function is used for training.

    eval_dataloaders sets set_name to reuse the function for all different sets.
    """
    if self.config.datapoint_time == "pre":
      dataset = datasets.LazyConcatDataset([x for x in self.dataset[set_name]['file']])
      sampler = datasets.LazyRandomSampler(dataset, replacement = False)
    elif self.config.datapoint_time == "online":
      dataset = datasets.OnlineDataset(self, is_train)
      sampler = torch.utils.data.RandomSampler(dataset, replacement = False)
    else:
      raise ValueError(self.config.datapoint_time)

    dataloader = torch.utils.data.dataloader.DataLoader(
      dataset    = dataset,
      batch_size = self.training_opts.batch_size,
      sampler    = (sampler
        if not pytorch.torch_tpu_available or pytorch.torch_xla.xrt_world_size() <= 1
        else torch.utils.data.distributed.DistributedSampler(
          dataset,
          num_replicas = pytorch.torch_xla.xrt_world_size(),
          rank = pytorch.torch_xla.get_ordinal()
        )
      ),
      num_workers = 0,
      drop_last   = False,
    )
    return dataloader

  def eval_dataloaders(self):
    """Pytorch dataloader used for validation."""
    if self.config.datapoint_time == "online":
      yield "Online Corpus", self.train_dataloader(is_train = False)
    else:
      for set_name in self.dataset:
        yield set_name, self.train_dataloader(set_name)

  def predict_dataloader(self):
    """
    Pytorch dataloader used for inference.
    
    isFixedStr == True means there is a fixed sample feed, e.g. 'kernel void [HOLE]'
    Otherwise, a set has been given to provide random samples from it.
    """
    if self.sampler.isFixedStr or self.sampler.is_live:
      input_sample = self.sampler.encoded_start_text
      target_idx   = np.where(np.in1d(input_sample, [self.tokenizer.maskToken, self.tokenizer.holeToken]))[0]
      num_targets  = (np.count_nonzero(input_sample == self.tokenizer.maskToken) + 
                     np.count_nonzero(input_sample == self.tokenizer.holeToken))

      assert np.ndim(input_sample) == 1, "Input samples have to be one-dimensional. {} given.".format(input_sample.shape)
      assert len(target_idx)       != 0, "No target prediction in sample text"

      seen_in_training     = np.zeros([1], dtype = np.int32)
      original_input       = np.full((self.sampler.sequence_length), self.tokenizer.padToken, dtype = np.int64)
      input_ids            = self._padToMaxPosition(input_sample)[:self.sampler.sequence_length]
      input_mask           = np.concatenate([
                                  np.ones(len(input_sample), dtype = np.int64),
                                  np.zeros(len(input_ids) - len(input_sample), dtype = np.int64)
                                ])      
      position_ids         = np.arange(self.sampler.sequence_length, dtype = np.int64)
      mask_labels          = np.full((self.sampler.sequence_length), -100, dtype = np.int64)
      masked_lm_lengths    = np.full((self.sampler.sequence_length), -1, dtype = np.int64)
      next_sentence_labels = np.zeros([1], dtype = np.int32)
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
      sampler = torch.utils.data.RandomSampler(dataset, replacement = False)
    else:
      if self.sampler.is_online:
        dataset = datasets.OnlineDataset(self, False)
        sampler = torch.utils.data.RandomSampler(dataset, replacement = False)
      elif self.sampler.is_active:
        raise NotImplementedError("Integrate active sampler here")
      else:
        path_list = self.configSampleSets()
        dataset = datasets.LazyConcatDataset(
                    [x for x in path_list]
                  )
        sampler = datasets.LazyRandomSampler(dataset, replacement = False)
    dataloader = torch.utils.data.dataloader.DataLoader(
      dataset    = dataset,
      # Model's batch size is divided by sampler's batch size, in order to get
      # multiple generation candidates from a given sample feed, but still
      # efficiently feed big batches to make sampling faster.
      # Example: model batch size 32 and sampler batch size 4.
      # This dataloader will return 8 feeds. Each will be repeated 4 times.
      # 32 sequences will be given to the model.
      batch_size = self.sample_batch_size,
      sampler    = (
            sampler
            if not pytorch.torch_tpu_available or pytorch.torch_xla.xrt_world_size() <= 1
            else torch.utils.data.distributed.DistributedSampler(
                  dataset, 
                  num_replicas = pytorch.torch_xla.xrt_world_size(), 
                  rank = pytorch.torch_xla.get_ordinal()
                 )
            ),
      num_workers = 0,
      drop_last   = False,
      )
    return dataloader

  def toTensorFormat(self,
                     datapoint: typing.Dict[str, np.array]
                     ) -> typing.Dict[str, torch.Tensor]:
    """Formats a datapoint mapped in generic numpy arrays to Torch tensors."""
    return {k: torch.from_numpy(v).unsqueeze(0) for (k, v) in datapoint.items()}

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
                                      self.tokenizer.tokensToString(instance['original_input'], ignore_token = self.tokenizer.padToken),
                                      self.tokenizer.tokensToString(instance['input_ids'],      ignore_token = self.tokenizer.padToken),
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
