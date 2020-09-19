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
    d.train_dataloader()
    return d

  @classmethod
  def SampleMaskLMBatchGenerator(cls,
                                sampler,
                                atomizer,
                                seed: int,
                                max_position_embeddings: int,
                                cache_path,
                                ) -> "data_generator.MaskLMBatchGenerator":
    """Initializes data generator for inference."""
    d = super(torchLMDataGenerator, torchLMDataGenerator()).SampleMaskLMBatchGenerator(
              sampler, atomizer, seed, max_position_embeddings, cache_path
        )
    d.predict_dataloader()
    return d

  def train_dataloader(self) -> None:
    """Pytorch dataloader that assembles all dataset files into a single-mapped dataset."""
    dataset = torch.utils.data.ConcatDataset(
                [torch.load(x) for x in self.dataset['train_dataset']['file']]
              )
    self.dataloader = torch.utils.data.dataloader.DataLoader(
      dataset    = dataset,
      batch_size = self.training_opts.batch_size,
      sampler    = (
            torch.utils.data.RandomSampler(dataset, replacement = False)
            if not pytorch.torch_tpu_available or pytorch.torch_xla.xrt_world_size() <= 1
            else torch.utils.data.distributed.DistributedSampler(
                  dataset, 
                  num_replicas = pytorch.torch_xla.xrt_world_size(), 
                  rank = pytorch.torch_xla.get_ordinal()
                 )
            ),
      num_workers = os.cpu_count(),
      drop_last   = True,
    )
    return

  def eval_dataloaders(self):
    for set_name in self.dataset:
      dataset = torch.utils.data.ConcatDataset(
                  [torch.load(x) for x in self.dataset[set_name]['file']]
                )
      dataloader = torch.utils.data.dataloader.DataLoader(
        dataset    = dataset,
        batch_size = self.training_opts.batch_size,
        sampler    = (
              torch.utils.data.RandomSampler(dataset, replacement = False)
              if not pytorch.torch_tpu_available or pytorch.torch_xla.xrt_world_size() <= 1
              else torch.utils.data.distributed.DistributedSampler(
                    dataset, 
                    num_replicas = pytorch.torch_xla.xrt_world_size(), 
                    rank = pytorch.torch_xla.get_ordinal()
                   )
              ),
        num_workers = os.cpu_count(),
        drop_last   = True,
      )
      yield set_name, dataloader

  def predict_dataloader(self):

    if self.sampler.isFixedStr:
      input_sample = self.sampler.encoded_start_text
      assert np.ndim(input_sample) == 1, "Input samples have to be one-dimensional. {} given.".format(input_sample.shape)

      target_idx = np.where(np.in1d(input_sample, [self.atomizer.maskToken, self.atomizer.holeToken]))[0]
      assert len(target_idx) != 0, "No target prediction in sample text"
      num_targets = (np.count_nonzero(input_sample == self.atomizer.maskToken) + 
                     np.count_nonzero(input_sample == self.atomizer.holeToken))

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
      raise ValuError("Check here that the metrics are correct.")
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
      dataset = torch.utils.data.ConcatDataset(
                  [torch.load(x) for x in path_list]
                )
    self.dataloader = torch.utils.data.dataloader.DataLoader(
      dataset    = dataset,
      batch_size = 1,
      sampler    = (
            torch.utils.data.RandomSampler(dataset, replacement = False)
            if not pytorch.torch_tpu_available or pytorch.torch_xla.xrt_world_size() <= 1
            else torch.utils.data.distributed.DistributedSampler(
                  dataset, 
                  num_replicas = pytorch.torch_xla.xrt_world_size(), 
                  rank = pytorch.torch_xla.get_ordinal()
                 )
            ),
      num_workers = os.cpu_count(),
      drop_last   = True,
      )
    return

  def _saveCorpusRecord(self, masked_corpus: typing.Dict) -> None:
    """Converts corpus nparrays to tf Features and stores corpus to TfRecord"""

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


