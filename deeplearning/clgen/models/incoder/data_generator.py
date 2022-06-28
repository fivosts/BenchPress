"""This file defines the streaming generators for model training data.

We train models on overlapping one-hot encoded text sequences. For a corpus of
a reasonable size, the full training data may not fit in memory. This modules
provides Python Generator classes for use by a sequential Keras model's
fit_generator() method to stream batches of training data.
"""
import typing
import copy
import sklearn
import pickle
import functools
import numpy as np
import multiprocessing
import math
import tqdm
import pathlib

from deeplearning.clgen.util import pytorch
from deeplearning.clgen.util.pytorch import torch
from deeplearning.clgen.util import distributions
from deeplearning.clgen.util import monitors
from deeplearning.clgen.util import environment
from deeplearning.clgen.util import distrib
from deeplearning.clgen.samplers import samplers
from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen.corpuses import corpuses
from deeplearning.clgen.corpuses import tokenizers
from deeplearning.clgen.features import extractor
from deeplearning.clgen.features import feature_sampler
from deeplearning.clgen.features import active_feed_database
from deeplearning.clgen.features import evaluate_cand_database
from deeplearning.clgen.models import sequence_masking
from deeplearning.clgen.models.torch_bert import datasets
from deeplearning.clgen.models.torch_bert import data_generator as torch_data_generator
from deeplearning.clgen.samplers import sample_observers
from absl import flags
from deeplearning.clgen.util import logging as l

FLAGS = flags.FLAGS

class IncoderDataGenerator(torch_data_generator.torchLMDataGenerator):
  """Data generator subclass designed for Incoder model."""
  @classmethod
  def TrainMaskLMBatchGenerator(cls,
                               corpus                  : corpuses.Corpus,
                               training_opts           : model_pb2.TrainingOptions,
                               cache_path              : pathlib.Path,
                               num_train_steps         : int  = None,
                               pre_train               : bool = False,
                               feature_encoder         : bool                        = False,
                               feature_tokenizer       : tokenizers.FeatureTokenizer = None,
                               feature_sequence_length : int                         = None,
                               ) -> torch_data_generator.MaskLMBatchGenerator:
    """Initializes data generator for training."""
    d = super(IncoderDataGenerator, IncoderDataGenerator()).TrainMaskLMBatchGenerator(
                corpus, training_opts, cache_path, num_train_steps, pre_train,
                feature_encoder, feature_tokenizer, feature_sequence_length,
        )
    return d

  @classmethod
  def SampleMaskLMBatchGenerator(cls,
                                 model_opts              : model_pb2.TrainingOptions,
                                 sampler                 : samplers.Sampler,
                                 tokenizer               : tokenizers.TokenizerBase,
                                 seed                    : int,
                                 sample_batch_size       : int,
                                 max_position_embeddings : int,
                                 cache_path              : pathlib.Path,
                                 corpus                  : corpuses.Corpus = None,
                                 feature_encoder         : bool            = False,
                                 feature_tokenizer       : tokenizers.FeatureTokenizer = None,
                                 feature_sequence_length : int = None,
                                 ) -> torch_data_generator.MaskLMBatchGenerator:
    """Initializes data generator for inference."""
    d = super(IncoderDataGenerator, IncoderDataGenerator()).SampleMaskLMBatchGenerator(
              model_opts, sampler, tokenizer, seed,
              sample_batch_size, max_position_embeddings, cache_path, corpus,
              feature_encoder, feature_tokenizer, feature_sequence_length
        )
    return d

  def __init__(self):
    super(IncoderDataGenerator, self).__init__()
    return

  def initOrGetQueue(self, target_features: typing.Dict[str, float] = None) -> np.array:
    """
    If feed queue is not initialized, initialize it by getting new datapoint.
    Otherwise, don't do anything as feed_queue is already loaded from checkpoint.
    Adds datapoint to InputFeed table of database.

    Returns:
      Starting input feed of sampling.
    """
    raise NotImplementedError
    if not self.feed_queue:
      if FLAGS.start_from_cached and target_features is not None:
        cached_samples = [[x.sample, {':'.join(f.split(':')[:-1]): float(f.split(':')[-1]) for f in x.output_features.split('\n')}, -1] for x in self.active_db.get_data]
        if len(cached_samples) == 0:
          return self.initOrGetQueue()
        else:
          for idx, cs in enumerate(cached_samples):
            cached_samples[idx][-1] = self.feat_sampler.calculate_distance(cs[1])
          sorted_cache_samples = sorted(cached_samples, key = lambda x: x[-1])
          for scs in sorted_cache_samples[:self.sampler.config.sample_corpus.corpus_config.active.active_search_width]:
            tokenized = self.tokenizer.TokenizeString(scs[0])
            w_start_end = self._addStartEndToken(tokenized)
            padded = self._padToMaxPosition(w_start_end)[:self.sampler.sequence_length]
            if padded[0] == self.tokenizer.padToken:
              l.logger().error("Pad token was found again at the beginning of the sequence.")
              l.logger().error(scs[0])
              l.logger().error(tokenized)
              l.logger().error(w_start_end)
              l.logger().error(padded)
            encoded = self._padToMaxPosition(self._addStartEndToken([int(x) for x in tokenized]))[:self.sampler.sequence_length]
            assert encoded[0] != self.tokenizer.padToken, encoded
            self.feed_queue.append(
              ActiveSampleFeed(
                input_feed     = encoded,
                input_features = scs[1],
                input_score    = scs[-1],
                gen_id         = 0,
              )
            )
            self.addToDB(
              active_feed_database.ActiveInput.FromArgs(
                tokenizer      = self.tokenizer, id = self.active_db.input_count,
                input_feed     = encoded,        input_features = scs[1],
              )
            )
      else:
        try:
          cf = next(self.loader).squeeze(0)
        except StopIteration:
          self.loader = iter(self.dataloader)
          cf = next(self.loader).squeeze(0)
        cf = [int(x) for x in cf]
        assert cf[0] != self.tokenizer.padToken, cf
        self.feed_queue.append(
          ActiveSampleFeed(
            input_feed     = cf,
            input_features = extractor.ExtractFeatures(self.tokenizer.ArrayToCode(cf), [self.feat_sampler.feature_space])[self.feat_sampler.feature_space],
            input_score    = math.inf,
            gen_id         = 0,
          )
        )
        self.addToDB(
          active_feed_database.ActiveInput.FromArgs(
            tokenizer      = self.tokenizer, id = self.active_db.input_count,
            input_feed     = cf, input_features = self.feed_queue[-1].input_features,
          )
        )
    l.logger().info("Feed queue input scores: {}".format(', '.join([str(round(c.input_score, 3)) for c in self.feed_queue])))
    return self.feed_queue[0].input_feed