import subprocess
import functools
import pickle
import typing

from deeplearning.clgen.models import lm_data_generator
from deeplearning.clgen.models import sequence_masking
from deeplearning.clgen.util import distributions

from eupy.native import logger as l

class ActiveSamplingGenerator(lm_data_generator.MaskLMDataGenerator):
  """
  Data generation object that performs active learning
  on sampling process.
  
  This does not implement active learning based training.
  A sample feed instance is fed to the model in different
  ways to find the closest match based on a feature vector.
  """

  @classmethod
  def FromDataGenerator(cls,
                        generator: lm_data_generator.MaskLMDataGenerator,
                        ) -> "active_generator.ActiveSamplingGenerator":
    """Initializes data generator for active sampling."""
    d = ActiveSamplingGenerator()

    d.data_generator = generator
    d.sampler        = d.data_generator.sampler
    d.atomizer       = d.data_generator.atomizer

    d.configSamplingParams()
    d.configSampleCorpus()

    d.dataloader = d.sample_dataloader()
    return d

  def __init__(self):
    self.data_generator = None

    # Wrapped data generator attributes
    self.sampler        = None
    self.atomizer       = None
    self.sample_corpus  = None

    # Inherent attributes
    self.distribution   = None
    self.masking_func   = None
    self.dataloader     = None
    return

  def configSampleCorpus(self) -> None:
    """
    Configure sampling corpus container to iterate upon.
    """
    if self.sampler.isFixedStr:
      if (self.atomizer.maskToken in self.sampler.encoded_start_text or
          self.atomizer.holeToken in self.sampler.encoded_start_text):
        raise ValueError("{} targets found in active sampler start text. This is wrong. Active sampler masks a sequence on the fly...")
      self.sample_corpus = [self.sampler.encoded_start_text]
    else:
      self.sample_corpus = self.data_generator.createCorpus(self.sampler.corpus_directory)
    return

  def configSamplingParams(self) -> None:
    """
    Configure masking function used by active sampler.
    """
    class SampleTrainingOpts(typing.NamedTuple):
      max_predictions_per_seq: int
      masked_lm_prob: float

    corpus_config = self.sampler.config.sample_corpus.corpus_config
    sampling_opts = SampleTrainingOpts(
      self.data_generator.training_opts.max_predictions_per_seq, corpus_config.masked_lm_prob
    )

    if corpus_config.HasField("hole"):
      self.distribution = distributions.Distribution.FromHoleConfig(
        corpus_config.hole, self.sampler.corpus_directory, "sample_corpus"
      )
      self.masking_func = functools.partial(sequence_masking.HoleSequence,
                            train_set            = False,
                            max_predictions      = corpus_config.max_predictions_per_seq,
                            pickled_distribution = pickle.dumps(self.distribution),
                            pickled_atomizer     = pickle.dumps(self.atomizer),
                            training_opts        = sampling_opts,
                            is_torch             = self.data_generator.is_torch,
                          )
    elif corpus_config.HasField("mask"):
      self.masking_func = functools.partial(sequence_masking.MaskSequence,
                            train_set          = False,
                            max_predictions    = corpus_config.max_predictions_per_seq,
                            config             = corpus_config,
                            pickled_atomizer   = pickle.dumps(self.atomizer),
                            training_opts      = sampling_opts,
                            rngen              = self.data_generator.rngen,
                            is_torch           = self.data_generator.is_torch,
                          )
    return

  def sample_dataloader(self):
    """
    Configurate data container that will be iterated for sampling.
    """
    for seed in self.sample_corpus:
      sample_feed, hole_lengths, masked_idxs = self.masking_func(seed)
      yield self.data_generator.toTensorFormat(sample_feed)
