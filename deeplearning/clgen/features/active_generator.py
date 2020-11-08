import subprocess
import typing

from deeplearning.clgen.models import lm_data_generator
from deeplearning.clgen.models import sequence_masking

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
                        # generator: lm_data_generator.MaskLMDataGenerator,
                        ) -> "active_generator.ActiveSamplingGenerator":
    """Initializes data generator for active sampling."""
    d = ActiveSamplingGenerator()
    # d.data_generator = generator
    d.configSampleCorpus()
    d.configMaskingFunc(d.config)
    d.dataloader = d.sample_dataloader()
    return d

  def __init__(self):
    # self.data_generator = None
    self.sample_corpus = None
    self.dataloader = None
    self.masking_func = None
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
      self.sample_corpus = self.createCorpus(self.sampler.corpus_directory)
    return

  def configMaskingFunc(self, config) -> None:
    """
    Configure masking function used by active sampler.
    """

    ## TODO sampler config if exists or training opts config if it doesn't.
    if config.HasField("hole"):
      self.masking_func = functools.partial(sequence_masking.HoleSequence,
                            train_set            = False,
                            max_predictions      = config.max_predictions_per_seq,
                            pickled_distribution = pickle.dumps(distribution),
                            pickled_atomizer     = pickle.dumps(self.atomizer),
                            training_opts        = self.training_opts,
                            is_torch             = self.is_torch,
                          )
    elif config.HasField("mask"):
      self.masking_func = functools.partial(self.mask_func._maskSequence,
                            train_set          = False,
                            max_predictions    = config.max_predictions_per_seq,
                            config             = config,
                            pickled_atomizer   = pickle.dumps(self.atomizer),
                            training_opts      = self.training_opts,
                            rngen              = self.rngen,
                            is_torch           = self.is_torch,
                          )
    return

  def sample_dataloader(self):
    """
    Configurate data container that will be iterated for sampling.
    """
    for seed in self.sample_corpus:
      yield self.masking_func(seed)