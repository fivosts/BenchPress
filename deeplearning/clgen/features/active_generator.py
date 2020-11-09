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
    d.configSampleParams(d.config)
    d.dataloader = d.sample_dataloader()
    return d

  def __init__(self):
    # self.data_generator = None
    self.sample_corpus = None
    self.dataloader    = None
    self.masking_func  = None
    self.distribution  = None
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

  def configSampleParams(self, config) -> None:
    """
    Configure masking function used by active sampler.
    """
    class SampleTrainingOpts(typing.NamedTuple):
      max_predictions_per_seq: int
      masked_lm_prob: float

    corpus_config = self.sampler.config.sample_corpus.corpus_config
    sampling_opts = SampleTrainingOpts(
      self.training_opts.max_predictions_per_seq, corpus_config.masked_lm_prob
    )

    if corpus_config.data_generator.HasField("hole"):
      self.distribution = distributions.Distribution.FromHoleConfig(
        corpus_config.data_generator.hole, path, "sample_corpus"
      )
      self.masking_func = functools.partial(sequence_masking.HoleSequence,
                            train_set            = False,
                            max_predictions      = corpus_config.max_predictions_per_seq,
                            pickled_distribution = pickle.dumps(self.distribution),
                            pickled_atomizer     = pickle.dumps(self.atomizer),
                            training_opts        = sampling_opts,
                            is_torch             = self.is_torch,
                          )
    elif corpus_config.data_generator.HasField("mask"):
      self.masking_func = functools.partial(self.mask_func._maskSequence,
                            train_set          = False,
                            max_predictions    = corpus_config.max_predictions_per_seq,
                            config             = corpus_config,
                            pickled_atomizer   = pickle.dumps(self.atomizer),
                            training_opts      = sampling_opts,
                            rngen              = self.rngen,
                            is_torch           = self.is_torch,
                          )
    return

  def sample_dataloader(self):
    """
    Configurate data container that will be iterated for sampling.
    """
    for seed in self.sample_corpus:
      sample_feed, hole_lengths, masked_idxs = self.masking_func(seed)
      yield sample_feed
