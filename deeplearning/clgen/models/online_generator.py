import subprocess
import functools
import pickle
import typing

from deeplearning.clgen.models import lm_data_generator
from deeplearning.clgen.models import sequence_masking
from deeplearning.clgen.features import extractor
from deeplearning.clgen.util import distributions

from eupy.native import logger as l

class OnlineSamplingGenerator(object):
  """
  Data generation object that performs  masking of original
  corpus on the fly.
  """

  @classmethod
  def FromDataGenerator(cls,
                        generator: lm_data_generator.MaskLMDataGenerator,
                        ) -> "online_generator.OnlineSamplingGenerator":
    """Initializes data generator for online sampling."""
    d = OnlineSamplingGenerator()

    d.data_generator = generator
    d.sampler        = d.data_generator.sampler
    d.atomizer       = d.data_generator.atomizer

    d.configSamplingParams()
    d.configSampleCorpus()

    d.dataloader = d.online_dataloader()
    return d

  @classmethod
  def FromDataGenerator(cls,
                        generator: lm_data_generator.MaskLMDataGenerator,
                        ) -> "online_generator.OnlineSamplingGenerator":
    """Initializes data generator for online sampling."""
    d = OnlineSamplingGenerator(generator)
    d.dataloader = d.online_dataloader()
    return d

  def __init__(self,
               generator: lm_data_generator.MaskLMDataGenerator
               ):
    self.data_generator = generator

    # Wrapped data generator attributes
    self.sampler       = self.data_generator.sampler
    self.atomizer      = self.data_generator.atomizer

    # Inherent attributes
    self.online_corpus = None
    self.distribution  = None
    self.masking_func  = None
    self.dataloader    = None

    self.configSamplingParams()
    self.configSampleCorpus()
    return

  def online_dataloader(self) -> typing.Union[
                                   typing.Dict[str, typing.TypeVar("Tensor")],
                                   typing.NamedTuple
                                 ]:
    """
    Configurate data container that will be iterated for sampling.
    Generates data points. 
    In TF, NamedTuples from str to np.array are generated.
    In torch, Dict[str, np.array] instances are generated.
    masking_func output goes through TensorFormat to convert np arrays to relevant tensors.
    """
    for seed in self.online_corpus:
      input_feed, masked_idxs = self.masking_func(seed)
      # TODO do sth with hole_lengths and masked_idxs
      yield self.data_generator.toTensorFormat(input_feed)

  def configSampleCorpus(self) -> None:
    """
    Configure sampling corpus container to iterate upon.
    """
    if self.sampler.isFixedStr:
      if (self.atomizer.maskToken in self.sampler.encoded_start_text or
          self.atomizer.holeToken in self.sampler.encoded_start_text):
        raise ValueError("Targets found in {} start text. This is wrong. Active sampler masks a sequence on the fly...".format(type(self).__name__))
      self.online_corpus = [self.sampler.encoded_start_text]
    else:
      self.online_corpus = self.data_generator.createCorpus(self.sampler.corpus_directory)
    return

  def configSamplingParams(self) -> None:
    """
    Configure masking function used by online sampler.
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
