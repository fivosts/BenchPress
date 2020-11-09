import subprocess
import functools
import pickle
import typing

from deeplearning.clgen.models import lm_data_generator
from deeplearning.clgen.models import sequence_masking
from deeplearning.clgen.features import extractor
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
    self.sampler       = None
    self.atomizer      = None
    self.sample_corpus = None

    # Inherent attributes
    self.distribution  = None
    self.masking_func  = None
    self.dataloader    = None
    self.feed_stack    = None
    return

  def sample_dataloader(self) -> typing.Union[
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
    for seed in self.sample_corpus:
      sample_feed, hole_lengths, masked_idxs = self.masking_func(seed)
      # TODO do sth with hole_lengths and masked_idxs
      self.feed_stack.append(seed)
      yield self.data_generator.toTensorFormat(sample_feed)

  def EvaluateFromFeatures(self,
                           samples: np.array,
                           ) -> typing.Union[
                                  typing.Dict[str, typing.TypeVar("Tensor")],
                                  typing.NamedTuple
                                ]:
    """
    Reads model sampling output and evaluates against active target features.
    If the sample output is not good enough based on the features,
    active sampler reconstructs the same sample feed and asks again for prediction.
    """
    if self.feed_stack is None:
      raise ValueError("Cannot evaluate from features when no sample has been asked.")
    if len(self.feed_stack) == 0:
      raise ValueError("Feed stack is empty. Cannot pop element.")

    # You might also need to check if they compile.
    features = [extractor.kernel_features(s) for s in samples]

    for sample, feature in zip(samples, features):
      if feature is not crap:
        keep(sample)

    if there is sample close to target feature:
      return good sample, True # Done
    else:
      latest_seed = self.feed_stack[-1]
      sample_feed, hole_lengths, masked_idxs = self.masking_func(latest_seed)
      # TODO do sth with the hole_lengths and masked_idxs
      return sample_feed, False, # Not Done

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
