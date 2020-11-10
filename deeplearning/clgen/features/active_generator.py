import subprocess
import functools
import pickle
import typing
import numpy as np

from deeplearning.clgen.models import lm_data_generator
from deeplearning.clgen.models import online_generator
from deeplearning.clgen.models import sequence_masking
from deeplearning.clgen.features import extractor
from deeplearning.clgen.util import distributions

from eupy.native import logger as l

class ActiveSamplingGenerator(online_generator.OnlineSamplingGenerator):
  """
  Data generation object that performs active learning
  on sampling process.
  
  This does not implement active learning based training.
  A sample feed instance is fed to the model in different
  ways to find the closest match based on a feature vector.
  """

  class ActiveSampleFeed(typing.NamedTuple):
    """
    Representation of a single instance containing
    original feed, features and the samples the model responded with.
    """
    # An array of original input
    input_feed       : np.array
    # The feature space of the original input
    input_features   : typing.Dict[str, float]
    """
    All fields below are lists of instances.
    Based on the same input_feed, each instance
    represents a single model inference step.
    The indices of the lists below correspond to
    the iteration of the active sampler for the given
    input feed.
    """
    # List of masked model input feeds.
    masked_input_ids : typing.List[np.array]
    # List of hole instances for masked input.
    hole_instances   : typing.List[typing.List[sequence_masking.MaskedLmInstance]]
    # List of model inference outputs.
    sample_outputs   : typing.List[np.array]
    # List of output_features for model inference outputs.
    output_features  : typing.List[typing.Dict[str, float]]
    # Binary quality flag of sample outputs wrt target features.
    good_samples     : typing.List[bool]

  @classmethod
  def FromDataGenerator(cls,
                        generator: lm_data_generator.MaskLMDataGenerator,
                        ) -> "active_generator.OnlineSamplingGenerator":
    """Initializes data generator for active sampling."""
    d = ActiveSamplingGenerator(generator)
    d.dataloader = d.active_dataloader()
    return d

  def __init__(self,
               generator: lm_data_generator.MaskLMDataGenerator
               ):
    super(ActiveSamplingGenerator, self).__init__(generator)
    # Active sampling attributes.
    self.feed_stack = []
    return

  def active_dataloader(self) -> typing.Union[
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
      seed_src = self.atomizer.ArrayToCode(seed)
      input_features = extractor.StrToDictFeatures(extractor.kernel_features(seed_src))
      input_feed, masked_idxs = self.masking_func(seed)
      # TODO do sth with hole_lengths and masked_idxs
      self.feed_stack.append(
        ActiveSamplingGenerator.ActiveSampleFeed(
          input_feed       = seed,
          input_features   = input_features,
          masked_input_ids = [input_feed['input_ids']],
          hole_instances   = [masked_idxs],
          sample_outputs   = [],
          output_features  = [],
          good_samples     = [],
        )
      )
      yield self.data_generator.toTensorFormat(input_feed)

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
    features = [
      extractor.StrToDictFeatures(extractor.kernel_features(self.atomizer.ArrayToCode(s)))
      for s in samples
    ]
    # Update outputs of most recent ActiveSampleFeed
    self.feed_stack[-1].sample_outputs  += samples
    self.feed_stack[-1].output_features += features

    # for sample, feature in zip(samples, features):
    #   # This line below is your requirement.
    #   # This is going to become more complex.
    #   bigger = feature_sampler.is_it_bigger(
    #     self.feed_stack[-1].input_features, feature
    #   )
    #   self.feed_stack[-1].append(bigger)
    #   if feature is not crap:
    #     keep(sample)

    # if there is sample close to target feature:
    #   return good sample, True
    # else:
    #   latest_seed = self.feed_stack[-1]
    #   sample_feed, hole_lengths, masked_idxs = self.masking_func(latest_seed)
    #   # TODO do sth with the hole_lengths and masked_idxs
    #   return sample_feed, False
