import subprocess
import functools
import pickle
import typing

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
    self.feed_stack = None
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
      input_feed, hole_lengths, masked_idxs = self.masking_func(seed)
      # TODO do sth with hole_lengths and masked_idxs
      self.feed_stack.append(seed)
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
    features = [extractor.kernel_features(s) for s in samples]

    for sample, feature in zip(samples, features):
      if feature is not crap:
        keep(sample)

    if there is sample close to target feature:
      return good sample, True
    else:
      latest_seed = self.feed_stack[-1]
      sample_feed, hole_lengths, masked_idxs = self.masking_func(latest_seed)
      # TODO do sth with the hole_lengths and masked_idxs
      return sample_feed, False
