"""
Data generator wrapper used for active learning sampling.
"""
import subprocess
import functools
import pickle
import typing
import numpy as np

from deeplearning.clgen.models import lm_data_generator
from deeplearning.clgen.models import online_generator
from deeplearning.clgen.models import sequence_masking
from deeplearning.clgen.features import extractor
from deeplearning.clgen.features import feature_sampler
from deeplearning.clgen.features import active_feed_database
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
    # List of model inference batch outputs.
    sample_outputs   : typing.List[typing.List[np.array]]
    # List of output_features for model batch inference outputs.
    output_features  : typing.List[typing.List[typing.Dict[str, float]]]
    # Binary quality flag of sample batch outputs wrt target features.
    good_samples     : typing.List[typing.List[bool]]

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
      feature, stderr = extractor.kernel_features(seed_src)
      input_features = extractor.StrToDictFeatures(feature)
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

  def EvaluateFeatures(self,
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
    if len(self.feed_stack) == 0:
      raise ValueError("Feed stack is empty. Cannot pop element.")

    features, gd = [], []
    for sample in samples:
      feature, stderr = extractor.kernel_features(self.atomizer.ArrayToCode(sample))

      if " error: " in stderr:
        features.append(None)
        gd.append(False)
      else:
        # This line below is your requirement.
        # This is going to become more complex.
        bigger = feature_sampler.is_kernel_smaller(
          self.feed_stack[-1].input_features, extractor.StrToDictFeatures(feature)
        )
        gd.append(bigger)
        features.append(feature)

      entry = active_feed_database.ActiveFeed.FromArgs(
        atomizer         = self.atomizer,
        id               = self.active_db.count(),
        input_feed       = self.feed_stack[-1].input_feed,
        input_features   = self.feed_stack[-1].input_features,
        masked_input_ids = self.feed_stack[-1].masked_input_ids[-1],
        hole_instances   = self.feed_stack[-1].hole_instances[-1],
        sample           = sample,
        output_features  = features[-1],
        sample_quality   = gd[-1],
      )
      self.addToDB(entry)

    self.feed_stack[-1].good_samples.append(gd)
    self.feed_stack[-1].sample_outputs.append(samples)
    self.feed_stack[-1].output_features.append(features)

    if any(self.feed_stack[-1].good_samples[-1]):
      return self.feed_stack[-1].sample_outputs[-1], True
    else:
      input_ids, masked_idxs = self.masking_func(self.feed_stack[-1].input_feed)
      # TODO do sth with hole_lengths and masked_idxs
      self.feed_stack[-1].masked_input_ids.append(input_ids)
      self.feed_stack[-1].hole_instances.append(masked_idxs)
      return self.data_generator.toTensorFormat(input_ids), False

  def addToDB(self, active_feed: active_feed_database.ActiveFeed) -> None:
    """If not exists, add current sample state to database"""
    with self.active_db.Session(commit = True) as session:
      exists = session.query(ActiveFeed).filter(sha256 == active_feed.sha256).scalar() is not None
      if not exists:
        session.add(active_feed)
    return
