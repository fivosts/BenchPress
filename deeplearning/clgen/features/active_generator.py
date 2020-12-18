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
from deeplearning.clgen.preprocessors import opencl
from deeplearning.clgen.util import distributions

from absl import flags
from eupy.native import logger as l

FLAGS = flags.FLAGS

flags.DEFINE_integer(
  "active_limit_per_feed",
  150,
  "Set limit on sample attempts per input_feed. [Default: 50]. Set to 0 for infinite."
)

flags.DEFINE_integer(
  "active_search_depth",
  20,
  "Set the maximum sampling generation depth that active sampler can reach. [Default: 10]."
)

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
    Representation of an active learning input to the model.
    """
    # An array of original input
    input_feed       : np.array
    # The feature space of the original input
    input_features   : typing.Dict[str, float]
    # Full model input
    input_blob       : typing.Dict[str, np.array]
    # List of masked model input feeds.
    masked_input_ids : np.array
    # List of hole instances for masked input.
    hole_instances   : typing.List[sequence_masking.MaskedLmInstance]
    # Depth increases when a valid inference sample is fed back as an input.
    gen_id           : int

  class ActiveSample(typing.NamedTuple):
    """
    Representation of an active learning sample.
    """
    # ActiveSampleFeed instance of model input
    sample_feed : typing.TypeVar("ActiveSamplingGenerator.ActiveSampleFeed")
    # Model prediction
    sample      : np.array
    # Output features of sample
    features    : typing.Dict[str, float]
    # Score of sample based on active learning search.
    score       : typing.Union[bool, float]

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
    self.active_db  = active_feed_database.ActiveFeedDatabase(
      url = "sqlite:///{}".format(self.data_generator.sampler.corpus_directory / "active_feeds.db")
    )
    self.feed_queue          = []
    self.step_candidates     = set()
    self.total_candidates    = []
    self.num_current_samples = 0 # How many samples has a specific feed delivered.
    self.active_dataset      = ActiveDataset(self.online_corpus)
    self.feat_sampler        = feature_sampler.EuclideanSampler()
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
    while True:
      """Model will ask with next(). As long as it asks, this loop will bring."""
      seed = next(self.active_dataset)
      input_feed, masked_idxs = self.masking_func(seed)
      # TODO do sth with hole_lengths and masked_idxs
      self.feed_queue.append(
        ActiveSamplingGenerator.ActiveSampleFeed(
          input_feed       = seed,
          input_features   = extractor.DictKernelFeatures(self.atomizer.ArrayToCode(seed)),
          input_blob       = input_feed,
          masked_input_ids = input_feed['input_ids'],
          hole_instances   = masked_idxs,
          gen_id           = 0,
        )
      )
      yield self.data_generator.toTensorFormat(input_feed)

  def EvaluateFeatures(self,
                       samples        : np.array,
                       sample_indices : np.array,
                       ) -> typing.Tuple[
                            typing.Dict[str, typing.TypeVar("Tensor")],
                            np.array,
                            bool
                            ]:
    """
    Reads model sampling output and evaluates against active target features.
    If the sample output is not good enough based on the features,
    active sampler reconstructs the same sample feed and asks again for prediction.
    """
    if len(self.feed_queue) == 0:
      raise ValueError("Feed stack is empty. Cannot pop element.")

    current_feed = self.feed_queue.pop(0)

    # If more candidates are needed for that specific sample,
    # store the feeds and ask more samples.
    for sample, indices in zip(samples, sample_indices):
      features = extractor.DictKernelFeatures(self.atomizer.ArrayToCode(current_feed))
      if features:
        self.step_candidates.add(
          ActiveSamplingGenerator.ActiveSample(
            sample_feed    = current_feed,
            sample         = sample,
            sample_indices = indices,
            features       = features,
            score          = None,
          )
        )

    # If all samples have syntax errors and have no features, skip to next iteration.
    if not self.step_candidates:
      return {}, [], True

    self.num_current_samples += len(samples)
    if self.num_current_samples < FLAGS.active_limit_per_feed:
      input_feed, masked_idxs = self.masking_func(current_feed.input_feed)
      self.feed_queue.insert(0,
        ActiveSamplingGenerator.ActiveSampleFeed(
          input_feed       = current_feed.input_feed,
          input_features   = current_feed.input_features,
          input_blob       = input_feed,
          masked_input_ids = input_feed['input_ids'],
          hole_instances   = masked_idxs,
          gen_id           = current_feed.gen_id,
        )
      )
      # If that specific input hasn't yet gathered all active samples,
      # send it back and ask for more.
      return self.data_generator.toTensorFormat(input_feed), [], False
    else:
      # For a given input feed, you got all sample candidates, so that's it.
      self.num_current_samples = 0

    # This function returns all candidates that succeed the threshold
    # TODO, is the threshold matching a job for feat sampler or active generator ?
    # Maybe the second.
    self.total_candidates += self.feat_sampler.sample_from_set(self.step_candidates)

    for candidate in self.total_candidates:
      # 1) Store them in the database
      try:
        _ = opencl.Compile(self.atomizer.ArrayToCode(sample))
        compile_status = True
      except ValueError:
        compile_status = False
      self.addToDB(
        active_feed_database.ActiveFeed.FromArgs(
          atomizer         = self.atomizer,
          id               = self.active_db.count,
          input_feed       = candidate.sample_feed.input_feed,
          input_features   = candidate.sample_feed.input_features,
          masked_input_ids = candidate.sample_feed.masked_input_ids,
          hole_instances   = candidate.sample_feed.hole_instances,
          sample           = candidate.sample,
          output_features  = candidate.features,
          sample_quality   = candidate.score,
          compile_status   = compile_status,
          generation_id    = candidate.sample_feed.gen_id,
        )
      )
      # 2) Push them back to the input, if not maximum depth is not reached.
      # self.active_dataset.add_active_feed(candidate.sample) # I strongly disagree with this after all.
      if 1 + candidate.sample_feed.gen_id <= FLAGS.active_search_depth:
        input_feed, masked_idxs = self.masking_func(candidate.sample)
        self.feed_queue.append(
          ActiveSamplingGenerator.ActiveSampleFeed(
            input_feed       = candidate.sample,
            input_features   = candidate.features,
            input_blob       = input_feed,
            masked_input_ids = input_feed['input_ids'],
            hole_instances   = masked_idxs,
            gen_id           = 1 + candidate.sample_feed.gen_id,
          )
        )
    # 3) Re-initialize all variables
    self.step_candidates = set() # Input feed is going to change, so reset this sample counter.

    if self.feed_queue:
      # Send the first good candidate back as an input.
      return self.data_generator.toTensorFormat(self.feed_queue[0].input_blob), [], False
    else:
      # Queue is empty and we can proceed to next feed from dataset.
      # Return back to models all good active samples, if any.
      active_batch   = [x.sample for x in self.total_candidates]
      active_indices = [x.sample_indices for x in self.total_candidates]
      self.total_candidates = []
      return active_batch, active_indices, True

  def addToDB(self, active_feed: active_feed_database.ActiveFeed) -> None:
    """If not exists, add current sample state to database"""
    with self.active_db.Session(commit = True) as session:
      exists = session.query(
        active_feed_database.ActiveFeed
      ).filter(active_feed_database.ActiveFeed.sha256 == active_feed.sha256).scalar() is not None
      if not exists:
        session.add(active_feed)
    return

class ActiveDataset(object):
  """Dataset as a concatenation of multiple datasets.

  This class is useful to assemble different existing datasets
  and instantiate them lazily, to avoid loading them all in
  memory at the same time/

  Arguments:
    datasets (sequence): List of paths for datasets to be concatenated
  """

  def __init__(self, dataset: typing.List[np.array]):
    assert len(dataset) > 0, 'Dataset is empty'
    self.dataset                 = dataset
    self.sample_idx              = 0
    return

  def __len__(self) -> int:
    return len(self.dataset)

  def __next__(self) -> np.array:
    """
    Iterator gives priority to additional active feeds if exist,
    otherwise pops a fresh feed from the dataset.
    """
    self.sample_idx += 1
    return self.dataset[(self.sample_idx - 1) % len(self.dataset)]
