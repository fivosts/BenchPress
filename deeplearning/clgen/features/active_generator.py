"""
Data generator wrapper used for active learning sampling.
"""
import subprocess
import functools
import pickle
import typing
import progressbar
import numpy as np

from deeplearning.clgen.models import lm_data_generator
from deeplearning.clgen.models import sequence_masking
from deeplearning.clgen.features import extractor
from deeplearning.clgen.features import feature_sampler
from deeplearning.clgen.features import active_feed_database
from deeplearning.clgen.preprocessors import opencl
from deeplearning.clgen.util import distributions
from deeplearning.clgen.util import monitors

from absl import flags
from eupy.native import logger as l

FLAGS = flags.FLAGS

flags.DEFINE_integer(
  "active_limit_per_feed",
  250,
  "Set limit on sample attempts per input_feed. [Default: 400]. Set to 0 for infinite."
)

flags.DEFINE_integer(
  "active_search_depth",
  30,
  "Set the maximum sampling generation depth that active sampler can reach. [Default: 20]."
)

class ActiveSamplingGenerator(object):
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
    sample_feed    : typing.TypeVar("ActiveSamplingGenerator.ActiveSampleFeed")
    # Model prediction
    sample         : np.array
    # Sample indices of given prediction.
    sample_indices : np.array
    # Output features of sample
    features       : typing.Dict[str, float]
    # Score of sample based on active learning search.
    score          : typing.Union[bool, float]

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

    self.data_generator = generator

    # Wrapped data generator attributes
    self.sampler   = self.data_generator.sampler
    self.tokenizer = self.data_generator.tokenizer

    # Inherent attributes
    self.active_corpus = None
    self.distribution  = None
    self.func          = None
    self.dataloader    = None
    self.init_masked   = False

    self.configSamplingParams()
    self.configSampleCorpus()

    # Active sampling attributes.
    self.active_db = active_feed_database.ActiveFeedDatabase(
      url = "sqlite:///{}".format(self.data_generator.sampler.corpus_directory / "active_feeds.db")
    )
    self.feed_queue            = []
    self.step_candidates       = []
    self.total_candidates      = []
    self.total_candidates_hash = set()
    self.active_dataset        = ActiveDataset(self.active_corpus)
    self.feat_sampler          = feature_sampler.EuclideanSampler()
    self.candidate_monitor     = monitors.MinRegulatedHistoryMonitor(
      self.data_generator.sampler.corpus_directory, "feature_distance"
    )
    self.bar = progressbar.ProgressBar(max_value = FLAGS.active_limit_per_feed)
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
      # seed = self.tokenizer.TokenizeString("[START]kernel void A(){ }[END][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD]")
      # print(len(seed))
      if not self.init_masked:
        input_feed = self.func(seed)
        input_ids = input_feed['input_ids']
      else:
        input_feed = sequence_masking.MaskedSeqToBlob(
          self.data_generator.sampler.encoded_start_text,
          self.tokenizer,
          self.data_generator.sampler.sequence_length,
          self.data_generator.max_position_embeddings
        )
      # TODO do sth with hole_lengths and masked_idxs
      self.feed_queue.append(
        ActiveSamplingGenerator.ActiveSampleFeed(
          input_feed       = seed,
          input_features   = extractor.DictKernelFeatures(self.tokenizer.ArrayToCode(seed)),
          input_blob       = input_feed,
          masked_input_ids = input_feed['input_ids'],
          hole_instances   = [x for x in input_feed['masked_lm_lengths'] if x >= 0],
          gen_id           = 0,
        )
      )
      self.addToDB(
        active_feed_database.ActiveInput.FromArgs(
          tokenizer      = self.tokenizer,
          id             = self.active_db.input_count,
          input_feed     = self.feed_queue[-1].input_feed,
          input_features = self.feed_queue[-1].input_features,
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

    # Pops the sample feed that created 'samples'
    current_feed = self.feed_queue.pop(0)
    for sample, indices in zip(samples, sample_indices):
      try:
        # If you can extract features from a sample, store it as a candidate.
        # _ = opencl.Compile(self.tokenizer.ArrayToCode(sample))
        features = extractor.DictKernelFeatures(self.tokenizer.ArrayToCode(sample))
        if features:
          self.step_candidates.append(
            ActiveSamplingGenerator.ActiveSample(
              sample_feed    = current_feed,
              sample         = sample,
              sample_indices = indices,
              features       = features,
              score          = None,
            )
          )
      except ValueError:
        pass

    if len(self.step_candidates) < FLAGS.active_limit_per_feed:
      # If gathered candidates are not as many as required, re-mask the same feed
      # place it back in the queue and ask the model for more samples.
      # The sample input is the same, but masks might be in different locations.

      if self.tokenizer.maskToken not in current_feed.input_feed and self.tokenizer.holeToken not in current_feed.input_feed:
        input_feed = self.func(current_feed.input_feed)
      else:
        input_feed = current_feed.input_blob

      self.feed_queue.insert(0,
        ActiveSamplingGenerator.ActiveSampleFeed(
          input_feed       = current_feed.input_feed,
          input_features   = current_feed.input_features,
          input_blob       = input_feed,
          masked_input_ids = input_feed['input_ids'],
          hole_instances   = [x for x in input_feed['masked_lm_lengths'] if x >= 0],
          gen_id           = current_feed.gen_id,
        )
      )
      self.bar.update(len(self.step_candidates))
      return self.data_generator.toTensorFormat(input_feed), [], False

    # Re-init bar for next candidate gathering
    self.bar = progressbar.ProgressBar(max_value = FLAGS.active_limit_per_feed)

    # total_candidates contains all candidates from all generations from a single starting feed
    # that succeeded the distance sampling. candidate_idx sets a checkpoint of which are old and
    # which are new. This is to avoid re-storing old total_candidates multiple times.
    candidate_idx = len(self.total_candidates)
    # Top-k candidates of ith generation.
    new_candidates = self.feat_sampler.sample_from_set(self.step_candidates)

    self.candidate_monitor.register(
      sum([x.score for x in new_candidates]) / len(new_candidates)
    )
    self.candidate_monitor.plot()
    # Very frequently, new candidates have been generated in the past.
    # No need to store them again, by keeping a hash of their string.
    for nc in new_candidates:
      sample_hash = ''.join([str(x) for x in nc.sample])
      if sample_hash not in self.total_candidates_hash:
        self.total_candidates.append(nc)
        self.total_candidates_hash.add(sample_hash)

    for candidate in self.total_candidates[candidate_idx:]:
      # For new total_candidates, check compilability, return error locations if they are incorrect.
      _, dloc = opencl.Compile(self.tokenizer.ArrayToCode(candidate.sample), return_diagnostics = True)
      compile_status = True
      if dloc:
        compile_status = False
        # The following function maps compiler location diagnostics, e.g. l:5, c:10, to token index of the encoded sequennce.
        # This will be used to repair broken kernels by targetted masking of wrong tokens.
        # indices = self.tokenizer.SrcLocationToIndex(
        #   locations = dloc,
        #   encoded = candidate.sample
        # )
      # Add to active_database.
      self.addToDB(
        active_feed_database.ActiveFeed.FromArgs(
          tokenizer        = self.tokenizer,
          id               = self.active_db.active_count,
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
      # If the current generation has not gone as deep as required, mask each new candidate
      # and place it at the tail of the sample feed queue.
      if 1 + candidate.sample_feed.gen_id <= FLAGS.active_search_depth:
        if not self.data_generator.config.use_start_end or (self.data_generator.config.use_start_end and self.tokenizer.endToken in candidate.sample):
          input_feed = self.func(candidate.sample)
          if len(input_feed['input_ids']) <= self.data_generator.sampler.sequence_length:
            self.feed_queue.append(
              ActiveSamplingGenerator.ActiveSampleFeed(
                input_feed       = candidate.sample,
                input_features   = candidate.features,
                input_blob       = input_feed,
                masked_input_ids = input_feed['input_ids'],
                hole_instances   = [x for x in input_feed['masked_lm_lengths'] if x >= 0],
                gen_id           = 1 + candidate.sample_feed.gen_id,
              )
            )
    # Step candidates contains all candidates of a single gen, therefore initialized before every new gen.
    self.step_candidates = []

    if self.feed_queue:
      # There are next generation candidates in the queue, and these will be used as sample feeds.
      return self.data_generator.toTensorFormat(self.feed_queue[0].input_blob), [], False
    else:
      # We don't have new generation good candidates. Go get a new starting feed from the dataset.
      # Active learning will be killed (see True value in return statement), the harvested batch
      # will be returned to models. Models will re-init active learner for the next dataset input.
      active_batch   = [x.sample for x in self.total_candidates]
      active_indices = [x.sample_indices for x in self.total_candidates]
      self.total_candidates      = []
      self.total_candidates_hash = set()
      # self.candidate_monitor   = monitors.HistoryMonitor(
      #   self.data_generator.sampler.corpus_directory, "feature_distance"
      # )
      return active_batch, active_indices, True

  def addToDB(self,
              db_input: typing.Union[
                          active_feed_database.ActiveFeed,
                          active_feed_database.ActiveInput
                        ]
              ) -> None:
    """If not exists, add current sample state to database"""
    with self.active_db.Session(commit = True) as session:
      exists = session.query(
        type(db_input)
      ).filter(type(db_input).sha256 == db_input.sha256).scalar() is not None
      if not exists:
        session.add(db_input)
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
      self.func = functools.partial(sequence_masking.HoleSequence,
                            train_set       = False,
                            max_predictions = corpus_config.max_predictions_per_seq,
                            distribution    = self.distribution,
                            tokenizer       = self.tokenizer,
                            training_opts   = sampling_opts,
                          )
    elif corpus_config.HasField("mask"):
      self.func = functools.partial(sequence_masking.MaskSequence,
                            train_set          = False,
                            max_predictions    = corpus_config.max_predictions_per_seq,
                            config             = corpus_config,
                            pickled_tokenizer   = self.tokenizer,
                            training_opts      = sampling_opts,
                            is_torch           = self.data_generator.is_torch,
                          )
    return

  def configSampleCorpus(self) -> None:
    """
    Configure sampling corpus container to iterate upon.
    """
    if self.sampler.isFixedStr:
      if (self.tokenizer.maskToken in self.sampler.encoded_start_text or
          self.tokenizer.holeToken in self.sampler.encoded_start_text):
        self.init_masked = True
      self.active_corpus = [self.sampler.encoded_start_text]
    else:
      self.active_corpus = self.data_generator.createCorpus(self.sampler.corpus_directory)
    return

class ActiveDataset(object):
  """Dataset as a concatenation of multiple datasets.

  This class is useful to assemble different existing datasets
  and instantiate them lazily, to avoid loading them all in
  memory at the same time/

  Arguments:
    datasets (sequence): List of paths for datasets to be concatenated
  """

  def __init__(self, dataset: typing.List[np.array], shuffle = True):
    assert len(dataset) > 0, 'Dataset is empty'
    self.dataset    = dataset
    self.sample_idx = 0
    if shuffle:
      np.random.shuffle(self.dataset)
    return

  def __len__(self) -> int:
    return len(self.dataset)

  def __next__(self) -> np.array:
    """
    Iterator pops a fresh feed from the dataset.
    If dataset is consumed, then it is from the beginning.
    """
    self.sample_idx += 1
    return self.dataset[(self.sample_idx - 1) % len(self.dataset)]
