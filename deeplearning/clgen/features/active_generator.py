"""
Data generator wrapper used for active learning sampling.
"""
import subprocess
import multiprocessing
import torch
import functools
import pickle
import math
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

torch.multiprocessing.set_sharing_strategy('file_system')

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

flags.DEFINE_integer(
  "active_search_width",
  5,
  "Set top-K surviving candidates per generation, sorted by distance from target feature space."
)

def candidate_worker(sample_out, tokenizer):
  sample, sample_indices, input_ids, masked_lm_lengths = sample_out
  try:
    # If you can extract features from a sample, store it as a candidate.
    code = tokenizer.ArrayToCode(sample, with_formatting = False)
    # print(code)
    _ = opencl.Compile(code)
    features = extractor.DictKernelFeatures(code)
    if features:
      return sample, sample_indices, features, input_ids, masked_lm_lengths
  except ValueError:
    pass
  except Exception:
    pass
  return None, None, None, None, None

def dataload_worker(x, feed, func, batch):
  try:
    return {
      k: torch.from_numpy(v).unsqueeze(0).repeat_interleave(batch, dim = 0)
      for (k, v) in func(feed).items()
    }
  except Exception:
    return None

class ActiveSampleFeed(typing.NamedTuple):
  """
  Representation of an active learning input to the model.
  """
  # An array of original input
  input_feed       : np.array
  # The feature space of the original input
  input_features   : typing.Dict[str, float]
  # Distance from target features of input feed. Valid after 1st generation.
  input_score      : float
  # Depth increases when a valid inference sample is fed back as an input.
  gen_id           : int

class ActiveSample(typing.NamedTuple):
  """
  Representation of an active learning sample.
  """
  # ActiveSampleFeed instance of model input
  sample_feed    : typing.TypeVar("ActiveSamplingGenerator.ActiveSampleFeed")
  # Input ids that led to this prediction
  input_ids      : np.array
  # hole lengths and positions of input ids.
  hole_instances : typing.List[sequence_masking.MaskedLmInstance]
  # Model prediction
  sample         : np.array
  # Sample indices of given prediction.
  sample_indices : np.array
  # Output features of sample
  features       : typing.Dict[str, float]
  # Score of sample based on active learning search.
  score          : typing.Union[bool, float]
  # Active batch timestep where sample was acquired.
  timestep       : int

class SampleTrainingOpts(typing.NamedTuple):
  max_predictions_per_seq: int
  masked_lm_prob: float

class ActiveSamplingGenerator(object):
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
    # d.dataloader = d.active_dataloader()
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

    self.current_generation = None
    self.comp_rate_per_gen  = {}

    self.configSamplingParams()
    self.configSampleCorpus()

    # Active sampling attributes.
    self.active_db = active_feed_database.ActiveFeedDatabase(
      url = "sqlite:///{}".format(self.data_generator.sampler.corpus_directory / "active_feeds.db")
    )
    if (self.data_generator.sampler.corpus_directory / "gen_state.pkl").exists():
      with open(self.data_generator.sampler.corpus_directory / "gen_state.pkl", 'rb') as infile:
        self.feed_queue = pickle.load(infile)
    else:
      self.feed_queue          = []
    self.step_candidates       = []
    self.total_candidates      = []
    self.total_candidates_hash = set()
    self.active_dataset        = ActiveDataset(self.active_corpus)
    self.feat_sampler          = feature_sampler.EuclideanSampler()
    self.candidate_monitor     = monitors.CategoricalDistribMonitor(
      self.data_generator.sampler.corpus_directory, "feature_distance"
    )
    self.comp_rate_monitor     = monitors.CategoricalHistoryMonitor(
      self.data_generator.sampler.corpus_directory, "comp_rate_per_gen"
    )
    return

  def initOrGetQueue(self) -> int:
    """
    If feed queue is not initialized, nitialize it by getting new datapoint.
    Adds datapoint to InputFeed table of database.

    Returns:
      generation_id
    """
    if not self.feed_queue:
      cf = next(self.active_dataset)
      self.feed_queue.append(
        ActiveSampleFeed(
          input_feed       = cf,
          input_features   = extractor.DictKernelFeatures(self.tokenizer.ArrayToCode(cf)),
          input_score      = math.inf,
          gen_id           = 0,
        )
      )
      self.addToDB(
        active_feed_database.ActiveInput.FromArgs(
          tokenizer      = self.tokenizer, id = self.active_db.input_count,
          input_feed     = cf, input_features = self.feed_queue[-1].input_features,
        )
      )
    return self.feed_queue[0].input_feed, self.feed_queue[0].input_feed

  def collateInputData(self,
                       feed: np.array,
                       wload_size: int,
                       ) -> typing.Dict[str, typing.TypeVar('torch.Tensor')]:
    """
    Create a full generation workload out of a sample feed.
    If feed is already masked, then just repeat it across the whole workload.
    If it is not masked, then feed is masked wload_size times.

    Args:
      feed: numpy array of input feed.
      wload_size: Number of inputs that will be fed to the model in a single workload.

    Returns:
      The tensor inputs dictionary filled for BERT.
    """
    if self.tokenizer.maskToken in feed or self.tokenizer.holeToken in feed:
      inputs = sequence_masking.MaskedSeqToBlob(
        feed, self.tokenizer,
        self.data_generator.sampler.sequence_length,
        self.data_generator.max_position_embeddings
      )
      inputs = {
        k: v.unsqueeze(0).repeat_interleave(wload_size, dim = 0) 
        for k, v in self.data_generator.toTensorFormat(inputs, batch = self.data_generator.sample_batch_size).items()
      }
    else:
      inputs = {
        'input_ids': [], 'input_mask': [], 'position_ids': [],
        'mask_labels': [], 'masked_lm_lengths': [], 'next_sentence_labels': []
      }
      try:
        pool = multiprocessing.Pool()
        for batch in pool.imap_unordered(
                          functools.partial(
                            dataload_worker, feed  = feed,
                            func  = self.func, batch = self.data_generator.sample_batch_size
                          ),range(wload_size)
                         ):
          if batch:
            inputs['input_ids'].append(batch['input_ids'])
            inputs['input_mask'].append(batch['input_mask'])
            inputs['position_ids'].append(batch['position_ids'])
            inputs['mask_labels'].append(batch['mask_labels'])
            inputs['masked_lm_lengths'].append(batch['masked_lm_lengths'])
            inputs['next_sentence_labels'].append(batch['next_sentence_labels'])
        pool.close()
      except KeyboardInterrupt as e:
        pool.close()
        pool.terminate()
        raise e
    return inputs

  def registerOutputData(self,
                         outputs: typing.Dict[str, typing.List[np.array]],
                         candidates: typing.List[ActiveSample],
                         bar: progressbar.ProgressBar,
                         ) -> typing.List[int]:
    """
    Gets workload output from model.
    In parallel, every sample is checked for compilability and features are extracted.
    If sample compiles, it is stored as an active learning candidate.

    Args:
      outputs: Dictionary output of workload
      candidates: Passed by reference and filled within this function
      bar: progressbar for status checking

    Returns:
      cm_rate: List of two elements that express compilation rate of workload.
               0th el: Total compiling.
               1st el: Total samples.
    """
    cm_rate = [0, 0]
    pool = multiprocessing.Pool()
    cm_rate[1] += len(outputs['generated_samples'])
    try:
      it = zip(
        outputs['generated_samples'], outputs['sample_indices'],
        outputs['input_ids'], outputs['masked_lm_lengths']
      )
      for batch in pool.imap_unordered(
                     functools.partial(candidate_worker, tokenizer = self.tokenizer), it
                   ):
        sample, indices, features, input_ids, masked_lm_lengths = batch
        if sample is not None:
          cm_rate[0] += 1
          bar.update(min(FLAGS.active_limit_per_feed, len(candidates)))
          candidates.append(
            ActiveSample(
              sample_feed    = feed,  sample         = sample,
              input_ids      = input_ids, hole_instances = [x for x in masked_lm_lengths if x >= 0],
              sample_indices = indices,   features       = features,
              score          = None,      timestep       = 1 + len(candidates),
            )
          )
      pool.close()
    except KeyboardInterrupt as e:
      pool.close()
      pool.terminate()
      raise e
    return cm_rate

  def ActiveGeneration(self,
                       mwrapper: typing.TypeVar('torch_bert.torchBert'),
                       estimator: typing.TypeVar('torch_bert.SampleBertEstimator')
                      ) -> typing.Tuple[np.array, np.array, np.array, np.array]:
    """
    Active Learning generation core routine.

    This function starts with a feed from a dataset
    and returns all active samples that have reached the requested feature space.

    Args:
      mwrapper: BERT model wrapper.
      estimator: BERT model pipeline.

    Returns:
      A tuple of 4 arrays:
        a) Original inputs
        b) Original input ids
        c) Generated samples
        d) Sample indices
      The arrays are ordered by index.
    """
    try:
      org_inp, org_ids = self.initOrGetQueue()
    except StopIteration:
      return [], [], [], []

    total_candidates, total_candidates_hash = [], set()

    while self.feed_queue:

      feed = self.feed_queue.pop(0)

      self.sampler.setStartText(self.tokenizer.tokensToString(feed.input_feed, ignore_token = self.tokenizer.padToken))
      self.sampler.Specialize(self.tokenizer)

      step_candidates = []
      rem = FLAGS.active_limit_per_feed // self.data_generator.sample_batch_size
      cmp_rate = [0, 0]

      bar = progressbar.ProgressBar(max_value = FLAGS.active_limit_per_feed)
      bar.update(0)

      if feed.gen_id not in self.comp_rate_per_gen:
        self.comp_rate_per_gen[feed.gen_id] = [0, 0]

      while len(step_candidates) < FLAGS.active_limit_per_feed:

        inputs = self.collateInputData(feed.input_feed, rem)

        outputs = mwrapper.sample_model_step(
          estimator.models, estimator.devices, inputs,
        )
        tcs, ts = self.registerOutputData(outputs, step_candidates, bar)
        cmp_rate[0] += tcs
        cmp_rate[1] += ts

        try:
          rcands = FLAGS.active_limit_per_feed - len(step_candidates)
          crate  = cmp_rate[0] / cmp_rate[1]
          rem = max(2, int((rcands // self.data_generator.sample_batch_size) / crate))
        except ZeroDivisionError:
          pass

      self.comp_rate_per_gen[feed.gen_id] = [sum(x) for x in zip(self.comp_rate_per_gen[feed.gen_id], cmp_rate)]
      self.comp_rate_monitor.register((feed.gen_id, self.comp_rate_per_gen[feed.gen_id][0] / self.comp_rate_per_gen[feed.gen_id][1]))
      self.comp_rate_monitor.plot()

      # total_candidates contains all candidates from all generations from a single starting feed
      # that succeeded the distance sampling. candidate_idx sets a checkpoint of which are old and
      # which are new. This is to avoid re-storing old total_candidates multiple times.
      candidate_idx = len(total_candidates)
      # Top-k candidates of ith generation.
      new_candidates = self.feat_sampler.sample_from_set(step_candidates)

      if feed.gen_id > 0:
        new_candidates = new_candidates[:1]

      self.candidate_monitor.register(
        {str(new_candidates[0].sample_feed.gen_id): [c.score for c in new_candidates]}
      )
      self.candidate_monitor.plot()
      # Very frequently, new candidates have been generated in the past.
      # No need to store them again, by keeping a hash of their string.
      for nc in new_candidates:
        sample_hash = ''.join([str(x) for x in nc.sample])
        if sample_hash not in total_candidates_hash:
          total_candidates.append(nc)
          total_candidates_hash.add(sample_hash)

      for candidate in total_candidates[candidate_idx:]:
        self.addToDB(
          active_feed_database.ActiveFeed.FromArgs(
            tokenizer        = self.tokenizer,
            id               = self.active_db.active_count,
            input_feed       = candidate.sample_feed.input_feed,
            input_features   = candidate.sample_feed.input_features,
            masked_input_ids = candidate.input_ids,
            hole_instances   = candidate.hole_instances,
            sample           = candidate.sample,
            output_features  = candidate.features,
            sample_quality   = candidate.score,
            compile_status   = True,
            generation_id    = candidate.sample_feed.gen_id,
            timestep         = candidate.timestep,
          )
        )
        # If the current generation has not gone as deep as required, mask each new candidate
        # and place it at the tail of the sample feed queue.
        if 0 < candidate.score < feed.input_score and 1+candidate.sample_feed.gen_id <= FLAGS.active_search_depth:
          self.feed_queue.append(
            ActiveSampleFeed(
              input_feed       = candidate.sample,
              input_features   = candidate.features,
              input_score      = candidate.score,
              gen_id           = 1 + candidate.sample_feed.gen_id,
            )
          )
      # Step candidates contains all candidates of a single gen, therefore initialized before every new gen.
      if self.feed_queue:
        # Update generation state pickle to start over.
        with open(self.data_generator.sampler.corpus_directory / "gen_state.pkl", 'wb') as outf:
          pickle.dump(self.feed_queue, outf)

    with open(self.data_generator.sampler.corpus_directory / "gen_state.pkl", 'wb') as outf:
      pickle.dump(self.feed_queue, outf)
    self.feat_sampler.iter_benchmark()
    return (np.repeat([org_inp], len(active_batch), axis = 0),
            np.repeat([org_ids], len(active_batch), axis = 0),
            [x.sample for x in total_candidates],
            [x.sample_indices for x in total_candidates])

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
      self.active_corpus = [self.tokenizer.TokenizeString(self.sampler.start_text)]
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
