"""This file defines the streaming generators for model training data.

We train models on overlapping one-hot encoded text sequences. For a corpus of
a reasonable size, the full training data may not fit in memory. This modules
provides Python Generator classes for use by a sequential Keras model's
fit_generator() method to stream batches of training data.
"""
import os
import typing
import copy
import datetime
import glob
import humanize
import sklearn
import pickle
import functools
import numpy as np
import pathlib
import multiprocessing
import math
import tqdm
import threading

from deeplearning.clgen.util import pytorch
from deeplearning.clgen.util.pytorch import torch
from deeplearning.clgen.util import distributions
from deeplearning.clgen.util import monitors
from deeplearning.clgen.util import environment
from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen.features import extractor
from deeplearning.clgen.features import feature_sampler
from deeplearning.clgen.features import active_feed_database
from deeplearning.clgen.features import evaluate_cand_database
from deeplearning.clgen.models import lm_data_generator
from deeplearning.clgen.models import sequence_masking
from deeplearning.clgen.models.torch_bert import datasets
from deeplearning.clgen.samplers import sample_observers
from deeplearning.clgen.preprocessors import opencl
from absl import flags
from deeplearning.clgen.util import logging as l

FLAGS = flags.FLAGS

flags.DEFINE_boolean(
  "skip_first_queue",
  False,
  "Hacky way to speedup active sampling experiments."
)

flags.DEFINE_boolean(
  "evaluate_candidates",
  False,
  "Select to do exhaustive evaluation of sampling search candidates."
)

flags.DEFINE_boolean(
  "evolutionary_search",
  True,
  "Select to perform independent per-generation candidate search instead of son-better-than-parent paradigm."
)

flags.DEFINE_integer(
  "sample_workload_size",
  8192,
  "Select size of workload per inference step."
)

flags.DEFINE_boolean(
  "features_standard_scaler",
  False,
  "Select to use sklearn StandardScaler for generation standardization."
)

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
  hole_lengths : typing.List[sequence_masking.MaskedLmInstance]
  # Model prediction
  sample         : np.array
  # Sample indices of given prediction.
  sample_indices : np.array
  # number of tokens the model filled holes with.
  sample_indices_size : int
  # Output features of sample
  features       : typing.Dict[str, float]
  # Score of sample based on active learning search.
  score          : typing.Union[bool, float]
  # Active batch timestep where sample was acquired.
  # timestep       : int

def IR_candidate_worker(sample       : np.array,
                        # feed         : np.array,
                        feat_sampler : feature_sampler.EuclideanSampler,
                        tokenizer    : typing.TypeVar('corpuses.tokenizers.TokenizerBase'),
                        ) -> ActiveSample:
  # sample, indices, input_ids, masked_lm_lengths = sample_out
  sample, sample_indices, input_ids, mlm_lengths, feed = sample
  try:
    code = tokenizer.ArrayToCode(sample, with_formatting = False)
    features = extractor.ExtractFeatures(code, [feat_sampler.feature_space])[feat_sampler.feature_space]
    if features:
      return (True, ActiveSample(
        sample_feed = feed,
        sample      = sample,
        sample_indices = [x for x in sample_indices if x != tokenizer.padToken],
        input_ids      = [x for x in input_ids if x != tokenizer.padToken],
        hole_lengths   = mlm_lengths,
        sample_indices_size = len([x for x in sample_indices if x != tokenizer.padToken]),
        features       = features,
        score          = feat_sampler.calculate_distance(features),
      ))
  except ValueError:
    pass
  except Exception as e:
    raise e
  return (False, ActiveSample(
    sample_feed = feed,
    sample      = sample,
    sample_indices = [x for x in sample_indices if x != tokenizer.padToken],
    input_ids      = [x for x in input_ids if x != tokenizer.padToken],
    hole_lengths   = mlm_lengths,
    sample_indices_size = len([x for x in sample_indices if x != tokenizer.padToken]),
    features       = {},
    score          = math.inf,
  ))

def text_candidate_worker(sample       : np.array,
                          # feed         : np.array,
                          feat_sampler : feature_sampler.EuclideanSampler,
                          tokenizer    : typing.TypeVar('corpuses.tokenizers.TokenizerBase'),
                          ) -> ActiveSample:
  sample, sample_indices, input_ids, mlm_lengths, feed = sample
  try:
    code = tokenizer.ArrayToCode(sample, with_formatting = False)
    _ = opencl.Compile(code)
    features = extractor.ExtractFeatures(code, [feat_sampler.feature_space])[feat_sampler.feature_space]
    if features:
      return (True, ActiveSample(
        sample_feed = feed,
        sample      = sample,
        sample_indices = [x for x in sample_indices if x != tokenizer.padToken],
        input_ids      = [x for x in input_ids if x != tokenizer.padToken],
        hole_lengths   = mlm_lengths,
        sample_indices_size = len([x for x in sample_indices if x != tokenizer.padToken]),
        features       = features,
        score          = feat_sampler.calculate_distance(features),
      ))
  except ValueError:
    pass
  except Exception as e:
    raise e
  return (False, ActiveSample(
    sample_feed = feed,
    sample      = sample,
    sample_indices = [x for x in sample_indices if x != tokenizer.padToken],
    input_ids      = [x for x in input_ids if x != tokenizer.padToken],
    hole_lengths   = mlm_lengths,
    sample_indices_size = len([x for x in sample_indices if x != tokenizer.padToken]),
    features       = {},
    score          = math.inf,
  ))

def dataload_worker(x              : int,
                    feed           : typing.List[np.array],
                    func           : typing.TypeVar('sequence_masking.MaskingFunction'),
                    batch          : int,
                    batch_per_feed : int,
                    ) -> typing.Dict[str, np.array]:
  try:
    # return [f for _ in range(batch // batch_per_feed) for f in [func(fd) for fd in feed] * batch_per_feed]
    return [f for _ in range(batch // batch_per_feed) for f in [func(fd) for fd in feed * batch_per_feed]]
  except Exception as e:
    raise e

def write_samples_cache(db_sample_obs : sample_observers.SamplesDatabaseObserver,
                        tokenizer     : "tokenizers.TokenizerBase",
                        samples       : typing.List[ActiveSample]
                        ) -> None:
  for sample in samples:
    try:
      s = model_pb2.Sample(
        train_step = -1,
        text = tokenizer.ArrayToCode(sample.sample, with_formatting = True),
        sample_indices = "",
        encoded_sample_indices = "",
        original_input = "",
        sample_feed    = tokenizer.ArrayToCode(sample.sample_feed.input_feed, with_formatting = True),
        encoded_text   = "",
        sample_start_epoch_ms_utc = 0,
        sample_time_ms = 0,
        wall_time_ms   = 0,
        feature_vector = '\n'.join(["{}:{}".format(k, v) for k, v in sample.features.items()]) if sample.features else "None",
        num_tokens     = np.where(sample.sample == tokenizer.padToken)[0][0] if tokenizer.padToken in sample.sample else len(sample),
        compile_status = True,
        categorical_sampling = FLAGS.categorical_sampling,
        date_added           = datetime.datetime.utcnow().strftime("%m/%d/%Y, %H:%M:%S"),
      )
      db_sample_obs.OnSample(s)
    except Exception:
      pass
  return

def write_eval_db(eval_db   : evaluate_cand_database.SearchCandidateDatabase,
                  tokenizer : "tokenizers.TokenizerBase",
                  samples   : typing.List[ActiveSample],
                  target_benchmark : typing.Tuple[str, str],
                  target_features  : typing.Dict[str, float],
                  gen_id    : int,
                  ) -> None:
  objs = {}
  # l.logger().warn("Before prep loop in eval db")
  for sample in samples:
    try:
      _ = opencl.Compile(tokenizer.ArrayToCode(sample.sample))
      compile_status = True
    except ValueError:
      compile_status = False

    sobj = evaluate_cand_database.SearchCandidate.FromArgs(
      tokenizer        = tokenizer,
      id               = eval_db.count,
      input_feed       = sample.sample_feed.input_feed,
      input_ids        = sample.input_ids,
      input_features   = sample.sample_feed.input_features,
      input_score      = sample.sample_feed.input_score,
      hole_lengths     = sample.hole_lengths,
      sample           = sample.sample,
      sample_indices   = sample.sample_indices,
      output_features  = sample.features,
      sample_score     = sample.score,
      target_benchmark = target_benchmark,
      target_features  = target_features,
      compile_status   = compile_status,
      generation_id    = gen_id,
    )
    if sobj.sha256 in objs:
      objs[sobj.sha256][1] += 1
    else:
      objs[sobj.sha256] = [sobj, 1]
  # l.logger().warn(eval_db.count)
  with eval_db.Session(commit = True) as session:
    offset_idx = 0
    for sha, obj in objs.items():
      try:
        entry = session.query(evaluate_cand_database.SearchCandidate).filter_by(sha256 = sha).first()
        if entry is not None:
          entry.frequency += obj[1]
        else:
          obj[0].frequency = obj[1]
          obj[0].id += offset_idx
          offset_idx += 1
          session.add(obj[0])
        session.commit()
      except Exception as e:
        l.logger().error(entry)
        if entry is not None:
          l.logger().error(entry.id)
          l.logger().error(entry.sha256)
        l.logger().error(sha)
        l.logger().error("count: {}".format(eval_db.count))
        l.logger().error("offset_idx: {}".format(offset_idx))
        print(e)
  # l.logger().warn("Finished eval_DB thread")
  return

class torchLMDataGenerator(lm_data_generator.MaskLMDataGenerator):
  """Data generator subclass designed for PyTorch BERT model."""
  @classmethod
  def TrainMaskLMBatchGenerator(cls,
                               corpus: "corpuses.Corpus",
                               training_opts: model_pb2.TrainingOptions,
                               cache_path,
                               num_train_steps: int = None,
                               pre_train: bool = False,
                               feature_encoder         : bool                        = False,
                               feature_tokenizer       : tokenizers.FeatureTokenizer = None,
                               feature_sequence_length : int                         = None,
                               ) -> "data_generator.MaskLMBatchGenerator":
    """Initializes data generator for training."""
    d = super(torchLMDataGenerator, torchLMDataGenerator()).TrainMaskLMBatchGenerator(
                corpus, training_opts, cache_path, num_train_steps, pre_train,
                feature_encoder, feature_tokenizer, feature_sequence_length,
        )
    d.dataloader = d.train_dataloader()
    return d

  @classmethod
  def SampleMaskLMBatchGenerator(cls,
                                 model_opts,
                                 sampler,
                                 tokenizer,
                                 seed: int,
                                 sample_batch_size: int,
                                 max_position_embeddings: int,
                                 cache_path,
                                 corpus: "corpuses.Corpus" = None,
                                 feature_encoder         : bool                        = False,
                                 feature_tokenizer       : tokenizers.FeatureTokenizer = None,
                                 feature_sequence_length : int                         = None,
                                 ) -> "data_generator.MaskLMBatchGenerator":
    """Initializes data generator for inference."""
    d = super(torchLMDataGenerator, torchLMDataGenerator()).SampleMaskLMBatchGenerator(
              model_opts, sampler, tokenizer, seed,
              sample_batch_size, max_position_embeddings, cache_path,
              feature_encoder, feature_tokenizer, feature_sequence_length
        )
    if sampler.is_active:
      corpus_config = d.sampler.config.sample_corpus.corpus_config
      if corpus_config.HasField("hole"):
        distribution = distributions.Distribution.FromHoleConfig(
          corpus_config.hole, d.sampler.corpus_directory, "sample_corpus"
        )
        d.func = functools.partial(sequence_masking.HoleSequence,
                              train_set       = False,
                              max_predictions = corpus_config.max_predictions_per_seq,
                              masked_lm_prob  = corpus_config.masked_lm_prob,
                              distribution    = distribution,
                              tokenizer       = d.tokenizer,
                            )
      elif corpus_config.HasField("mask_seq"):
        distribution = distributions.Distribution.FromHoleConfig(
          corpus_config.mask_seq, d.sampler.corpus_directory, "sample_corpus"
        )
        d.func = functools.partial(sequence_masking.HoleSequenceSeqMasks,
                              train_set       = False,
                              max_predictions = corpus_config.max_predictions_per_seq,
                              masked_lm_prob  = corpus_config.masked_lm_prob,
                              distribution    = distribution,
                              tokenizer       = d.tokenizer,
                            )
      elif corpus_config.HasField("mask"):
        d.func = functools.partial(sequence_masking.MaskSequence,
                              train_set         = False,
                              max_predictions   = corpus_config.max_predictions_per_seq,
                              masked_lm_prob    = corpus_config.masked_lm_prob,
                              config            = corpus_config,
                              pickled_tokenizer = d.tokenizer,
                              is_torch          = True,
                            )
      d.loadCheckpoint()
      # Active sampling attributes.
      d.active_db = active_feed_database.ActiveFeedDatabase(
        url = "sqlite:///{}".format(d.sampler.corpus_directory / "active_feeds.db")
      )
      d.samples_cache_obs = sample_observers.SamplesDatabaseObserver(
        path = d.sampler.corpus_directory / "samples_cache.db",
        must_exist = False,
      )
      if FLAGS.evaluate_candidates:
        d.eval_db = evaluate_cand_database.SearchCandidateDatabase(
          url = "sqlite:///{}".format(d.sampler.corpus_directory / "evaluated_candidates.db"),
          must_exist = False,
        )
      d.feat_sampler      = feature_sampler.EuclideanSampler(
        d.sampler.corpus_directory,
        corpus_config.active.feature_space,
        corpus_config.active.target,
        git_corpus = corpus
      )
      d.candidate_monitor = monitors.CategoricalDistribMonitor.loadCheckpoint(
        d.sampler.corpus_directory, "feature_distance"
      )
      d.tsne_monitor      = monitors.TSNEMonitor.loadCheckpoint(
        d.sampler.corpus_directory, "tsne_feature_map"
      )
      d.comp_rate_mon     = monitors.CategoricalHistoryMonitor.loadCheckpoint(
        d.sampler.corpus_directory, "comp_rate_per_gen"
      )
      d.exec_time_mon     = monitors.CategoricalHistoryMonitor.loadCheckpoint(
        d.sampler.corpus_directory, "exec_time_per_gen"
      )
      # Check if benchmark set has been registed to monitor.
      if d.feat_sampler.target not in d.tsne_monitor.groups_set:
        for b in d.feat_sampler.benchmarks:
          d.tsne_monitor.register((b.features, d.feat_sampler.target, b.name))
        d.tsne_monitor.plot()
      # Store unique specs to database once.
      d.addToDB(
        active_feed_database.ActiveSamplingSpecs.FromArgs(
          act_l_pf   = corpus_config.active.active_limit_per_feed,
          act_s_dep  = corpus_config.active.active_search_depth,
          act_s_wid  = corpus_config.active.active_search_width,
          feat_space = corpus_config.active.feature_space
        )
      )
      d.raised_keyboard_int = False
      d.raised_exception    = None
      d.skip_first_queue    = FLAGS.skip_first_queue

    d.dataloader = d.predict_dataloader()
    d.loader     = iter(d.dataloader)
    return d

  def __init__(self):
    super(torchLMDataGenerator, self).__init__("pt_record")
    self.dataloader = None
    ## Active learning attributes initialization.
    self.loader     = None
    self.comp_rate  = {}
    self.exec_time  = {}
    self.feed_queue = []
    self.active_db  = None
    self.samples_cache_obs = None
    self.eval_db           = None
    self.feat_sampler      = None
    self.candidate_monitor = None
    self.tsne_monitor      = None
    self.comp_rate_mon     = None
    self.exec_time_mon     = None
    self.raised_keyboard_int = None
    self.raised_exception  = None
    self.skip_first_queue  = None
    self.bench_idx         = None
    return

  def train_dataloader(self, set_name = 'train_dataset', is_train = True) -> torch.utils.data.dataloader:
    """
    Pytorch dataloader used for training.
  
    set_name defaults to train_dataset, and that way this function
    this dataloader's function is used for training.

    eval_dataloaders sets set_name to reuse the function for all different sets.
    """
    if self.config.datapoint_time == "pre":
      dataset = datasets.LazyConcatDataset([x for x in self.dataset[set_name]['file']])
      sampler = datasets.LazyRandomSampler(dataset, replacement = False)
    elif self.config.datapoint_time == "online":
      if self.pre_train:
        dataset = datasets.LazyOnlineDataset(self, is_train)
        sampler = datasets.LazyRandomSampler(dataset, replacement = False)
      else:
        dataset = datasets.OnlineDataset(self, is_train)
        if environment.WORLD_SIZE == 1:
          sampler = torch.utils.data.RandomSampler(dataset, replacement = False)
        else:
          sampler = torch.utils.data.DistributedSampler(dataset)
    else:
      raise ValueError(self.config.datapoint_time)

    dataloader = torch.utils.data.dataloader.DataLoader(
      dataset    = dataset,
      batch_size = self.training_opts.batch_size,
      sampler    = (sampler
        if pytorch.num_nodes <= 1 or not pytorch.torch_tpu_available or pytorch.torch_xla.xrt_world_size() <= 1
        else torch.utils.data.distributed.DistributedSampler(
          dataset      = dataset,
          num_replicas = pytorch.num_nodes if not pytorch.torch_tpu_available else pytorch.torch_xla.xrt_world_size(),
          rank         = pytorch.torch.distributed.get_rank() if not pytorch.torch_tpu_available else pytorch.torch_xla.get_ordinal()
        )
      ),
      num_workers = 0,
      drop_last   = True if environment.WORLD_SIZE > 1 else False,
    )
    return dataloader

  def eval_dataloaders(self) -> torch.utils.data.dataloader:
    """Pytorch dataloader used for validation."""
    if self.config.datapoint_time == "online":
      yield "Online Corpus", self.train_dataloader(is_train = False)
    else:
      for set_name in self.dataset:
        yield set_name, self.train_dataloader(set_name)

  def predict_dataloader(self) -> torch.utils.data.dataloader:
    """
    Pytorch dataloader used for inference.
    
    isFixedStr == True means there is a fixed sample feed, e.g. 'kernel void [HOLE]'
    Otherwise, a set has been given to provide random samples from it.
    """
    batch_size = self.sample_batch_size
    if not self.sampler.is_active and (self.sampler.isFixedStr or self.sampler.is_live):
      sample_element = sequence_masking.MaskedSeqToBlob(
        self.sampler.encoded_start_text, self.tokenizer, self.sampler.sequence_length, self.max_position_embeddings
      )
      dataset = [{k: torch.from_numpy(v) for (k, v) in sample_element.items()}] * self.sample_batch_size
      sampler = torch.utils.data.SequentialSampler(dataset)
    else:
      if self.sampler.is_online:
        """
        TODO maybe add configSampleSets here as well.
        """
        if self.pre_train:
          dataset = datasets.LazyOnlineDataset(self, False)
          sampler = datasets.LazyRandomSampler(dataset, replacement = False)
        else:
          dataset = datasets.OnlineDataset(self, False)
          if environment.WORLD_SIZE == 1:
            sampler = torch.utils.data.RandomSampler(dataset, replacement = False)
          else:
            sampler = torch.utils.data.DistributedSampler(dataset)
      elif self.sampler.is_active:
        if self.sampler.isFixedStr:
          dataset = [np.asarray(self.tokenizer.TokenizeString(self.sampler.start_text))]
        else:
          dataset = self.createCorpus(self.sampler.corpus_directory)
        batch_size = 1
        sampler = torch.utils.data.SequentialSampler(dataset)
      else:
        path_list = self.configSampleSets()
        dataset = datasets.LazyConcatDataset(
                    [x for x in path_list]
                  )
        sampler = datasets.LazyRandomSampler(dataset, replacement = False)
    dataloader = torch.utils.data.dataloader.DataLoader(
      dataset    = dataset,
      # Model's batch size is divided by sampler's batch size, in order to get
      # multiple generation candidates from a given sample feed, but still
      # efficiently feed big batches to make sampling faster.
      # Example: model batch size 32 and sampler batch size 4.
      # This dataloader will return 8 feeds. Each will be repeated 4 times.
      # 32 sequences will be given to the model.
      batch_size = batch_size,
      sampler    = (sampler
        if pytorch.num_nodes <= 1 or not pytorch.torch_tpu_available or pytorch.torch_xla.xrt_world_size() <= 1
        else torch.utils.data.distributed.DistributedSampler(
          dataset      = dataset,
          num_replicas = pytorch.num_nodes if not pytorch.torch_tpu_available else pytorch.torch_xla.xrt_world_size(),
          rank         = pytorch.torch.distributed.get_rank() if not pytorch.torch_tpu_available else pytorch.torch_xla.get_ordinal()
          )
      ),
      num_workers = 0,
      drop_last   = False,
      )
    return dataloader

  def ActiveGeneration(self,
                       mwrapper  : typing.TypeVar('torch_bert.torchBert'),
                       estimator : typing.TypeVar('torch_bert.SampleBertEstimator')
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
    if not self.feat_sampler.target_benchmark:
      raise StopIteration
    if self.raised_keyboard_int:
      self.raised_keyboard_int = False
      raise KeyboardInterrupt
    if self.raised_exception:
      raise self.raised_exception

    # Active sampling specs initialization
    active_limit_per_feed = self.sampler.config.sample_corpus.corpus_config.active.active_limit_per_feed
    active_search_depth   = self.sampler.config.sample_corpus.corpus_config.active.active_search_depth
    active_search_width   = self.sampler.config.sample_corpus.corpus_config.active.active_search_width
    sample_batch_per_feed = self.sampler.config.sample_corpus.corpus_config.active.batch_size_per_feed

    # Initialize feed queue
    org_inp = self.initOrGetQueue()
    org_ids = copy.copy(org_inp)
    total_cand, total_cand_hash = [], set()

    # Sample cache thread, eval cand DB thread.
    write_cache_proc = None
    if FLAGS.evaluate_candidates:
      write_eval_proc = None

    try:
      ## BFS style. While you have jobs, keep going.
      while self.feed_queue:

        ## Pop the feed that will probide a sample workload.
        if FLAGS.evolutionary_search:
          try:
            init_feed = self.feed_queue.pop(0)
            feeds = [init_feed]
            cur_gen = init_feed.gen_id
            while self.feed_queue[0].gen_id == cur_gen:
              feeds.append(self.feed_queue.pop(0))
          except Exception:
            pass
        else:
          feeds = [self.feed_queue.pop(0)]
          if self.skip_first_queue:
            self.skip_first_queue = False
            try:
              feeds = [self.feed_queue.pop(0)]
            except Exception:
              pass

        l.logger().info("Benchmark {}, generation {}".format(self.bench_idx, feeds[0].gen_id))
        # Compilation rate, execution time, per generation.
        cmp_rate        = [0, 0]
        exec_time       = 0.0

        if feeds[0].gen_id not in self.comp_rate:
          self.comp_rate[feeds[0].gen_id] = [0, 0]
        if feeds[0].gen_id not in self.exec_time:
          self.exec_time[feeds[0].gen_id] = 0.0

        # Specialize sampler to current sampling input.
        for feed in feeds[:1]:
          self.sampler.setStartText(self.tokenizer.tokensToString(feed.input_feed, ignore_token = self.tokenizer.padToken))
          self.sampler.Specialize(self.tokenizer)

        # Iterate until you get a better sample or surpass the limit.
        better_found, it, threshold = None, 0, 160000

        l.logger().info("Current input feed scores: {}".format(', '.join([str(round(feed.input_score, 3)) for feed in feeds])))
        while not better_found and cmp_rate[1] < threshold:
          ## Pre-process inputs
          # workload size: how many batches of sequences you need.
          wsize = FLAGS.sample_workload_size // self.sample_batch_size
          if FLAGS.evolutionary_search and feeds[0].gen_id == 0:
            wsize = wsize * active_search_width
          # Give the input feed and some specs, get the tensor ready to feed.
          inputs = self.collateInputData([feed.input_feed for feed in feeds], wsize, sample_batch_per_feed)
          ## Workload inference.
          outputs, time = mwrapper.sample_model_step(
            estimator.model,
            inputs,
            iteration = it,
          )
          ## Post-process outputs.
          # Keep step_candidates and evaluate them. Keep rejected candidates only for eval_cand database.
          step_candidates, rejected_candidates = [], []
          bar = lambda x: tqdm.tqdm(x, total = len(feeds) * wsize * self.sample_batch_size, desc = "Register Output Data", leave = False)
          tcs, ts = 0, 0
          # outputs = torch.reshape(outputs.unsqueeze(0), (len(feeds), -1, 768))
          # for idx, feed in enumerate(feeds):
          (cs, s), better_found = self.registerOutputData(
            outputs,
            # (idx*wsize*self.sample_batch_size, (idx+1)*wsize*self.sample_batch_size),
            [feeds[idx] for fidx, _ in enumerate(feeds) for idx in [fidx]*wsize*self.sample_batch_size],
            step_candidates,
            rejected_candidates,
            bar
          )
          tcs += cs
          ts  =  s

          ## Register good offsprings, along with step candidates in tsne monitor.
          if not FLAGS.evolutionary_search and better_found:
            self.tsne_monitor.register((better_found.features, "gen_{}_accepted".format(str(feeds[0].gen_id)), str(better_found.score)))
            for c in step_candidates:
              self.tsne_monitor.register((c.features, "gen_{}".format(str(feeds[0].gen_id))))

          ## Recalculate compilation rate of generation.
          cmp_rate[0] += tcs
          cmp_rate[1] += ts
          exec_time   += time

          # ## Write to samples cache DB.
          # if write_cache_proc:
          #   write_cache_proc.join()
          # self.samples_cache_obs.sample_id = self.samples_cache_obs.db.count
          # write_cache_proc = multiprocessing.Process(
          #   target = write_samples_cache,
          #   kwargs = {
          #     'db_sample_obs' : self.samples_cache_obs,
          #     'tokenizer'     : self.tokenizer,
          #     'samples'       : step_candidates,
          #   }
          # )
          # write_cache_proc.start()

          ## Write all candidates to eval_cand DB.
          if FLAGS.evaluate_candidates:
            # l.logger().warn("Before join: {}".format(write_eval_proc))
            if write_eval_proc:
              write_eval_proc.join()
            # l.logger().warn("After join: {}".format(write_eval_proc))
            write_eval_proc = multiprocessing.Process(
              target = write_eval_db,
              kwargs = {
                'eval_db'   : self.eval_db,
                'tokenizer' : self.tokenizer,
                'samples'   : step_candidates + rejected_candidates,
                'target_benchmark' : (self.feat_sampler.target_benchmark.name, self.feat_sampler.target_benchmark.contents),
                'target_features'  : self.feat_sampler.target_benchmark.features,
                'gen_id'           : feeds[0].gen_id,
              }
            )
            write_eval_proc.start()
            # l.logger().warn("Started process")

          if not FLAGS.evolutionary_search and better_found and feeds[0].gen_id > 0:
            l.logger().info("Improved score {} -> {} in {} iterations".format(round(feed.input_score, 3), round(better_found.score, 3), it))
          # Calculate how many more to infer.
          try:
            rcands = active_limit_per_feed - len(step_candidates) # Deprecated.
            crate  = cmp_rate[0] / cmp_rate[1] # Get current compilation rate.
            wsize = max(2, int((rcands // self.sample_batch_size) / crate)) # Deprecated.
          except ZeroDivisionError:
            pass
          # Step counter.
          it += 1
          if FLAGS.evolutionary_search:
            # No need to keep looking for better samples than parents.
            # In this mode, you get a workload and keep the best independently.
            break
        ######## End of while.

        ## Update all monitors.
        self.comp_rate[feeds[0].gen_id] = [sum(x) for x in zip(self.comp_rate[feeds[0].gen_id], cmp_rate)]
        self.exec_time[feeds[0].gen_id] += exec_time
        self.comp_rate_mon.register((feeds[0].gen_id, self.comp_rate[feeds[0].gen_id][0] / self.comp_rate[feeds[0].gen_id][1]))
        self.exec_time_mon.register((feeds[0].gen_id, self.exec_time[feeds[0].gen_id]    / self.comp_rate[feeds[0].gen_id][1]))
        self.comp_rate_mon.plot()
        self.exec_time_mon.plot()
        # self.tsne_monitor.plot()

        ## Collect surviving candidates of generation.
        # If we just started, get top-K.
        if FLAGS.evolutionary_search:
          best_cands = self.feat_sampler.sample_from_set(step_candidates, active_search_width)
          # if environment.WORLD_SIZE > 0:
            #1. gather best cands from every process
            #2. Per rank, call again sample from set for the gathered stuff.
            #3. Replace best cands with the new stuff
          l.logger().info("Top-{} ({} unique) samples of generation {}: {}".format(active_search_width, len(best_cands), feeds[0].gen_id, ', '.join([str(round(c.score, 3)) for c in best_cands])))
        elif feeds[0].gen_id == 0:
          best_cands = self.feat_sampler.sample_from_set(step_candidates, active_search_width)
          l.logger().info("Starting scores: {}".format(', '.join([str(round(c.score, 3)) for c in best_cands])))
        else:
          # If nothing was found, there are no best cands, and we will keep searching.
          if not better_found:
            best_cands = []
            l.logger().warn("No better candidate found...")
          else:
            # Otherwise, this single input feed, provides a new single better sample.
            best_cands = [better_found]

        # Monitor the new better candidate(s), if any.
        if best_cands:
          self.candidate_monitor.register(
            {str(best_cands[0].sample_feed.gen_id): [c.score for c in best_cands]}
          )
          self.candidate_monitor.plot()
        # Add them back to queue and to active feed database.
        found_match = False
        for nc in best_cands:
          if FLAGS.evolutionary_search:
            self.tsne_monitor.register((nc.features, "gen_{}_accepted".format(str(feeds[0].gen_id))))
          sample_hash = ''.join([str(x) for x in nc.sample])
          if FLAGS.evolutionary_search or (sample_hash not in total_cand_hash):
            if sample_hash not in total_cand_hash:
              total_cand.append(nc)
              total_cand_hash.add(sample_hash)
            if nc.score == 0.0 and FLAGS.evolutionary_search:
              found_match = True
            if not found_match and 1+nc.sample_feed.gen_id <= active_search_depth and (FLAGS.evolutionary_search or 0 < nc.score < feed.input_score):
              self.feed_queue.append(
                ActiveSampleFeed(
                  input_feed       = nc.sample,
                  input_features   = nc.features,
                  input_score      = nc.score,
                  gen_id           = 1 + nc.sample_feed.gen_id,
                )
              )
            self.addToDB(
              active_feed_database.ActiveFeed.FromArgs(
                tokenizer        = self.tokenizer,
                id               = self.active_db.active_count,
                input_feed       = nc.sample_feed.input_feed,
                input_features   = nc.sample_feed.input_features,
                sample           = nc.sample,
                output_features  = nc.features,
                sample_quality   = nc.score,
                target_benchmark = (self.feat_sampler.target_benchmark.name, self.feat_sampler.target_benchmark.contents),
                target_features  = self.feat_sampler.target_benchmark.features,
                compile_status   = True,
                generation_id    = nc.sample_feed.gen_id,
              )
            )
        self.tsne_monitor.plot()
        # save state and re-loop.
        self.saveCheckpoint()

      # Catch threads on last iteration.
      if write_cache_proc:
        write_cache_proc.join()
      if FLAGS.evaluate_candidates and write_eval_proc:
        write_eval_proc.join()

      ## Finished, save state, switch benchmark, return samples.
      self.bench_idx += 1
      self.saveCheckpoint()
      self.feat_sampler.iter_benchmark()
      return (np.repeat([org_inp], len(total_cand), axis = 0),
              np.repeat([org_ids], len(total_cand), axis = 0),
              [x.sample for x in total_cand],
              [[]] * len(total_cand))
    except KeyboardInterrupt:
      self.raised_keyboard_int = True
      if write_cache_proc:
        write_cache_proc.terminate()
      if FLAGS.evaluate_candidates and write_eval_proc:
        write_eval_proc.terminate()
      return (np.repeat([org_inp], len(total_cand), axis = 0),
              np.repeat([org_ids], len(total_cand), axis = 0),
              [x.sample for x in total_cand],
              [[]] * len(total_cand))
    except Exception as e:
      l.logger().error(e)
      self.raised_exception = e
      return (np.repeat([org_inp], len(total_cand), axis = 0),
              np.repeat([org_ids], len(total_cand), axis = 0),
              [x.sample for x in total_cand],
              [[]] * len(total_cand))

  def initOrGetQueue(self) -> np.array:
    """
    If feed queue is not initialized, initialize it by getting new datapoint.
    Otherwise, don't do anything as feed_queue is already loaded from checkpoint.
    Adds datapoint to InputFeed table of database.

    Returns:
      Starting input feed of sampling.
    """
    if not self.feed_queue:
      try:
        cf = next(self.loader).squeeze(0)
      except StopIteration:
        self.loader = iter(self.dataloader)
        cf = next(self.loader).squeeze(0)
      cf = [int(x) for x in cf]
      self.feed_queue.append(
        ActiveSampleFeed(
          input_feed     = cf,
          input_features = extractor.ExtractFeatures(self.tokenizer.ArrayToCode(cf), [self.feat_sampler.feature_space])[self.feat_sampler.feature_space],
          input_score    = math.inf,
          gen_id         = 0,
        )
      )
      if environment.WORLD_RANK == 0:
        self.addToDB(
          active_feed_database.ActiveInput.FromArgs(
            tokenizer      = self.tokenizer, id = self.active_db.input_count,
            input_feed     = cf, input_features = self.feed_queue[-1].input_features,
          )
        )
    l.logger().info("Feed queue input scores: {}".format(', '.join([str(round(c.input_score, 3)) for c in self.feed_queue])))
    return self.feed_queue[0].input_feed

  def collateInputData(self,
                       feed: typing.List[np.array],
                       wload_size: int,
                       sample_batch_per_feed: int,
                       ) -> typing.Dict[str, typing.TypeVar('torch.Tensor')]:
    """
    Create a full generation workload out of a sample feed.
    If feed is already masked, then just repeat it across the whole workload.
    If it is not masked, then feed is masked wload_size times.

    Args:
      feed: numpy array of input feed (expressed as list of a single np element),
            or a list of numpys in case multiple workloads are merged.
      wload_size: Number of inputs that will be fed to the model in a single workload.

    Returns:
      The tensor inputs dictionary filled for BERT.
    """
    if self.tokenizer.maskToken in feed[0] or self.tokenizer.holeToken in feed[0]:
      inputs = sequence_masking.MaskedSeqToBlob(
        feed[0], self.tokenizer,
        self.sampler.sequence_length,
        self.max_position_embeddings
      )
      inputs = {
        k: torch.from_numpy(v).unsqueeze(0).repeat_interleave(self.sample_batch_size, dim = 0).unsqueeze(0).repeat_interleave(wload_size, dim = 0) 
        for k, v in inputs.items()
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
                            dataload_worker, feed = feed,
                            func  = self.func, batch = self.sample_batch_size,
                            batch_per_feed = sample_batch_per_feed
                          ),range(wload_size)
                         ):
          if batch:
            # convert dict values from np -> torch.Tensor.
            out = {
              k: torch.from_numpy(v).unsqueeze(0)
              for (k, v) in batch[0].items()
            }
            for f in batch[1:]:
              for k, v in f.items():
                nt = torch.from_numpy(v).unsqueeze(0)
                out[k] = torch.cat((out[k], nt), 0)
            for k in inputs.keys():
              inputs[k].append(out[k])
        for k, v in inputs.items():
          s = torch.stack(v)
          inputs[k] = s.view(-1, self.sample_batch_size, s.shape[-1])
        pool.close()
        pool.terminate()
      except KeyboardInterrupt as e:
        pool.close()
        pool.terminate()
        raise e
    return inputs

  def registerOutputData(self,
                         outputs    : typing.Dict[str, typing.List[np.array]],
                         # rng        : typing.Tuple[int, int],
                         feeds      : ActiveSampleFeed,
                         candidates : typing.List[ActiveSample],
                         rejected_candidates: typing.List[ActiveSample],
                         bar: tqdm.tqdm,
                         ) -> typing.List[int]:
    """
    Gets workload output from model.
    In parallel, every sample is checked for compilability and features are extracted.
    If sample compiles, it is stored as an active learning candidate.

    Args:
      outputs: Dictionary output of workload
      candidates: Passed by reference and filled within this function
      bar: tqdm bar for status checking

    Returns:
      cm_rate: List of two elements that express compilation rate of workload.
               0th el: Total compiling.
               1st el: Total samples.
    """
    cm_rate = [0, 0]
    # l.logger().warn("Opening pool")
    pool = multiprocessing.Pool()
    cm_rate[1] += len(outputs['generated_samples'])
    better_found = None
    try:
      it = zip(
        outputs['generated_samples'], outputs['sample_indices'],
        outputs['input_ids'], outputs['masked_lm_lengths'],
        feeds
      )
      if self.feat_sampler.feature_space != "GreweFeatures":
        candidate_worker = functools.partial(
          IR_candidate_worker, tokenizer = self.tokenizer, feat_sampler = self.feat_sampler,
        )
      else:
        candidate_worker = functools.partial(
          text_candidate_worker, tokenizer = self.tokenizer, feat_sampler = self.feat_sampler,
        )
      t = 0
      # l.logger().warn("Pool opened")
      for idx, batch in bar(enumerate(pool.map(candidate_worker, it))):
        t = idx
        if batch[0]:
          cm_rate[0] += 1
          candidates.append(batch[1])
          if 0 < batch[1].score < batch[1].sample_feed.input_score:
            if better_found is None or batch[1].score < better_found.score:
              better_found = batch[1]
        else:
          if FLAGS.evaluate_candidates:
            rejected_candidates.append(batch[1])

      if FLAGS.features_standard_scaler:
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit([[float(y) for y in x.features.values()] for x in candidates + [self.feat_sampler.target_benchmark]])
        target_feats = {k: v for k, v in zip(self.feat_sampler.target_benchmark.features.keys(), scaler.transform([[float(x) for x in self.feat_sampler.target_benchmark.features.values()]])[0])}
        for idx, cd in enumerate(candidates):
          outfeats = {k: v for k, v in zip(cd.features.keys(), scaler.transform([[float(x) for x in cd.features.values()]])[0])}
          candidates[idx]._replace(score = feature_sampler.calculate_distance(outfeats, target_feats, self.feat_sampler.feature_space))

      pool.close()
      pool.terminate()
    except KeyboardInterrupt as e:
      pool.close()
      pool.terminate()
      raise e
    return cm_rate, better_found

  def saveCheckpoint(self):
    """
    Save feed queue checkpoint for easy restart.
    """
    with open(self.sampler.corpus_directory / "gen_state.pkl", 'wb') as outf:
      pickle.dump({'feed_queue': self.feed_queue, 'bench_idx': self.bench_idx}, outf)
    self.candidate_monitor.saveCheckpoint()
    self.tsne_monitor.saveCheckpoint()
    self.comp_rate_mon.saveCheckpoint()
    self.exec_time_mon.saveCheckpoint()
    return

  def loadCheckpoint(self):
    """
    Load checkpointed feed queue, if exists.
    """
    if (self.sampler.corpus_directory / "gen_state.pkl").exists():
      with open(self.sampler.corpus_directory / "gen_state.pkl", 'rb') as infile:
        checkpoint = pickle.load(infile)
        self.feed_queue = checkpoint['feed_queue']
        self.bench_idx  = checkpoint['bench_idx']
    else:
      self.feed_queue = []
      self.bench_idx  = 1
    return

  def addToDB(self,
              db_input: typing.Union[
                          active_feed_database.ActiveSamplingSpecs,
                          active_feed_database.ActiveInput,
                          active_feed_database.ActiveFeed
                        ]
              ) -> None:
    """
    If not exists, add current sample state to database
    """
    with self.active_db.Session(commit = True) as session:
      exists = session.query(
        type(db_input)
      ).filter(type(db_input).sha256 == db_input.sha256).scalar() is not None
      if not exists:
        session.add(db_input)
    return

  def _saveCorpusRecord(self, masked_corpus: typing.Dict) -> None:
    """Converts corpus nparrays to torch tensors and stores corpus to pt_record"""

    torch.save(
      [{k: torch.from_numpy(v) for (k, v) in inst.items()} for inst in masked_corpus['corpus']],
      masked_corpus['file']
    )
    if FLAGS.write_text_dataset:
      with open(masked_corpus['txt'], 'w') as file_writer:
        for instance in masked_corpus['corpus']:
          file_writer.write("'seen_in_training': {}\n'original_input': {}\n'input_ids': {}\n'input_mask': {}\n'position_ids': {}\n'mask_labels': {}\n'masked_lm_lengths': {}\n'next_sentence_labels': {}\n\n"
                              .format((True if instance['seen_in_training'] == 1 else False),
                                      self.tokenizer.tokensToString(instance['original_input'], ignore_token = self.tokenizer.padToken),
                                      self.tokenizer.tokensToString(instance['input_ids'],      ignore_token = self.tokenizer.padToken),
                                      instance['input_mask'],
                                      instance['position_ids'],
                                      instance['mask_labels'],
                                      instance['masked_lm_lengths'],
                                      instance['next_sentence_labels']
                                    )
                              )
    l.logger().info("Wrote {} instances ({} batches of {} datapoints) to {}"
                 .format(len(masked_corpus['corpus']), self.steps_per_epoch, self.training_opts.batch_size, masked_corpus['file']))
    return
