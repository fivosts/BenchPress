"""This file defines the streaming generators for model training data.

We train models on overlapping one-hot encoded text sequences. For a corpus of
a reasonable size, the full training data may not fit in memory. This modules
provides Python Generator classes for use by a sequential Keras model's
fit_generator() method to stream batches of training data.
"""
import os
import typing
import glob
import humanize
import pickle
import functools
import numpy as np
import pathlib
import multiprocessing
import math
import progressbar

from deeplearning.clgen.util import pytorch
from deeplearning.clgen.util.pytorch import torch
from deeplearning.clgen.util import distributions
from deeplearning.clgen.util import monitors
from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen.features import extractor
from deeplearning.clgen.features import feature_sampler
from deeplearning.clgen.features import active_feed_database
from deeplearning.clgen.models import lm_data_generator
from deeplearning.clgen.models import sequence_masking
from deeplearning.clgen.models.torch_bert import datasets
from deeplearning.clgen.preprocessors import opencl
from absl import flags
from eupy.native import logger as l

torch.multiprocessing.set_sharing_strategy('file_system')

FLAGS = flags.FLAGS

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

def candidate_worker(sample_out   : typing.Dict[str, np.array],
                     feed         : np.array,
                     feat_sampler : feature_sampler.EuclideanSampler,
                     tokenizer    : typing.TypeVar('corpuses.tokenizers.TokenizerBase'),
                     ) -> ActiveSample:
  sample, indices, input_ids, masked_lm_lengths = sample_out
  try:
    code = tokenizer.ArrayToCode(sample, with_formatting = False)
    _ = opencl.Compile(code)
    features = extractor.ExtractFeatures(code, [feat_sampler.feature_space])[feat_sampler.feature_space]
    if features:
      return ActiveSample(
        sample_feed    = feed,      sample         = sample,
        input_ids      = input_ids, hole_instances = [x for x in masked_lm_lengths if x >= 0],
        sample_indices = indices,   features       = features,
        score          = feat_sampler.calculate_distance(features),
        timestep       = -1,
      )
      # return sample, indices, features, input_ids, masked_lm_lengths
  except ValueError:
    pass
  except Exception:
    pass
  return None

def dataload_worker(x    : int,
                    feed : np.array,
                    func : typing.TypeVar('sequence_masking.MaskingFunction'),
                    batch: int,
                    ) -> typing.Dict[str, torch.Tensor]:
  try:
    return {
      k: torch.from_numpy(v).unsqueeze(0).repeat_interleave(batch, dim = 0)
      for (k, v) in func(feed).items()
    }
  except Exception:
    return None

class torchLMDataGenerator(lm_data_generator.MaskLMDataGenerator):
  """Data generator subclass designed for PyTorch BERT model."""
  @classmethod
  def TrainMaskLMBatchGenerator(cls,
                               corpus: "corpuses.Corpus",
                               training_opts: model_pb2.TrainingOptions,
                               cache_path,
                               num_train_steps: int = None,
                               pre_train: bool = False,
                               ) -> "data_generator.MaskLMBatchGenerator":
    """Initializes data generator for training."""
    d = super(torchLMDataGenerator, torchLMDataGenerator()).TrainMaskLMBatchGenerator(
                corpus, training_opts, cache_path, num_train_steps, pre_train
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
                                 ) -> "data_generator.MaskLMBatchGenerator":
    """Initializes data generator for inference."""
    d = super(torchLMDataGenerator, torchLMDataGenerator()).SampleMaskLMBatchGenerator(
              model_opts, sampler, tokenizer, seed,
              sample_batch_size, max_position_embeddings, cache_path
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
      d.feat_sampler      = feature_sampler.EuclideanSampler(
        d.sampler.corpus_directory,
        corpus_config.active.feature_space,
        corpus_config.active.target
      )
      d.candidate_monitor = monitors.CategoricalDistribMonitor(
        d.sampler.corpus_directory, "feature_distance"
      )
      d.comp_rate_mon     = monitors.CategoricalHistoryMonitor(
        d.sampler.corpus_directory, "comp_rate_per_gen"
      )
      # Store unique specs to database once.
      d.addToDB(
        active_feed_database.ActiveSamplingSpecs.FromArgs(
          act_l_pf   = corpus_config.active.active_limit_per_feed,
          act_s_dep  = corpus_config.active.active_search_depth,
          act_s_wid  = corpus_config.active.active_search_width,
          feat_space = corpus_config.active.feature_space
        )
      )

    d.dataloader = d.predict_dataloader()
    d.loader     = iter(d.dataloader)
    return d

  def __init__(self):
    super(torchLMDataGenerator, self).__init__("pt_record")
    self.dataloader = None
    ## Active learning attributes initialization.
    self.loader     = None
    self.comp_rate  = {}
    self.feed_queue = []
    self.feat_sampler      = None
    self.candidate_monitor = None
    self.comp_rate_mon     = None
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
        sampler = torch.utils.data.RandomSampler(dataset, replacement = False)
    else:
      raise ValueError(self.config.datapoint_time)

    dataloader = torch.utils.data.dataloader.DataLoader(
      dataset    = dataset,
      batch_size = self.training_opts.batch_size,
      sampler    = (sampler
        if not pytorch.torch_tpu_available or pytorch.torch_xla.xrt_world_size() <= 1
        else torch.utils.data.distributed.DistributedSampler(
          dataset,
          num_replicas = pytorch.torch_xla.xrt_world_size(),
          rank = pytorch.torch_xla.get_ordinal()
        )
      ),
      num_workers = 0,
      drop_last   = False,
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
          sampler = torch.utils.data.RandomSampler(dataset, replacement = False)
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
      sampler    = (
            sampler
            if not pytorch.torch_tpu_available or pytorch.torch_xla.xrt_world_size() <= 1
            else torch.utils.data.distributed.DistributedSampler(
                  dataset, 
                  num_replicas = pytorch.torch_xla.xrt_world_size(), 
                  rank = pytorch.torch_xla.get_ordinal()
                 )
            ),
      num_workers = 0,
      drop_last   = False,
      )
    return dataloader

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
    if not self.feat_sampler.target_benchmark:
      raise StopIteration
    # Active sampling specs initialization
    active_limit_per_feed = self.sampler.config.sample_corpus.corpus_config.active.active_limit_per_feed
    active_search_depth = self.sampler.config.sample_corpus.corpus_config.active.active_search_depth
    active_search_width = self.sampler.config.sample_corpus.corpus_config.active.active_search_width

    # Initialize feed queue
    org_inp, org_ids = self.initOrGetQueue()
    total_cand, total_cand_hash = [], set()

    while self.feed_queue:

      feed = self.feed_queue.pop(0)
      rem  = active_limit_per_feed // self.sample_batch_size
      step_candidates = []
      cmp_rate        = [0, 0]

      self.sampler.setStartText(self.tokenizer.tokensToString(feed.input_feed, ignore_token = self.tokenizer.padToken))
      self.sampler.Specialize(self.tokenizer)

      bar = progressbar.ProgressBar(max_value = active_limit_per_feed)
      bar.update(0)

      if feed.gen_id not in self.comp_rate:
        self.comp_rate[feed.gen_id] = [0, 0]

      # Iterate until you get the required amount of candidates
      while len(step_candidates) < active_limit_per_feed:
        # Pre-process inputs
        inputs = self.collateInputData(feed.input_feed, rem)
        # Infer
        outputs = mwrapper.sample_model_step(
          estimator.models, estimator.devices, inputs,
        )
        # Post-process outputs.
        tcs, ts = self.registerOutputData(outputs, feed, step_candidates, bar)
        cmp_rate[0] += tcs
        cmp_rate[1] += ts
        # Calculate how many more to infer.
        try:
          rcands = active_limit_per_feed - len(step_candidates)
          crate  = cmp_rate[0] / cmp_rate[1]
          rem = max(2, int((rcands // self.sample_batch_size) / crate))
        except ZeroDivisionError:
          pass

      self.comp_rate[feed.gen_id] = [sum(x) for x in zip(self.comp_rate[feed.gen_id], cmp_rate)]
      self.comp_rate_mon.register((feed.gen_id, self.comp_rate[feed.gen_id][0] / self.comp_rate[feed.gen_id][1]))
      self.comp_rate_mon.plot()

      # Top-k candidates of ith generation.
      best_cands = self.feat_sampler.sample_from_set(step_candidates, active_search_width)

      if feed.gen_id > 0:
        best_cands = best_cands[:1]

      self.candidate_monitor.register(
        {str(best_cands[0].sample_feed.gen_id): [c.score for c in best_cands]}
      )
      self.candidate_monitor.plot()
      for nc in best_cands:
        sample_hash = ''.join([str(x) for x in nc.sample])
        if sample_hash not in total_cand_hash:
          total_cand.append(nc)
          total_cand_hash.add(sample_hash)
          if 0 < nc.score < feed.input_score and 1+nc.sample_feed.gen_id <= active_search_depth:
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
              masked_input_ids = nc.input_ids,
              hole_instances   = nc.hole_instances,
              sample           = nc.sample,
              output_features  = nc.features,
              sample_quality   = nc.score,
              target_benchmark = (self.feat_sampler.target_benchmark.name, self.feat_sampler.target_benchmark.contents),
              target_features  = self.feat_sampler.target_benchmark.feature_vector,
              compile_status   = True,
              generation_id    = nc.sample_feed.gen_id,
              timestep         = nc.timestep,
            )
          )
      self.saveCheckpoint()

    self.saveCheckpoint()
    self.feat_sampler.iter_benchmark()
    return (np.repeat([org_inp], len(total_cand), axis = 0),
            np.repeat([org_ids], len(total_cand), axis = 0),
            [x.sample for x in total_cand],
            [x.sample_indices for x in total_cand])

  def initOrGetQueue(self) -> int:
    """
    If feed queue is not initialized, nitialize it by getting new datapoint.
    Adds datapoint to InputFeed table of database.

    Returns:
      generation_id
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
          input_features = list(extractor.ExtractFeatures(self.tokenizer.ArrayToCode(cf), [self.feat_sampler.feature_space]).values())[0],
          input_score    = math.inf,
          gen_id         = 0,
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
                            dataload_worker, feed  = feed,
                            func  = self.func, batch = self.sample_batch_size
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
                         outputs    : typing.Dict[str, typing.List[np.array]],
                         feed       : ActiveSampleFeed,
                         candidates : typing.List[ActiveSample],
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
                     functools.partial(
                       candidate_worker,
                       feed         = feed,
                       tokenizer    = self.tokenizer,
                       feat_sampler = self.feat_sampler, 
                     ), it
                   ):
        if batch is not None:
          cm_rate[0] += 1
          bar.update(min(bar.max_value, len(candidates)))
          candidates.append(batch)
      pool.close()
    except KeyboardInterrupt as e:
      pool.close()
      pool.terminate()
      raise e
    return cm_rate

  def saveCheckpoint(self):
    """
    Save feed queue checkpoint for easy restart.
    """
    with open(self.sampler.corpus_directory / "gen_state.pkl", 'wb') as outf:
      pickle.dump(self.feed_queue, outf)
    return

  def loadCheckpoint(self):
    """
    Load checkpointed feed queue, if exists.
    """
    if (self.sampler.corpus_directory / "gen_state.pkl").exists():
      with open(self.sampler.corpus_directory / "gen_state.pkl", 'rb') as infile:
        self.feed_queue = pickle.load(infile)
    else:
      self.feed_queue = []
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
    l.getLogger().info("Wrote {} instances ({} batches of {} datapoints) to {}"
                 .format(len(masked_corpus['corpus']), self.steps_per_epoch, self.training_opts.batch_size, masked_corpus['file']))
    return
