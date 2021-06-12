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
      # self.comp_rate_per_gen[cur_feed.gen_id][0] += 1
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
      if self.feed_queue:
        # This will be triggered only if there is a checkpoint state (gen_state.pkl)
        self.current_generation = self.feed_queue[0].gen_id
        yield self.data_generator.toTensorFormat(self.feed_queue[0].input_blob)
      """Model will ask with next(). As long as it asks, this loop will bring."""
      self.current_generation = 0
      seed = next(self.active_dataset)
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
        ActiveSampleFeed(
          input_feed       = seed,
          input_features   = extractor.DictKernelFeatures(self.tokenizer.ArrayToCode(seed)),
          input_score      = math.inf,
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
    if current_feed.gen_id not in self.comp_rate_per_gen:
      self.comp_rate_per_gen[current_feed.gen_id] = [0, len(samples)]
    else:
      self.comp_rate_per_gen[current_feed.gen_id][1] += len(samples)
    for sample, indices in zip(samples, sample_indices):
      try:
        # If you can extract features from a sample, store it as a candidate.
        code = self.tokenizer.ArrayToCode(sample, with_formatting = False)
        _ = opencl.Compile(code)
        features = extractor.DictKernelFeatures(code)
        if features:
          self.comp_rate_per_gen[current_feed.gen_id][0] += 1
          self.step_candidates.append(
            ActiveSample(
              sample_feed    = current_feed,
              sample         = sample,
              sample_indices = indices,
              features       = features,
              score          = None,
              timestep       = 1 + len(self.step_candidates),
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
        ActiveSampleFeed(
          input_feed       = current_feed.input_feed,
          input_features   = current_feed.input_features,
          input_score      = math.inf,
          input_blob       = input_feed,
          masked_input_ids = input_feed['input_ids'],
          hole_instances   = [x for x in input_feed['masked_lm_lengths'] if x >= 0],
          gen_id           = current_feed.gen_id,
        )
      )
      self.bar.update(len(self.step_candidates))
      return self.data_generator.toTensorFormat(input_feed), [], False

    self.comp_rate_monitor.register((current_feed.gen_id, self.comp_rate_per_gen[current_feed.gen_id][0] / self.comp_rate_per_gen[current_feed.gen_id][1]))
    self.comp_rate_monitor.plot()
    # Re-init bar for next candidate gathering
    self.bar = progressbar.ProgressBar(max_value = FLAGS.active_limit_per_feed)

    # total_candidates contains all candidates from all generations from a single starting feed
    # that succeeded the distance sampling. candidate_idx sets a checkpoint of which are old and
    # which are new. This is to avoid re-storing old total_candidates multiple times.
    candidate_idx = len(self.total_candidates)
    # Top-k candidates of ith generation.
    new_candidates = self.feat_sampler.sample_from_set(self.step_candidates)

    if current_feed.gen_id > 0:
      new_candidates = new_candidates[:1]

    self.candidate_monitor.register(
      {str(new_candidates[0].sample_feed.gen_id): [c.score for c in new_candidates]}
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
      # _, dloc = opencl.Compile(self.tokenizer.ArrayToCode(candidate.sample), return_diagnostics = True)
      compile_status = True
      # if dloc:
        # compile_status = False
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
          timestep         = candidate.timestep,
        )
      )
      # If the current generation has not gone as deep as required, mask each new candidate
      # and place it at the tail of the sample feed queue.
      if 1 + candidate.sample_feed.gen_id <= FLAGS.active_search_depth:
        if not self.data_generator.config.use_start_end or (self.data_generator.config.use_start_end and self.tokenizer.endToken in candidate.sample):
          input_feed = self.func(candidate.sample)
          if len(input_feed['input_ids']) <= self.data_generator.sampler.sequence_length and 0 < candidate.score < current_feed.input_score:
            self.feed_queue.append(
              ActiveSampleFeed(
                input_feed       = candidate.sample,
                input_features   = candidate.features,
                input_score      = candidate.score,
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
      # But first, do a beam search on next generation, otherwise things will explode pretty quickly.
      next_gen = []
      while self.feed_queue and self.feed_queue[0].gen_id == self.current_generation + 1:
        next_gen.append(self.feed_queue.pop(0))
      next_gen = sorted(next_gen, key = lambda k: k.input_score)[:FLAGS.active_search_width]
      self.feed_queue = next_gen + self.feed_queue
      self.current_generation = self.feed_queue[0].gen_id

      # Update generation state pickle to start over.
      with open(self.data_generator.sampler.corpus_directory / "gen_state.pkl", 'wb') as outf:
        pickle.dump(self.feed_queue, outf)

      return self.data_generator.toTensorFormat(self.feed_queue[0].input_blob), [], False
    else:
      # We don't have new generation good candidates. Go get a new starting feed from the dataset.
      # Active learning will be killed (see True value in return statement), the harvested batch
      # will be returned to models. Models will re-init active learner for the next dataset input.
      active_batch   = [x.sample for x in self.total_candidates]
      active_indices = [x.sample_indices for x in self.total_candidates]
      self.total_candidates      = []
      self.total_candidates_hash = set()
      self.feat_sampler.iter_benchmark()
      # self.candidate_monitor   = monitors.HistoryMonitor(
      #   self.data_generator.sampler.corpus_directory, "feature_distance"
      # )
      return active_batch, active_indices, True

  def ActiveGeneration(self,
                       mwrapper: typing.TypeVar('torch_bert.torchBert'),
                       estimator: typing.TypeVar('torch_bert.SampleBertEstimator')
                      ) -> typing.Tuple[np.array, np.array, np.array, np.array]:
    """
    Get text.
    """
    try:
      if self.feed_queue:
        current_generation = self.feed_queue[0].gen_id
      else:
        current_generation = 0
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
    except StopIteration:
      return [], [], [], []

    original_input, org_input_ids = self.feed_queue[0].input_feed, self.feed_queue[0].input_feed

    total_candidates, total_candidates_hash = [], set()

    while self.feed_queue:

      cur_feed  = self.feed_queue.pop(0)
      self.sampler.SetStartText(self.tokenizer.tokensToString(cur_feed.input_feed, ignore_token = self.tokenizr.padToken))
      self.sampler.Specialize(self.tokenizer)
      step_candidates = []
      init_mask = True if self.tokenizer.maskToken in cur_feed.input_feed or self.tokenizer.holeToken in cur_feed.input_feed else False
      bar = progressbar.ProgressBar(max_value = FLAGS.active_limit_per_feed)
      bar.update(0)
      rem = FLAGS.active_limit_per_feed // self.data_generator.sample_batch_size

      cur_comp_rate = [0, 0]
      if cur_feed.gen_id not in self.comp_rate_per_gen:
        self.comp_rate_per_gen[cur_feed.gen_id] = [0, 0]

      if init_mask:
        input_feed = sequence_masking.MaskedSeqToBlob(
          cur_feed.input_feed, self.tokenizer,
          self.data_generator.sampler.sequence_length,
          self.data_generator.max_position_embeddings
        )
        inputs = self.data_generator.toTensorFormat(input_feed, batch = self.data_generator.sample_batch_size)
        inputs = {k: v.unsqueeze(0).repeat_interleave(rem, dim = 0) for k, v in inputs.items()}

      while len(step_candidates) < FLAGS.active_limit_per_feed:
        """
        Replicate and mask it for the full workload.
        """
        if init_mask:
          if rem > len(inputs['input_ids']):
            inputs = {k: v.repeat_interleave(rem // len(inputs['input_ids']), dim = 0) for k, v in inputs.items()}
          else:
            inputs = {k: v[:rem] for k, v in inputs.items()}
        else:
          inputs = {
            'input_ids': [], 'input_mask': [], 'position_ids': [],
            'mask_labels': [], 'masked_lm_lengths': [], 'next_sentence_labels': []
          }
          try:
            pool = multiprocessing.Pool()
            for batch in pool.imap_unordered(
                              functools.partial(
                                dataload_worker, feed  = cur_feed.input_feed,
                                func  = self.func, batch = self.data_generator.sample_batch_size
                              ),range(rem)
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
        outputs = mwrapper.sample_model_step(
          estimator.models, estimator.devices, inputs,
        )
        pool = multiprocessing.Pool()
        cur_comp_rate[1] += len(outputs['generated_samples'])
        try:
          for batch in pool.imap_unordered(
                         functools.partial(
                           candidate_worker, tokenizer = self.tokenizer
                         ), zip(
                              outputs['generated_samples'], outputs['sample_indices'],
                              outputs['input_ids'], outputs['masked_lm_lengths']
                            )
                       ):
            sample, indices, features, input_ids, masked_lm_lengths = batch
            if sample is not None:
              cur_comp_rate[0] += 1
              bar.update(min(FLAGS.active_limit_per_feed, len(step_candidates)))
              step_candidates.append(
                ActiveSample(
                  sample_feed    = cur_feed,  sample         = sample,
                  input_ids      = input_ids, hole_instances = [x for x in masked_lm_lengths if x >= 0],
                  sample_indices = indices,   features       = features,
                  score          = None,      timestep       = 1 + len(step_candidates),
                )
              )
          pool.close()
        except KeyboardInterrupt as e:
          pool.close()
          pool.terminate()
          raise e
        try:
          rem = max(
                  2,
                  int(
                    ((FLAGS.active_limit_per_feed - len(step_candidates)) // self.data_generator.sample_batch_size)
                    / (cur_comp_rate[0] / cur_comp_rate[1])
          ))
        except ZeroDivisionError:
          pass

      self.comp_rate_per_gen[cur_feed.gen_id] = [sum(x) for x in zip(self.comp_rate_per_gen[cur_feed.gen_id], cur_comp_rate)]
      self.comp_rate_monitor.register((cur_feed.gen_id, self.comp_rate_per_gen[cur_feed.gen_id][0] / self.comp_rate_per_gen[cur_feed.gen_id][1]))
      self.comp_rate_monitor.plot()

      # total_candidates contains all candidates from all generations from a single starting feed
      # that succeeded the distance sampling. candidate_idx sets a checkpoint of which are old and
      # which are new. This is to avoid re-storing old total_candidates multiple times.
      candidate_idx = len(total_candidates)
      # Top-k candidates of ith generation.
      new_candidates = self.feat_sampler.sample_from_set(step_candidates)

      if cur_feed.gen_id > 0:
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
        if 0 < candidate.score < cur_feed.input_score and 1+candidate.sample_feed.gen_id <= FLAGS.active_search_depth:
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
    return (np.repeat([original_input], len(active_batch), axis = 0),
            np.repeat([org_input_ids], len(active_batch), axis = 0),
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
