"""
Feature space sampling of source code.
"""
import typing
import pathlib
import pickle
import math
import functools
# import multiprocessing
import time
import numpy as np
from numpy.random import default_rng

from deeplearning.clgen.features import normalizers
from deeplearning.clgen.corpuses import corpuses
from deeplearning.clgen.corpuses import benchmarks
from deeplearning.clgen.util import logging as l
from deeplearning.clgen.util import distrib
from deeplearning.clgen.util import environment

from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer(
  "randomize_selection",
  None,
  "Debugging integer flag that abolishes euclidean distance priority and picks X randomly selected elements to return as top-X."
)

def grid_walk_generator(feature_space: str) -> typing.Iterator[typing.Dict[str, float]]:
  """
  Walk through feature space and generate
  target feature instances to approximate.
  """
  step_size = 100
  target = normalizers.normalizer[feature_space]
  for i in range(1, step_size+1):
    ct = {}
    for k in target.keys():
      if k != "F2:coalesced/mem" and k != "F4:comp/mem":
        ct[k] = int(i * (target[k] / step_size))
    if "F2:coalesced/mem" in ct:
      ct["F2:coalesced/mem"] = ct["coalesced"] / max(1, ct["mem"])
    if "F4:comp/mem" in ct:
      ct["F4:comp/mem"] = ct["comp"] / max(1, ct["mem"])
    yield ct

def calculate_distance(infeat: typing.Dict[str, float],
                       tarfeat: typing.Dict[str, float],
                       feature_space: str,
                       ) -> float:
  """
  Euclidean distance between sample feature vector
  and current target benchmark.
  """
  d = 0
  for key in tarfeat.keys():
    n = 1.0 # tarfeat[key] if tarfeat[key] != 0 and key != "F2:coalesced/mem" and key != "F4:comp/mem" else 1# normalizers.normalizer[feature_space][key]
    i = float(infeat[key]) / n
    t = float(tarfeat[key]) / n
    d += abs((t**2) - (i**2))
  return math.sqrt(d)

class Benchmark(typing.NamedTuple):
  path             : pathlib.Path
  name             : str
  contents         : str
  features         : typing.Dict[str, float]
  runtime_features : typing.Dict[str, float]

class FeatureSampler(object):
  """
  Abstract class for sampling features.
  """
  @property
  def is_active(self):
    return False

  def __init__(self,
               workspace     : pathlib.Path,
               feature_space : str,
               target        : str,
               ):
    self.workspace        = workspace
    self.feature_space    = feature_space
    self.target           = target
    self.benchmarks       = []
    self.target_benchmark = None
    return

  def calculate_distance(self, infeat: typing.Dict[str, float]) -> float:
    """
    Euclidean distance between sample feature vector
    and current target benchmark.
    """
    return calculate_distance(infeat, self.target_benchmark.features, self.feature_space)

  def topK_candidates(self,
                      candidates   : typing.List[typing.TypeVar("ActiveSample")],
                      K            : int,
                      dropout_prob : float,
                      ) -> typing.List[typing.TypeVar("ActiveSample")]:
    """
    Return top-K candidates.
    """
    if FLAGS.randomize_selection is None:
      sorted_cands = sorted(candidates, key = lambda x: x.score)  # [:K]
      if dropout_prob > 0.0:
        rng = default_rng()
        # for kidx in range(max(K, len(sorted_cands))): # if K > len(sorted_cands) because your LM's compilation rate sucks, this will give you an IndexError.
        for kidx in range(min(K, len(sorted_cands))): # if K > len(sorted_cands) because your LM's compilation rate sucks, this will give you an IndexError.
          rep = np.random.RandomState().rand()
          visited = set()
          if rep <= dropout_prob and len(visited) < len(sorted_cands):
            swap_idx = rng.choice(list(set(range(len(sorted_cands))) - visited))
            sorted_cands[kidx], sorted_cands[swap_idx] = sorted_cands[swap_idx], sorted_cands[kidx]
            visited.add(swap_idx)
      return sorted_cands[:K]
    else:
      if FLAGS.randomize_selection == 0:
        raise ValueError("randomize_selection, {}, cannot be 0.".format(FLAGS.randomize_selection))
      l.logger().warn("Randomized generation selection has been activated. You must know what you are doing!")
      kf = min(FLAGS.randomize_selection, len(candidates))
      rng = default_rng()
      indices = set(rng.choice(len(candidates), size = kf, replace = False))
      return [c for idx, c in enumerate(candidates) if idx in indices]

  def sample_from_set(self, 
                      candidates   : typing.List[typing.TypeVar("ActiveSample")],
                      search_width : int,
                      dropout_prob : float,
                      only_unique  : bool = True,
                      ) -> bool:
    """
    Find top K candidates by getting minimum
    euclidean distance from set of rodinia benchmarks.
    """
    """
    for idx in range(len(candidates)):
      candidates[idx] = candidates[idx]._replace(
        score = self.calculate_distance(candidates[idx].features)
      )
    """
    if only_unique:
      hset = set()
      unique_candidates = []
      for c in candidates:
        sample_str = ','.join([str(x) for x in c.sample])
        if sample_str not in hset:
          unique_candidates.append(c)
          hset.add(sample_str)
      candidates = unique_candidates
    return self.topK_candidates(candidates, search_width, dropout_prob = dropout_prob)

  def step_generation(self, candidates: typing.List['ActiveSample']) -> None:
    """
    End of LM generation's epoch hook.
    """
    return

  def iter_benchmark(self, *unused_args, **unused_kwargs) -> None:
    """
    Override this method to set how new parts of the feature space are going to
    be targetted.
    """
    raise NotImplementedError("Abstract class.")

  def is_terminated(self) -> bool:
    raise NotImplementedError

  def saveCheckpoint(self) -> None:
    """
    Save feature sampler state.
    """
    state_dict = {
      'benchmarks'       : self.benchmarks,
      'target_benchmark' : self.target_benchmark,
    }
    if environment.WORLD_RANK == 0:
      with open(self.workspace / "feature_sampler_state.pkl", 'wb') as outf:
        pickle.dump(state_dict, outf)
    return

  def loadCheckpoint(self) -> None:
    """
    Override to select checkpoints are going to be loaded.
    """
    raise NotImplementedError("Abstract class.")

class BenchmarkSampler(FeatureSampler):
  """
  This is a shitty experimental class to work with benchmark comparison.
  Will be refactored obviously.
  """
  def __init__(self,
               workspace     : pathlib.Path,
               feature_space : str,
               target        : str,
               git_corpus    : corpuses.Corpus = None,
               ):
    super(BenchmarkSampler, self).__init__(workspace, feature_space, target)
    if self.target  != "grid_walk":
      self.path        = pathlib.Path(benchmarks.targets[target]).resolve()
    self.reduced_git_corpus = [
      (cf, feats[self.feature_space])
      for cf, feats in git_corpus.getFeaturesContents(sequence_length = 768)
      if self.feature_space in feats and feats[self.feature_space]
    ]
    self.loadCheckpoint()
    try:
      if self.target_benchmark is None:
        self.benchmarks.pop(0)
        self.target_benchmark = self.benchmarks.pop(0)
        # l.logger().info("Target benchmark: {}\nTarget fetures: {}".format(self.target_benchmark.name, self.target_benchmark.features))
    except IndexError:
      self.target_benchmark = None
    return

  def iter_benchmark(self, *unused_args, **unused_kwargs):
    """
    When it's time, cycle through the next target benchmark.
    """
    # self.benchmarks.append(self.benchmarks.pop(0))
    del unused_args
    del unused_kwargs
    try:
      self.target_benchmark = self.benchmarks.pop(0)
      # l.logger().info("Target benchmark: {}\nTarget fetures: {}".format(self.target_benchmark.name, self.target_benchmark.features))
    except IndexError:
      self.target_benchmark = None
    self.saveCheckpoint()
    return

  def is_terminated(self) -> bool:
    if not self.target_benchmark:
      return True
    return False

  def loadCheckpoint(self) -> None:
    """
    Load feature sampler state.
    """
    if (self.workspace / "feature_sampler_state.pkl").exists():
      distrib.lock()
      with open(self.workspace / "feature_sampler_state.pkl", 'rb') as infile:
        state_dict = pickle.load(infile)
      distrib.unlock()
      self.benchmarks       = state_dict['benchmarks']
      self.target_benchmark = state_dict['target_benchmark']
    else:
      self.benchmarks = []
      self.target_benchmark = None
      if self.target == "grid_walk":
        for target_features in grid_walk_generator(self.feature_space):
          self.benchmarks.append(
            Benchmark(
              "",
              "",
              "",
              target_features,
              {}
            )
          )
        self.saveCheckpoint()
      else:
        if environment.WORLD_RANK == 0:
          kernels = benchmarks.yield_cl_kernels(self.path)
          # pool = multiprocessing.Pool()
          # for benchmark in pool.map(
          #                   functools.partial(
          #                     benchmarks.benchmark_worker,
          #                     feature_space = self.feature_space,
          #                     reduced_git_corpus = self.reduced_git_corpus
          #                   ), kernels
          #                 ):
          #   if benchmark:
          #     self.benchmarks.append(benchmark)
          # pool.close()
          for kernel in kernels:
            benchmark = benchmarks.benchmark_worker(kernel, self.feature_space, self.reduced_git_corpus)
            if benchmark:
              self.benchmarks.append(benchmark)
          benchmarks.resolve_benchmark_names(self.benchmarks)
          self.benchmarks = sorted(self.benchmarks, key = lambda b: b.name)
          distrib.write_broadcast(pickle.dumps(self.benchmarks), is_bytes = True)
        else:
          self.benchmarks = pickle.loads(distrib.read_broadcast(is_bytes = True))
        distrib.barrier()
    l.logger().info("Loaded {}, {} benchmarks".format(self.target, len(self.benchmarks)))
    l.logger().info(', '.join([x for x in set([x.name for x in self.benchmarks])]))
    return

class ActiveSampler(FeatureSampler):
  """
  Euclidean distance-based feature space sampler for active learning.
  The downstream task and active learner are encapsulated.
  This class is the API between the language model's searching method/generation
  and the active learner's query by committee.
  """
  @property
  def is_active(self):
    return True

  def __init__(self,
               workspace      : pathlib.Path,
               feature_space  : str,
               active_learner : 'active_models.Model',
               tokenizer      : 'tokenizers.TokenizerBase',
               ):
    super(ActiveSampler, self).__init__(workspace, feature_space, str(active_learner.downstream_task))
    self.active_learner = active_learner
    self.loadCheckpoint()
    try:
      if self.target_benchmark is None:
        self.target_benchmark = self.benchmarks.pop(0)
        # l.logger().info("Target benchmark: {}\nTarget fetures: {}".format(self.target_benchmark.name, self.target_benchmark.features))
    except IndexError:
      self.benchmarks = self.sample_active_learner()
      self.target_benchmark = self.benchmarks.pop(0)
    self.tokenizer = tokenizer
    self.saveCheckpoint()
    return

  def sample_active_learner(self,
                            keep_top_k  : int = 1,
                            num_samples : int = 16384,
                            ) -> typing.List[Benchmark]:
    """
    Sample active learner for num_samples and sort by highest entropy.
    """
    return [
      Benchmark("", "", "", sample['static_features'], sample['runtime_features'])
      for sample in self.active_learner.Sample(num_samples = num_samples)
    ][:keep_top_k]

  def teach_active_learner(self,
                           target_samples: typing.List['ActiveSample'],
                           top_k: int
                           ) -> None:
    """
    Update active learner with targetted generated samples by the language model.
    """
    if FLAGS.disable_active_learning:
      return
    upd_samples, upd_loader = self.active_learner.downstream_task.UpdateDataGenerator(target_samples, top_k, self.tokenizer)
    if len(upd_loader) > 0:
      self.active_learner.UpdateLearn(upd_loader)
      self.active_learner.downstream_task.UpdateTrainDataset(upd_loader)
    else:
      l.logger().info("Skipping empty update dataset.")
    return

  def step_generation(self, candidates: typing.List['ActiveSample']) -> None:
    """
    End of LM generation's epoch hook.
    """
    self.active_learner.downstream_task.step_generation(candidates)
    return

  def iter_benchmark(self,
                     target_samples: typing.List['ActiveSample'] = None,
                     top_k: int = -1
                     ) -> None:
    """
    Set the next item from list to target.
    If doesn't exist, ask from the active learner for new stuff,
    unless a termination criteria has been met.

    At the first iteration, target samples is None.
    But when the LM generates targets and calls for iter_benchmark()
    it provides the generated samples and gives them to the active_learner
    for updated learning.
    """
    if target_samples:
      self.teach_active_learner(target_samples, top_k)
    try:
      self.target_benchmark = self.benchmarks.pop(0)
      # l.logger().info("Target fetures: {}".format(self.target_benchmark.features))
    except IndexError:
      l.logger().warn("Implement a termination criteria here.")
      self.benchmarks = self.sample_active_learner()
      self.iter_benchmark()
      return
    self.saveCheckpoint()
    return

  def is_terminated(self) -> bool:
    l.logger().warn("You need to find a termination criteria for the active learner.")
    return False

  def saveCheckpoint(self) -> None:
    if environment.WORLD_RANK == 0:
      super(ActiveSampler, self).saveCheckpoint()
    distrib.barrier()
    return

  def loadCheckpoint(self) -> None:
    """
    Load pickled list of benchmarks, if exists.
    Otherwise, ask the first batch of features from the active learner.
    """
    if (self.workspace / "feature_sampler_state.pkl").exists():
      distrib.lock()
      try:
        with open(self.workspace / "feature_sampler_state.pkl", 'rb') as infile:
          state_dict = pickle.load(infile)
          infile.close()
        self.benchmarks       = state_dict['benchmarks']
        self.target_benchmark = state_dict['target_benchmark']
        while not infile.closed:
          time.sleep(1)
      except EOFError:
        distrib.unlock()
        self.loadCheckpoint()
        return
      distrib.unlock()
    else:
      self.benchmarks = []
      self.target_benchmark = None
    l.logger().info("Loaded {}, {} benchmarks".format(self.target, len(self.benchmarks)))
    return
