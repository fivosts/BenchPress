"""
Feature space sampling of source code.
"""
import typing
import pathlib
import pickle
import math
import functools
import multiprocessing
from numpy.random import default_rng

from deeplearning.clgen.features import normalizers
from deeplearning.clgen.corpuses import corpuses
from deeplearning.clgen.corpuses import benchmarks
from deeplearning.clgen.util import logging as l

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
    n = 1# tarfeat[key] if tarfeat[key] != 0 and key != "F2:coalesced/mem" and key != "F4:comp/mem" else 1# normalizers.normalizer[feature_space][key]
    i = infeat[key] / n
    t = tarfeat[key] / n
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
                      candidates: typing.List[typing.TypeVar("ActiveSample")],
                      K : int,
                      ) -> typing.List[typing.TypeVar("ActiveSample")]:
    """
    Return top-K candidates.
    """
    if FLAGS.randomize_selection is None:
      return sorted(candidates, key = lambda x: x.score)[:K]
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
    return self.topK_candidates(candidates, search_width)

  def iter_benchmark(self) -> None:
    """
    Override this method to set how new parts of the feature space are going to
    be targetted.
    """
    raise NotImplementedError("Abstract class.")

  def saveCheckpoint(self) -> None:
    """
    Save feature sampler state.
    """
    with open(self.workspace / "feature_sampler_state.pkl", 'wb') as outf:
      pickle.dump(self.benchmarks, outf)
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
      self.path        = pathlib.Path(targets[target]).resolve()
    self.reduced_git_corpus = [
      (cf, feats[self.feature_space])
      for cf, feats in git_corpus.getFeaturesContents(sequence_length = 768)
      if self.feature_space in feats and feats[self.feature_space]
    ]
    self.loadCheckpoint()
    try:
      self.target_benchmark = self.benchmarks.pop(0)
      l.logger().info("Target benchmark: {}\nTarget fetures: {}".format(self.target_benchmark.name, self.target_benchmark.features))
    except IndexError:
      self.target_benchmark = None
    return

  def iter_benchmark(self):
    """
    When it's time, cycle through the next target benchmark.
    """
    # self.benchmarks.append(self.benchmarks.pop(0))
    try:
      self.target_benchmark = self.benchmarks.pop(0)
      l.logger().info("Target benchmark: {}\nTarget fetures: {}".format(self.target_benchmark.name, self.target_benchmark.features))
    except IndexError:
      self.target_benchmark = None
    self.saveCheckpoint()
    return

  def loadCheckpoint(self) -> None:
    """
    Load feature sampler state.
    """
    if (self.workspace / "feature_sampler_state.pkl").exists():
      with open(self.workspace / "feature_sampler_state.pkl", 'rb') as infile:
        self.benchmarks = pickle.load(infile)
    else:
      self.benchmarks = []
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
        kernels = benchmarks.yield_cl_kernels(self.path)
        pool = multiprocessing.Pool()
        for benchmark in pool.map(
                          functools.partial(
                            benchmarks.benchmark_worker,
                            feature_space = self.feature_space,
                            reduced_git_corpus = self.reduced_git_corpus
                          ), kernels
                        ):
          if benchmark:
            self.benchmarks.append(benchmark)
        pool.close()
        benchmarks.resolve_benchmark_names(self.benchmarks)
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
  def __init__(self,
               workspace      : pathlib.Path,
               feature_space  : str,
               active_learner : 'active_models.Model',
               ):
    super(ActiveSampler, self).__init__(workspace, feature_space, str(active_learner.downstream_task))
    self.active_learner = active_learner
    self.loadCheckpoint()
    try:
      self.target_benchmark = self.benchmarks.pop(0)
      l.logger().info("Target benchmark: {}\nTarget fetures: {}".format(self.target_benchmark.name, self.target_benchmark.features))
    except IndexError:
      self.target_benchmark = None
    return

  def sample_active_learner(self) -> typing.List[Benchmark]:
    return [Benchmark("", "", "", sample['static_features'], sample['runtime_features']) for sample in self.active_learner.Sample()]

  def iter_benchmark(self) -> None:
    """
    Set the next item from list to target.
    If doesn't exist, ask from the active learner for new stuff,
    unless a termination criteria has been met.
    """
    try:
      self.target_benchmark = self.benchmarks.pop(0)
      l.logger().info("Target fetures: {}".format(self.target_benchmark.features))
    except IndexError:
      l.logger().warn("Implement a termination criteria here.")
      self.benchmarks = self.sample_active_learner()
      self.iter_benchmark()
      return
    self.saveCheckpoint()
    return

  def loadCheckpoint(self) -> None:
    """
    Load pickled list of benchmarks, if exists.
    Otherwise, ask the first batch of features from the active learner.
    """
    if (self.workspace / "feature_sampler_state.pkl").exists():
      with open(self.workspace / "feature_sampler_state.pkl", 'rb') as infile:
        self.benchmarks = pickle.load(infile)
    else:
      self.benchmarks = self.sample_active_learner()
    return
    l.logger().info("Loaded {}, {} benchmarks".format(self.target, len(self.benchmarks)))
    return
