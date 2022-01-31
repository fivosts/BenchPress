"""
Feature space sampling of source code.
"""
import typing
import tempfile
import contextlib
import pathlib
import pickle
import gdown
import json
import tqdm
import math
import subprocess
import functools
import multiprocessing
from numpy.random import default_rng

from deeplearning.clgen.features import extractor
from deeplearning.clgen.features import normalizers
from deeplearning.clgen.preprocessors import opencl
from deeplearning.clgen.preprocessors import c
from deeplearning.clgen.corpuses import corpuses
from deeplearning.clgen.util import logging as l

from absl import flags
from deeplearning.clgen.util import logging as l

FLAGS = flags.FLAGS

flags.DEFINE_integer(
  "randomize_selection",
  None,
  "Debugging integer flag that abolishes euclidean distance priority and picks X randomly selected elements to return as top-X."
)

targets = {
  'rodinia'      : './model_zoo/benchmarks/rodinia_3.1.tar.bz2',
  'BabelStream'  : './model_zoo/benchmarks/BabelStream.tar.bz2',
  'cf4ocl'       : './model_zoo/benchmarks/cf4ocl.tar.bz2',
  'CHO'          : './model_zoo/benchmarks/cho.tar.bz2',
  'FinanceBench' : './model_zoo/benchmarks/FinanceBench.tar.bz2',
  'HeteroMark'   : './model_zoo/benchmarks/HeteroMark.tar.bz2',
  'mixbench'     : './model_zoo/benchmarks/mixbench.tar.bz2',
  'OpenDwarfs'   : './model_zoo/benchmarks/OpenDwarfs.tar.bz2',
  'parboil'      : './model_zoo/benchmarks/parboil.tar.bz2',
  'polybench'    : './model_zoo/benchmarks/polybench.tar.bz2',
  'grid_walk'    : '',
}

def preprocessor_worker(contentfile_batch):
  kernel_batch = []
  p, cf = contentfile_batch
  ks = opencl.ExtractSingleKernelsHeaders(
       opencl.InvertKernelSpecifier(
       opencl.StripDoubleUnderscorePrefixes(
       opencl.ClangPreprocessWithShim(
       c.StripIncludes(cf)))))
  for k, h in ks:
    kernel_batch.append((p, k, h))
  return kernel_batch

def benchmark_worker(benchmark, feature_space, reduced_git_corpus):
  p, k, h = benchmark
  features = extractor.ExtractFeatures(
    k,
    [feature_space],
    header_file = h,
    use_aux_headers = False
  )
  closest_git = sorted([(cf, calculate_distance(fts, features[feature_space], feature_space)) for cf, fts in reduced_git_corpus], key = lambda x: x[1])[0]
  if features[feature_space] and closest_git[1] > 0:
    return Benchmark(p, p.name, k, features[feature_space])

@contextlib.contextmanager
def GetContentFileRoot(path: pathlib.Path) -> typing.Iterator[pathlib.Path]:
  """
  Extract tar archive of benchmarks and yield the root path of all files.
  If benchmarks don't exist, download from google drive.

  Yields:
    The path of a directory containing content files.
  """
  if not (path.parent / "benchmarks_registry.json").exists():
    raise FileNotFoundError("benchmarks_registry.json file not found.")

  with open(path.parent / "benchmarks_registry.json", 'r') as js:
    reg = json.load(js)

  if path.name not in reg:
    raise FileNotFoundError("Corpus {} is not registered in benchmarks_registry".format(path.name))

  if not path.is_file():
    l.logger().info("Benchmark found in registry. Downloading from Google Drive...")
    gdown.download("https://drive.google.com/uc?id={}".format(reg[path.name]['url']), str(path))

  with tempfile.TemporaryDirectory(prefix=path.stem, dir = FLAGS.local_filesystem) as d:
    cmd = [
      "tar",
      "-xf",
      str(path),
      "-C",
      d,
    ]
    subprocess.check_call(cmd)
    l.logger().info("Unpacked benchmark suite {}".format(str(d)))
    yield pathlib.Path(d)

def iter_cl_files(path: pathlib.Path) -> typing.List[str]:
  """
  Iterate base path and yield the contents of all .cl files.
  """
  contentfiles = []
  with GetContentFileRoot(path) as root:
    file_queue = [p for p in root.iterdir()]
    while file_queue:
      c = file_queue.pop(0)
      if c.is_symlink():
        continue
      elif c.is_dir():
        file_queue += [p for p in c.iterdir()]
      elif c.is_file() and c.suffix == ".cl":
        with open(c, 'r') as inf:
          contentfiles.append((c, inf.read()))
  l.logger().info("Scanned \'.cl\' files in {}".format(str(path)))
  return contentfiles

def yield_cl_kernels(path: pathlib.Path) -> typing.List[typing.Tuple[pathlib.Path, str, str]]:
  """
  Fetch all cl files from base path and atomize, preprocess
  kernels to single instances.

  Original benchmarks extracted from suites, go through a series of pre-processors:
  1. Include statements are removed.
  2. Code is preprocessed with shim (macro expansion).
  3. Double underscores are removed.
  4. void kernel -> kernel void
  5. Translation units are split to tuples of (kernel, utility/global space)
  """
  contentfiles = iter_cl_files(path)
  kernels = []
  pool = multiprocessing.Pool()
  for kernel_batch in tqdm.tqdm(pool.map(preprocessor_worker, contentfiles), total = len(contentfiles), desc = "Yield {} benchmarks".format(path.stem)):
    kernels += kernel_batch
  l.logger().info("Pre-processed {} OpenCL benchmarks".format(len(kernels)))
  return kernels

def resolve_benchmark_names(benchmarks: typing.List["Benchmark"]) -> typing.List["Benchmark"]:
  """
  Resolves duplicate benchmark names. e.g. X, X, X -> X-1, X-2, X-3.
  """
  renaming = {}
  for benchmark in benchmarks:
    if benchmark.name not in renaming:
      renaming[benchmark.name] = [0, 0]
    else:
      renaming[benchmark.name][1] += 1
  for idx, benchmark in enumerate(benchmarks):
    if renaming[benchmark.name][1] != 0:
      renaming[benchmark.name][0] += 1
      benchmarks[idx] = benchmarks[idx]._replace(
        name = "{}-{}".format(benchmark.name, renaming[benchmark.name][0])
      )
  return sorted(benchmarks, key = lambda x: x.name)

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
  path : pathlib.Path
  name : str
  contents : str
  features : typing.Dict[str, float]

class EuclideanSampler(object):
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
    self.target     = target
    self.benchmarks = []
    if self.target  != "grid_walk":
      self.path        = pathlib.Path(targets[target]).resolve()
    self.workspace     = workspace
    self.feature_space = feature_space
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
                      candidates: typing.List[typing.TypeVar("ActiveSample")],
                      search_width: int,
                      only_unique: bool = True,
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

  def saveCheckpoint(self) -> None:
    """
    Save feature sampler state.
    """
    with open(self.workspace / "feature_sampler_state.pkl", 'wb') as outf:
      pickle.dump(self.benchmarks, outf)
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
            )
          )
        self.saveCheckpoint()
      else:
        kernels = yield_cl_kernels(self.path)
        pool = multiprocessing.Pool()
        for benchmark in pool.map(
                          functools.partial(
                            benchmark_worker,
                            feature_space = self.feature_space,
                            reduced_git_corpus = self.reduced_git_corpus
                          ), kernels
                        ):
          if benchmark:
            self.benchmarks.append(benchmark)
        resolve_benchmark_names(self.benchmarks)
    l.logger().info("Loaded {}, {} benchmarks".format(self.target, len(self.benchmarks)))
    l.logger().info(', '.join([x for x in set([x.name for x in self.benchmarks])]))
    return
