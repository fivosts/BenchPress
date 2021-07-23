"""
Feature space sampling of source code.
"""
import typing
import tempfile
import contextlib
import pathlib
import pickle
import math
import subprocess

from deeplearning.clgen.features import extractor
from deeplearning.clgen.features import normalizers
from deeplearning.clgen.preprocessors import opencl
from deeplearning.clgen.preprocessors import c
from eupy.native import logger as l

from absl import flags
from eupy.native import logger as l

FLAGS = flags.FLAGS

targets = {
  'rodinia': './benchmarks/rodinia_3.1.tar.bz2',
  'grid_walk': '',
}

@contextlib.contextmanager
def GetContentFileRoot(path: pathlib.Path) -> typing.Iterator[pathlib.Path]:
  """
  Extract tar archive of benchmarks and yield the root path of all files.

  Yields:
    The path of a directory containing content files.
  """
  with tempfile.TemporaryDirectory(prefix=path.stem) as d:
    cmd = [
      "tar",
      "-xf",
      str(path),
      "-C",
      d,
    ]
    subprocess.check_call(cmd)
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
  return contentfiles

def yield_cl_kernels(path: pathlib.Path) -> typing.List[typing.Tuple[pathlib.Path, str, str]]:
  """
  Fetch all cl files from base path and atomize, preprocess
  kernels to single instances.
  """
  # cf = pathlib.Path("/home/foivos/PhD/Code/clgen/rodinia_benchmarks/histogram1024.cl")
  # cf2 = ""
  # with open(cf, 'r') as inf:
  #   cf2 = inf.read() 
  # ks = opencl.ExtractSingleKernelsHeaders(
  #        opencl.InvertKernelSpecifier(
  #        opencl.StripDoubleUnderscorePrefixes(
  #        opencl.ClangPreprocessWithShim(
  #        c.StripIncludes(cf2)))))
  # print(len(ks))
  # feats1 = extractor.ExtractRawFeatures(ks[0][0], ['GreweFeatures'], header_file = ks[0][1], use_aux_headers = False)  
  # feats3 = extractor.ExtractFeatures(ks[0][0], ['GreweFeatures'], header_file = ks[0][1], use_aux_headers = False)  
  # feats2 = extractor.ExtractRawFeatures(ks[0][0], ['AutophaseFeatures'], header_file = ks[0][1], use_aux_headers = False)  
  # print(feats1)
  # print(feats3)
  # # print(feats2)
  # exit()
  contentfiles = iter_cl_files(path)
  kernels = []
  for p, cf in contentfiles:
    ks = opencl.ExtractSingleKernelsHeaders(
         opencl.InvertKernelSpecifier(
         opencl.StripDoubleUnderscorePrefixes(
         opencl.ClangPreprocessWithShim(
         c.StripIncludes(cf)))))
    for k, h in ks:
      kernels.append((p, k, h))
  return kernels

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
    n = tarfeat[key] if tarfeat[key] != 0 and key != "F2:coalesced/mem" and key != "F4:comp/mem" else 1# normalizers.normalizer[feature_space][key]
    i = infeat[key] / n
    t = tarfeat[key] / n
    d += abs((t**2) - (i**2))
  return math.sqrt(d)

class Benchmark(typing.NamedTuple):
  path: pathlib.Path
  name: str
  contents: str
  feature_vector: typing.Dict[str, float]

class EuclideanSampler(object):
  """
  This is a shitty experimental class to work with benchmark comparison.
  Will be refactored obviously.
  """
  def __init__(self,
               workspace: pathlib.Path,
               feature_space: str,
               target: str
               ):
    self.target = target
    if self.target != "grid_walk":
      self.path        = pathlib.Path(targets[target]).resolve()
    self.workspace     = workspace
    self.feature_space = feature_space
    self.loadCheckpoint()
    try:
      self.target_benchmark = self.benchmarks.pop(0)
      l.getLogger().info("Target benchmark: {}\nTarget fetures: {}".format(self.target_benchmark.name, self.target_benchmark.feature_vector))
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
    except IndexError:
      self.target_benchmark = None
    self.saveCheckpoint()
    l.getLogger().info("Target benchmark: {}\nTarget fetures: {}".format(self.target_benchmark.name, self.target_benchmark.feature_vector))
    return

  def calculate_distance(self, infeat: typing.Dict[str, float]) -> float:
    """
    Euclidean distance between sample feature vector
    and current target benchmark.
    """
    return calculate_distance(infeat, self.target_benchmark.feature_vector, self.feature_space)

  def topK_candidates(self,
                      candidates: typing.List[typing.TypeVar("ActiveSample")],
                      K : int,
                      ) -> typing.List[typing.TypeVar("ActiveSample")]:
    """
    Return top-K candidates.
    """
    return sorted(candidates, key = lambda x: x.score)[:K]

  def sample_from_set(self, 
                      candidates: typing.List[typing.TypeVar("ActiveSample")],
                      search_width: int,
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
    if False: # (self.workspace / "feature_sampler_state.pkl").exists():
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
        for p, k, h in kernels:
          features = extractor.ExtractFeatures(
            k,
            [self.feature_space],
            header_file = h,
            use_aux_headers = False
          )
          if features[self.feature_space]:
            self.benchmarks.append(
              Benchmark(
                  p,
                  p.name,
                  k,
                  features[self.feature_space],
                )
            )
    l.getLogger().info("Loaded {}, {} benchmarks".format(self.target, len(self.benchmarks)))
    l.getLogger().info(', '.join([x for x in set([x.name for x in self.benchmarks])]))
    return
