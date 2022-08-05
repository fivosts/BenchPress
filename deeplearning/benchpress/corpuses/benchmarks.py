# coding=utf-8
# Copyright 2022 Foivos Tsimpourlas.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Module for benchmarks suite pre-processing, encoding and feature extraction.
"""
import typing
import tempfile
import contextlib
import pathlib
import gdown
import json
import tqdm
import subprocess
# import multiprocessing

from deeplearning.benchpress.features import extractor
from deeplearning.benchpress.features import feature_sampler
from deeplearning.benchpress.preprocessors import opencl
from deeplearning.benchpress.preprocessors import c
from deeplearning.benchpress.util import logging as l
from deeplearning.benchpress.util import environment

from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_boolean(
  "filter_benchmarks_by_git",
  False,
  "Select to filter out benchmarks for which github-seq len has 0 distanced samples."
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

class Benchmark(typing.NamedTuple):
  path             : pathlib.Path
  name             : str
  contents         : str
  features         : typing.Dict[str, float]
  runtime_features : typing.Dict[str, float]

def preprocessor_worker(contentfile_batch):
  kernel_batch = []
  p, cf = contentfile_batch
  try:
    ks = opencl.ExtractSingleKernelsHeaders(
         opencl.InvertKernelSpecifier(
         opencl.StripDoubleUnderscorePrefixes(
         opencl.ClangPreprocessWithShim(
         c.StripIncludes(cf)))))
    for k, h in ks:
      kernel_batch.append((p, k, h))
  except ValueError:
    pass
  return kernel_batch

def benchmark_worker(benchmark, feature_space, reduced_git_corpus = None):
  p, k, h = benchmark
  features = extractor.ExtractFeatures(
    k,
    [feature_space],
    header_file = h,
    use_aux_headers = False
  )
  if reduced_git_corpus and FLAGS.filter_benchmarks_by_git:
    closest_git = sorted([(cf, feature_sampler.calculate_distance(fts, features[feature_space], feature_space)) for cf, fts in reduced_git_corpus], key = lambda x: x[1])[0]
    if features[feature_space] and closest_git[1] > 0:
      return Benchmark(p, p.name, k, features[feature_space], {})
  else:
    if features[feature_space]:
      return Benchmark(p, p.name, k, features[feature_space], {})

@contextlib.contextmanager
def GetContentFileRoot(path: pathlib.Path) -> typing.Iterator[pathlib.Path]:
  """
  Extract tar archive of benchmarks and yield the root path of all files.
  If benchmarks don't exist, download from google drive.

  Yields:
    The path of a directory containing content files.
  """
  if not (path.parent / "benchmarks_registry.json").exists():
    l.logger().warn("benchmarks_registry.json file not found. Assuming provided path is the benchmarks root path.")
    yield pathlib.Path(path)
    return

  with open(path.parent / "benchmarks_registry.json", 'r') as js:
    reg = json.load(js)

  if path.name not in reg:
    raise FileNotFoundError("Corpus {} is not registered in benchmarks_registry".format(path.name))

  if not path.is_file():
    l.logger().info("Benchmark found in registry. Downloading from Google Drive...")
    gdown.download("https://drive.google.com/uc?id={}".format(reg[path.name]['url']), str(path))

  try:
    tdir = FLAGS.local_filesystem
  except Exception:
    tdir = None

  with tempfile.TemporaryDirectory(prefix=path.stem, dir = tdir) as d:
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

def iter_cl_files(path: pathlib.Path) -> typing.List[typing.Tuple[pathlib.Path, str]]:
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
        try:
          with open(c, 'r') as inf:
            contentfiles.append((c, inf.read()))
        except UnicodeDecodeError:
          continue
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
  # pool = multiprocessing.Pool()
  # if environment.WORLD_RANK == 0:
  #   it = tqdm.tqdm(pool.map(preprocessor_worker, contentfiles), total = len(contentfiles), desc = "Yield {} benchmarks".format(path.stem))
  # else:
  #   it = pool.map(preprocessor_worker, contentfiles)
  for batch in contentfiles:
    kernel_batch = preprocessor_worker(batch)
    kernels += kernel_batch
  l.logger().info("Pre-processed {} OpenCL benchmarks".format(len(kernels)))
  # pool.close()
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
