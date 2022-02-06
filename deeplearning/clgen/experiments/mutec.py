"""
Evaluation script for mutec mutation program.
"""
import typing
import glob
import tempfile
import subprocess
import pathlib
import json
import os
import tqdm
import functools
import math
import multiprocessing

from absl import flags

from deeplearning.clgen.corpuses import encoded
from deeplearning.clgen.features import extractor
from deeplearning.clgen.samplers import samples_database
from deeplearning.clgen.preprocessors import clang
from deeplearning.clgen.preprocessors import opencl
from deeplearning.clgen.util import plotter
from deeplearning.clgen.util import logging as l
from deeplearning.clgen.util import environment
from deeplearning.clgen.experiments import workers
from deeplearning.clgen.experiments import public
from deeplearning.clgen.experiments import clsmith

FLAGS = flags.FLAGS

MUTEC = environment.MUTEC
CLSMITH_INCLUDE = environment.CLSMITH_INCLUDE

## Some hard limits in order to finish the experiments this year.
# max amount of mutants per input source.
PER_INPUT_HARD_LIMIT = 1000
SEARCH_DEPTH_HARD_LIMIT = 30

def generate_mutants(src: str, incl: str, timeout_seconds: int = 45) -> typing.Set[typing.Tuple[str, str]]:
  """
  Collect all mutants from src and return them
  """
  try:
    tdir = pathlib.Path(FLAGS.local_filesystem).resolve()
  except Exception:
    tdir = None

  with tempfile.NamedTemporaryFile("w", prefix="mutec_src", suffix='.cl', dir = tdir) as f:
    try:
      f.write(src)
      f.flush()
    except UnicodeDecodeError:
      return []
    except UnicodeEncodeError:
      return []

    if incl:
      with open("/tmp/mutec_src_temp_header.h", 'w') as f:
        f.write(incl)
        f.flush()

    # Fix compile_commands.json for source file.
    base_path = pathlib.Path(f.name).resolve().parent
    compile_command = {
      'directory' : str(base_path),
      'arguments' : [str(clang.CLANG), f.name] +
                    ["-S", "-emit-llvm", "-o", "-"] +
                    opencl.GetClangArgs(use_shim = False, use_aux_headers = False, extra_args = ["-include{}".format(pathlib.Path(CLSMITH_INCLUDE) / "CLSmith.h")] if incl else None) +
                    ["-include/tmp/mutec_src_temp_header.h" if incl else ""],
      'file'      : str(f.name)
    }
    with open(base_path / "compile_commands.json", 'w') as ccf:
      json.dump([compile_command], ccf)
    # Construct and execute mutec command
    mutec_cmd = [
      "timeout",
      "-s9",
      str(timeout_seconds),
      MUTEC,
      str(f.name),
      "-o",
      str(base_path)
    ]
    process = subprocess.Popen(
      mutec_cmd,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      universal_newlines=True,
    )
    try:
      stdout, stderr = process.communicate()
    except TimeoutError:
      pass
    os.remove(str(base_path / "compile_commands.json"))

    mutec_paths = glob.glob("{}.mutec*".format(f.name))
    templates   = glob.glob("{}.code_template".format(f.name))
    mutants = set([(open(x, 'r').read(), incl) for x in mutec_paths[:PER_INPUT_HARD_LIMIT]])

  for m in mutec_paths:
    os.remove(m)
  for m in templates:
    os.remove(m)
  os.remove("/tmp/mutec_src_temp_header.h")
  return mutants

def beam_mutec(srcs            : typing.List[typing.Tuple[str, str, float]],
               target_features : typing.Dict[str, float],
               feat_space      : str,
               beam_width      : int,
               mutec_cache     : samples_database.SamplesDatabase,
               ) -> typing.List[typing.Tuple[str, float]]:
  """
  Run generational beam search over starting github kernels
  to minimize distance from target features.
  """
  better_score = True
  total_beams, beam, closest = set(), [], []
  gen_id = 0

  while better_score:

    cands = set()
    ## Generate mutants for current generation.
    for src, incl, dist in tqdm.tqdm(srcs, total = len(srcs), desc = "Mutec candidates {}".format(gen_id), leave = False):
      cands.update(generate_mutants(src, incl)) ### This should collect all mutants and return them, out of a single source.

    ## Extract their features and calculate distances.
    pool = multiprocessing.Pool()
    f = functools.partial(
          workers.ExtractAndCalculate,
          target_features = target_features,
          feature_space = feat_space,
        )
    # total.update(cands)
    try:
      for cand in tqdm.tqdm(pool.imap_unordered(f, cands), total = len(cands), desc = "Extract Features {}".format(gen_id), leave = False):
        if cand:
          beam.append(cand)
    except Exception as e:
      l.logger().error(e)
      pool.terminate()
      raise e
    pool.close()

    ## Sort by distance in ascending order. If score is better, keep doing beam search
    ## srcs are included to the outputs, in order to keep them if the offsprings are worse.
    closest = sorted(beam + srcs, key = lambda x: x[2])[:beam_width]
    total_beams.update([(x, y) for x, y, _ in closest])

    min_length = min(len(closest), len(srcs))
    if sum([x for _, _, x in closest[:min_length]]) < sum([x for _, _, x in srcs[:min_length]]) and gen_id < SEARCH_DEPTH_HARD_LIMIT:
      srcs = closest
      beam = []
    else:
      better_score = False
    gen_id += 1

  ## Store all mutants in database.
  with mutec_cache.Session(commit = True) as s:
    pool = multiprocessing.Pool()
    try:
      idx = mutec_cache.count
      for dp in tqdm.tqdm(pool.imap_unordered(workers.FeatureExtractor, total_beams), total = len(total_beams), desc = "Add mutants to DB", leave = False):
        if dp:
          src, incl, feats = dp
          sample = samples_database.Sample.FromArgsLite(idx, incl + src, feats)
          exists = s.query(samples_database.Sample.sha256).filter_by(sha256 = sample.sha256).scalar() is not None
          if not exists:
            s.add(sample)
            idx += 1
    except Exception as e:
      l.logger().error(e)
      pool.terminate()
      raise e
    pool.close()
    s.commit()
  return closest

@public.evaluator
def MutecVsBenchPress(**kwargs) -> None:
  """
  Compare mutec mutation tool on github's database against BenchPress.
  Comparison is similar to KAverageScore comparison.
  """
  seed           = kwargs.get('seed')
  benchpress     = kwargs.get('benchpress')
  mutec_cache    = kwargs.get('mutec_cache', '')
  target         = kwargs.get('targets')
  feature_space  = kwargs.get('feature_space')
  top_k          = kwargs.get('top_k')
  beam_width     = kwargs.get('beam_width')
  unique_code    = kwargs.get('unique_code', False)
  plot_config    = kwargs.get('plot_config')
  workspace_path = kwargs.get('workspace_path')

  if not pathlib.Path(MUTEC).exists():
    raise FileNotFoundError("Mutec executable not found: {}".format(MUTEC))
  if seed.db_type != encoded.EncodedContentFiles and seed.db_type != clsmith.CLSmithDatabase:
    raise ValueError("Scores require EncodedContentFiles or CLSmithDatabase but received", seed.db_type)
  if benchpress.db_type != samples_database.SamplesDatabase:
    raise ValueError("BenchPress scores require SamplesDatabase but received", benchpress.db_type)

  ## Load database and checkpoint of targets.
  mutec_db = samples_database.SamplesDatabase(url = "sqlite:///{}".format(pathlib.Path(mutec_cache).resolve()), must_exist = False)
  done = set()
  with mutec_db.Session(commit = True) as s:
    res = s.query(samples_database.SampleResults).filter_by(key = feature_space).first()
    if res is not None:
      done.update([str(x) for x in res.results.split('\n')])
    s.commit()

  ## Initialize dictionary.
  groups = {}
  groups["Mutec"] = ([], [])
  groups[seed.group_name] = ([], [])
  groups[benchpress.group_name] = ([], [])

  ## Fix fetching data functions.
  if unique_code:
    git_get_data = lambda x: seed.get_unique_data_features(x)
    bp_get_data  = lambda x: benchpress.get_unique_data_features(x)
  else:
    git_get_data = lambda x: seed.get_data_features(x)
    bp_get_data  = lambda x: benchpress.get_data_features(x)

  ## Run engine on mutec.
  benchmarks = target.get_benchmarks(feature_space)
  for benchmark in tqdm.tqdm(benchmarks, total = len(benchmarks), desc = "Benchmarks"):

    ## This has already been searched for.
    if benchmark.name in done:
      continue

    ## Tuple of closest src, distance from target benchmark.0
    closest = workers.SortedSrcDistances(git_get_data(feature_space), benchmark.features, feature_space)

    ## IF CLsmith takes too long here, collect only features, then for the beam size go and fetch
    ## the code.

    # Split source and distances lists.
    git_dist = [x for _, _, x in closest]

    ## If distances are already minimized, nothing to do.
    if sum(git_dist[:top_k]) == 0:
      continue

    l.logger().info(benchmark.name)

    closest_mutec_src  = beam_mutec([(src, inc, dist) for src, inc, dist in closest[:beam_width] if dist > 0], benchmark.features, feature_space, beam_width, mutec_db)[:top_k] # tuple of (src, distance)
    closest_mutec_dist = [x for _, _, x in closest_mutec_src]

    assert len(closest_mutec_dist) == len(git_dist[:top_k])
    ## If mutec has provided a better score
    if sum(closest_mutec_dist) < sum(git_dist[:top_k]):

      l.logger().info("Score reduced from {} to {}".format(sum(git_dist[:top_k]), sum(closest_mutec_dist)))
      l.logger().info("Best score from {} to {}".format(git_dist[0], closest_mutec_dist[0]))

      with mutec_db.Session(commit = True) as s:
        res = s.query(samples_database.SampleResults).filter_by(key = feature_space).first()
        if res is not None:
          res.results = res.results + "\n" + benchmark.name
        else:
          s.add(samples_database.SampleResults(key = feature_space, results = benchmark.name))
        s.commit()

      # Compute target's distance from O(0,0)
      target_origin_dist = math.sqrt(sum([x**2 for x in benchmark.features.values()]))
      mutec_avg_dist     = sum(closest_mutec_dist) / top_k

      groups["Mutec"][0].append(benchmark.name)
      groups["Mutec"][1].append(100 * ((target_origin_dist - mutec_avg_dist) / target_origin_dist))

      # Compute target's distance from O(0,0)
      git_avg_dist = sum(git_dist[:top_k]) / top_k
      groups[seed.group_name][0].append(benchmark.name)
      groups[seed.group_name][1].append(100 * ((target_origin_dist - git_avg_dist) / target_origin_dist))

  ## Run engine on benchpress.
  benchmarks = target.get_benchmarks(feature_space)
  for benchmark in tqdm.tqdm(benchmarks, total = len(benchmarks), desc = "Benchpress"):
    ## Run only for benchmarks mutec has improved.
    if benchmark.name in groups["Mutec"][0]:

      l.logger().info(benchmark.name)
      distances = workers.SortedDistances(bp_get_data(feature_space), benchmark.features, feature_space)

      # Compute target's distance from O(0,0)
      target_origin_dist = math.sqrt(sum([x**2 for x in benchmark.features.values()]))
      avg_dist = sum(distances[:top_k]) / len(distances[:top_k])

      groups[benchpress.group_name][0].append(benchmark.name)
      groups[benchpress.group_name][1].append(100 * ((target_origin_dist - avg_dist) / target_origin_dist))

  plotter.GrouppedBars(
    groups = groups,
    plot_name = "mutec_avg_{}_{}_{}".format(top_k, seed.group_name, feature_space.replace("Features", " Features")),
    path = workspace_path,
    **plot_config if plot_config else {},
  )
  return
