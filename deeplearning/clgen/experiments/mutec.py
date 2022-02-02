"""
Evaluation script for mutec mutation program.
"""
import typing
import tempfile
import subprocess
import pathlib
import json
import os
import tqdm
import math

from absl import flags

from deeplearning.clgen.features import extractor
from deeplearning.clgen.util import plotter

FLAGS = flags.FLAGS

def run_single(src: str, depth = 0, visited: set = set()):

  try:
    tdir = pathlib.Path(FLAGS.local_filesystem).resolve() / feat_space
  except Exception:
    tdir = pathlib.Path("/tmp/{}".format(feat_space)).resolve()
    tdir.mkdir(exist_ok = True, parents = True)
  with tempfile.NamedTemporaryFile("w", prefix="mutec_src", suffix='.cl', dir = tdir) as f:
    # Write source file.
    f.write(src)
    f.flush()
    # Fix compile_commands.json for source file.
    base_path = pathlib.Path(f.name).resolve().parent
    compile_command = {
      'directory' : str(base_path),
      'arguments' : 
        [str(clang.CLANG), f.name] +
        ["-S", "-emit-llvm", "-o", "-"] +
        opencl.GetClangArgs(use_shim = False, use_aux_headers = False),
      'file'      : str(f.name)
    }
    with open(base_path / "compile_commands.json", 'w') as ccf:
      json.dump([compile_command], ccf)
    # Construct and execute mutec command
    mutec_cmd = [
      str(mutec),
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
    stdout, stderr = process.communicate()
    # Cleanup compile commands
    os.remove(str(base_path / "compile_commands.json"))
    mutecs = set(
      [x for x in 
        [open(x, 'r').read() for x in glob.glob(str("{}.mutec*".format(str(f.name))))]
      if x not in visited]
    )
    visited.update(mutecs)
    if depth < 3 and len(visited) < 50:
      ret = set()
      for mutated in mutecs:
        ret.update(run_single(mutated, depth = depth + 1, visited = visited))
      mutecs.update(ret)
    return mutecs

def run_mutec(srcs: typing.List[str], target_features: typing.Dict[str, float], feat_space: str, top_k: int) -> typing.List[typing.Tuple[str, float]]:
  try:
    tdir = FLAGS.local_filesystem
  except Exception:
    tdir = None

  cands = []
  for src in tqdm.tqdm(srcs, total = len(srcs), desc = "Mutec candidates", leave = False):
    cands += run_single(src)
  ## Tuple of closest src, distance from target benchmark.
  closest = []
  for src in cands:
    feats = extractor.ExtractFeatures(src, [feat_space])
    if feat_space in feats and feats[feat_space]:
      closest.append((src, feature_sampler.calculate_distance(feats[feat_space], target_features, feat_space)))
  closest = sorted(closest, key = lambda x: x[1])[:top_k]
  return closest

def MutecVsBenchPress(**kwargs) -> None:
  """
  Compare mutec mutation tool on github's database against BenchPress.
  Comparison is similar to KAverageScore comparison.
  """
  mutec          = kwargs.get('mutec')
  github         = kwargs.get('github')
  benchpress     = kwargs.get('benchpress')
  target         = kwargs.get('targets')
  feature_space  = kwargs.get('feature_space')
  top_k          = kwargs.get('top_k')
  unique_code    = kwargs.get('unique_code', False)
  plot_config    = kwargs.get('plot_config')
  workspace_path = kwargs.get('workspace_path')

  if github.db_type != encoded.EncodedContentFiles:
    raise ValueError("Scores require EncodedContentFiles but received", github.db_type)

  if benchpress.db_type != samples_database.SamplesDatabase:
    raise ValueError("BenchPress scores require SamplesDatabase but received", benchpress.db_type)

  ## Initialize dictionary.
  groups = {}
  groups["Mutec"] = ([], [])
  groups["GitHub"] = ([], [])
  groups[benchpress.group_name] = ([], [])

  ## Fix fetching data functions.
  if unique_code:
    git_get_data = lambda x: github.get_unique_data_features(x)
    bp_get_data  = lambda x: benchpress.get_unique_data_features(x)
  else:
    git_get_data = lambda x: github.get_data_features(x)
    bp_get_data  = lambda x: benchpress.get_data_features(x)

  ## Run engine on mutec.
  benchmarks = target.get_benchmarks(feature_space)
  for benchmark in tqdm.tqdm(benchmarks, total = len(benchmarks), desc = "Mutec"):
    ## Tuple of closest src, distance from target benchmark.0
    closest = SortedSrcDistances(git_get_data(feature_space), benchmark.features, feature_space)[:5]

    # Split source and distances lists.
    git_src  = [x for x, _ in closest]
    git_dist = [x for _, x in closest]

    ## If distances are already minimized, nothing to do.
    if sum(git_dist[:top_k]) == 0:
      continue

    l.logger().info(benchmark.name)

    closest_mutec_src  = beam_mutec(git_src, benchmark.features, feature_space, top_k) # tuple of (src, distance)
    closest_mutec_dist = [x for _, x in closest_mutec_src]

    ## If mutec has provided a better score
    if sum(closest_mutec_dist) < sum(git_dist[:top_k]):

      l.logger().info("Score reduced from {} to {}".format(sum(git_dist[:top_k]), sum(closest_mutec_dist)))
      l.logger().info("Best score from {} to {}".format(git_dist[0], closest_mutec_dist[0]))

      # Compute target's distance from O(0,0)
      target_origin_dist = math.sqrt(sum([x**2 for x in benchmark.features.values()]))
      mutec_avg_dist     = sum(closest_mutec_dist) / top_k

      groups["Mutec"][0].append(benchmark.name)
      groups["Mutec"][1].append(100 * ((target_origin_dist - mutec_avg_dist) / target_origin_dist))

      # Compute target's distance from O(0,0)
      git_avg_dist = sum(git_dist[:top_k]) / top_k
      groups["GitHub"][0].append(benchmark.name)
      groups["GitHub"][1].append(100 * ((target_origin_dist - git_avg_dist) / target_origin_dist))

  ## Run engine on benchpress.
  benchmarks = target.get_benchmarks(feature_space)
  for benchmark in tqdm.tqdm(benchmarks, total = len(benchmarks), desc = "Benchpress"):
    ## Run only for benchmarks mutec has improved.
    if benchmark.name in groups["Mutec"][0]:

      l.logger().info(benchmark.name)
      distances = SortedDistances(bp_get_data(feature_space), benchmark.features, feature_space)

      # Compute target's distance from O(0,0)
      target_origin_dist = math.sqrt(sum([x**2 for x in benchmark.features.values()]))
      avg_dist = sum(distances[:top_k]) / len(distances[:top_k])

      groups[benchpress.group_name][0].append(benchmark.name)
      groups[benchpress.group_name][1].append(100 * ((target_origin_dist - avg_dist) / target_origin_dist))

  plotter.GrouppedBars(
    groups = groups,
    plot_name = "mutec_avg_{}".format(top_k, feature_space.replace("Features", " Features")),
    path = workspace_path,
    **plot_config if plot_config else {},
  )
  return
