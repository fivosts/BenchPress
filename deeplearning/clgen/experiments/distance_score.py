"""
Top-K or min distance of database groups against target benchmark suites.
"""
import tqdm
import math

from deeplearning.clgen.corpuses import encoded
from deeplearning.clgen.samplers import samples_database
from deeplearning.clgen.util import plotter
from deeplearning.clgen.experiments import public
from deeplearning.clgen.experiments import clsmith
from deeplearning.clgen.experiments import workers

@public.evaluator
def KAverageScore(**kwargs) -> None:
  """
  Compare the average of top-K closest per target benchmark
  for all different database groups.
  """
  db_groups      = kwargs.get('db_groups')
  target         = kwargs.get('targets')
  feature_space  = kwargs.get('feature_space')
  top_k          = kwargs.get('top_k')
  unique_code    = kwargs.get('unique_code', False)
  plot_config    = kwargs.get('plot_config')
  workspace_path = kwargs.get('workspace_path')
  groups = {}

  # You need this if you want to have the same (github) baseline but when github is not plotted.
  reduced_git = None
  for dbg in db_groups:
    if dbg.group_name == "GitHub-768-inactive" or dbg.group_name == "GitHub-768":
      reduced_git = dbg.get_data_features(feature_space)
      break

  benchmarks = target.get_benchmarks(feature_space, reduced_git_corpus = reduced_git)
  target_origin_dists = {}
  for dbg in db_groups:
    if dbg.group_name == "GitHub-768-inactive":
      # Skip baseline DB group.
      continue
    if not (dbg.db_type == samples_database.SamplesDatabase or dbg.db_type == encoded.EncodedContentFiles or dbg.db_type == clsmith.CLSmithDatabase):
      raise ValueError("Scores require SamplesDatabase or EncodedContentFiles but received", dbg.db_type)
    groups[dbg.group_name] = ([], [])
    for benchmark in tqdm.tqdm(benchmarks, total = len(benchmarks), desc = "Benchmarks"):
      groups[dbg.group_name][0].append(benchmark_name)
      # Find shortest distances.
      if unique_code:
        get_data = lambda x: dbg.get_unique_data_features(x)
      else:
        get_data = lambda x: dbg.get_data_features(x)

      distances = workers.SortedDistances(get_data(feature_space), benchmark.features, feature_space)
      # Compute target's distance from O(0,0)
      assert len(distances) != 0, "Sorted src list for {} is empty!".format(dbg.group_name)
      avg_dist = sum(distances[:top_k]) / top_k
      if benchmark_name in target_origin_dists:
        target_origin_dists[benchmark_name] = max(target_origin_dists[benchmark_name], avg_dist)
      else:
        target_origin_dists[benchmark_name] = max(math.sqrt(sum([x**2 for x in benchmark.features.values()])), avg_dist)

      groups[dbg.group_name][1].append(avg_dist)

  for group_name, tup in groups.items():
    bench_names, raw_dists = tup
    for idx, (bench_name, raw_dist) in enumerate(zip(bench_names, raw_dists)):
      groups[group_name][1][idx] = 100 * ( (target_origin_dists[bench_name] - raw_dist ) / target_origin_dists[bench_name])

  plotter.GrouppedBars(
    groups = groups,
    plot_name = "avg_{}_dist_{}_{}".format(top_k, feature_space.replace("Features", " Features"), '-'.join([dbg.group_name for dbg in db_groups])),
    path = workspace_path,
    **plot_config if plot_config else {},
  )
  return

@public.evaluator
def MinScore(**kwargs) -> None:
  """
  Compare the closest sample per target benchmark
  for all different database groups.
  """
  if 'top_k' in kwargs:
    del kwargs['top_k']
  KAverageScore(top_k = 1, unique_code = False, **kwargs)
  return
