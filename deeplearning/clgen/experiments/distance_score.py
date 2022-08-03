"""
Top-K or min distance of database groups against target benchmark suites.
"""
from numpy import extract
import tqdm
import typing
import math

from deeplearning.clgen.features import active_feed_database
from deeplearning.clgen.features import extractor
from deeplearning.clgen.corpuses import encoded
from deeplearning.clgen.samplers import samples_database
from deeplearning.clgen.experiments import public
from deeplearning.clgen.experiments import clsmith
from deeplearning.clgen.experiments import workers
from deeplearning.clgen.util import plotter
from deeplearning.clgen.util import logging as l

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
    if not (
        dbg.db_type == samples_database.SamplesDatabase or
        dbg.db_type == encoded.EncodedContentFiles or
        dbg.db_type == clsmith.CLSmithDatabase or
        dbg.db_type == active_feed_database.ActiveFeedDatabase
      ):
      raise ValueError("Scores require SamplesDatabase or EncodedContentFiles but received", dbg.db_type)
    groups[dbg.group_name] = ([], [])
    for benchmark in tqdm.tqdm(benchmarks, total = len(benchmarks), desc = "Benchmarks"):
      groups[dbg.group_name][0].append(benchmark.name)
      # Find shortest distances.
      if unique_code:
        get_data = lambda x: dbg.get_unique_data_features(x)
      else:
        get_data = lambda x: dbg.get_data_features(x)

      distances = workers.SortedDistances(get_data(feature_space), benchmark.features, feature_space)
      # Compute target's distance from O(0,0)
      assert len(distances) != 0, "Sorted src list for {} is empty!".format(dbg.group_name)
      avg_dist = sum(distances[:top_k]) / top_k
      if benchmark.name in target_origin_dists:
        target_origin_dists[benchmark.name] = max(target_origin_dists[benchmark.name], avg_dist)
      else:
        target_origin_dists[benchmark.name] = max(math.sqrt(sum([x**2 for x in benchmark.features.values()])), avg_dist)

      groups[dbg.group_name][1].append(avg_dist)

  for group_name, tup in groups.items():
    bench_names, raw_dists = tup
    for idx, (bench_name, raw_dist) in enumerate(zip(bench_names, raw_dists)):
      groups[group_name][1][idx] = 100 * ( (target_origin_dists[bench_name] - raw_dist ) / target_origin_dists[bench_name])

  plotter.GrouppedBars(
    groups = groups,
    plot_name = "avg_{}_dist_{}_{}".format(top_k, feature_space.replace("Features", "Features"), '-'.join([dbg.group_name for dbg in db_groups])),
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

@public.evaluator
def AnalyzeBeamSearch(**kwargs) -> None:
  """
  Analyze active feed databases and provide statistics
  on distance convergence from target.

  Two types of plots are exported:
    1. For each target benchmark, a radar plot with its features, along with the closest candidate per db group.
    2. For each target benchmark, a convergence line per generation for all db groups is shown.
  Also, a final converge distribution line per db group is exported for all target benchmarks.
  """
  db_groups      = kwargs.get('db_groups')
  target         = kwargs.get('targets')
  feature_space  = kwargs.get('feature_space')
  plot_config    = kwargs.get('plot_config')
  workspace_path = kwargs.get('workspace_path')

  def feats_to_list(feats: typing.Dict[str, float]) -> typing.Tuple[typing.List, typing.List]:
    k, v = list(feats.keys()), list(feats.values())
    k, v = zip(*sorted(zip(k, v)))
    k, v = list(k), list(v)
    return k, v

  benchmarks = target.get_benchmarks(feature_space)
  for benchmark in tqdm.tqdm(benchmarks, total = len(benchmarks), desc = "Benchmarks"):
    keys, vals = feats_to_list(benchmark.features)

    radar_features = {}
    generations_score = {}
    radar_features[benchmark.name] = [
      vals,
      keys,
    ]
    for dbg in db_groups:
      if not dbg.db_type == active_feed_database.ActiveFeedDatabase:
        raise ValueError("Beam search analysis requires ActiveFeedDatabase, but received {}", dbg.db_type)
      data = [dp for dp in dbg.get_data if target.shorten_benchmark_name(dp.target_benchmark.split('\n')[0]) == "// {}".format(benchmark.name)]
      if len(data) == 0:
        l.logger().warn("{} not found in {}, here are the features: {}".format(benchmark.name, dbg.group_name, benchmark.features))
        continue
      closest = sorted(data, key = lambda dp: dp.sample_quality)[0]
      dict_feats = {':'.join(l.split(':')[:-1]) : float(l.split(':')[-1]) for l in closest.output_features.split('\n')}
      keys, vals = feats_to_list(dict_feats)
      radar_features[dbg.group_name] = [
        vals,
        keys
      ]
      score_gens = {}
      for dp in data:
        if dp.generation_id not in score_gens:
          score_gens[dp.generation_id] = dp.sample_quality
        else:
          score_gens[dp.generation_id] = min(score_gens[dp.generation_id], dp.sample_quality)
      generations_score[dbg.group_name] = {
        'data': [[idx, v] for idx, v in score_gens.items()],
        'names': [x for x, _ in score_gens.items()]
      }
    plotter.GrouppedRadar(
      groups    = radar_features,
      plot_name = "feeds_radar_{}_{}_{}".format(feature_space, benchmark.name, '-'.join([dbg.group_name for dbg in db_groups])),
      path      = workspace_path,
      title     = benchmark.name,
      **plot_config if plot_config else {},
    )
    plotter.GroupScatterPlot(
      groups    = generations_score,
      plot_name = "Beam_generation_{}_{}_{}".format(feature_space, benchmark.name, '-'.join([dbg.group_name for dbg in db_groups])),
      path      = workspace_path,
      mode      = "lines+markers",
      title     = benchmark.name,
      **plot_config if plot_config else {},
    )

  return
