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
Top-K or min distance of database groups against target benchmark suites.
"""
import json
import tqdm
import typing
import math

from deeplearning.benchpress.features import active_feed_database
from deeplearning.benchpress.features import evaluate_cand_database
from deeplearning.benchpress.features import extractor
from deeplearning.benchpress.corpuses import encoded
from deeplearning.benchpress.samplers import samples_database
from deeplearning.benchpress.experiments import public
from deeplearning.benchpress.experiments import clsmith
from deeplearning.benchpress.experiments import workers
from deeplearning.benchpress.util import plotter
from deeplearning.benchpress.util import distributions
from deeplearning.benchpress.util import logging as l

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
  workspace_path = kwargs.get('workspace_path') / "{}_avg_score".format(top_k) / feature_space
  workspace_path.mkdir(exist_ok = True, parents = True)
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
    groups[dbg.group_name] = ([], [], [])
    for benchmark in tqdm.tqdm(benchmarks, total = len(benchmarks), desc = "Benchmarks"):
      groups[dbg.group_name][0].append(benchmark.name)
      # Find shortest distances.
      if unique_code:
        raise NotImplementedError
        get_data = lambda x: dbg.get_unique_data_features(x, target_name = benchmark.full_name)
      else:
        get_data = lambda x: dbg.get_data_features(x)

      src_distances = workers.SortedSrcDistances(get_data(feature_space), benchmark.features, feature_space)
      distances = [d for _, _, d in src_distances]
      # Compute target's distance from O(0,0)
      if len(distances) == 0:
        l.logger().error("{}-{}: Empty list, make sure this is expected.".format(benchmark.name, dbg.group_name))
        continue
      assert len(distances) != 0, "Sorted src list for {} is empty!".format(dbg.group_name)
      avg_dist = sum(distances[:top_k]) / top_k
      if benchmark.name in target_origin_dists:
        target_origin_dists[benchmark.name] = max(target_origin_dists[benchmark.name], avg_dist)
      else:
        target_origin_dists[benchmark.name] = max(math.sqrt(sum([x**2 for x in benchmark.features.values()])), avg_dist)

      groups[dbg.group_name][1].append(avg_dist)
      groups[dbg.group_name][2].append([s for s, _, _ in src_distances[:top_k]])

  averages = {}
  abs_average = {}
  counters = {}
  for group_name, tup in groups.items():
    bench_names, raw_dists, _ = tup
    averages[group_name] = 0.0
    abs_average[group_name] = 0.0
    counters[group_name] = 0
    for idx, (bench_name, raw_dist) in enumerate(zip(bench_names, raw_dists)):
      groups[group_name][1][idx] = 100 * ( (target_origin_dists[bench_name] - raw_dist ) / target_origin_dists[bench_name])
      averages[group_name] += ( (target_origin_dists[bench_name] - raw_dist ) / target_origin_dists[bench_name])
      abs_average[group_name] += raw_dist
      if ( (target_origin_dists[bench_name] - raw_dist ) / target_origin_dists[bench_name]) == 1.0:
        counters[group_name] += 1
    averages[group_name] = averages[group_name] / len(bench_names)
    abs_average[group_name] = abs_average[group_name] / len(bench_names)

  l.logger().info(feature_space)
  l.logger().info("Average euclidean distance:\n{}".format(abs_average))
  l.logger().info("Average relative proxmity:\n{}".format(averages))
  l.logger().info("Exact target feature matches: {}".format(counters))
  plotter.GrouppedBars(
    groups = {dbname: (c[0], c[1]) for dbname, c in groups.items()},
    plot_name = "avg_{}_dist_{}_{}".format(top_k, feature_space.replace("Features", "Features"), '-'.join([dbg.group_name for dbg in db_groups])),
    path = workspace_path,
    **plot_config if plot_config else {},
  )
  """
  ## Grewe
  groups["BenchDirect"]['data'] = [[267*2048, 73.56], [266*1024, 77.79], [512*290, 81.56], [256*289, 82.94], [128*272, 85.30], [64*282, 87.62], [32*151, 96.24]]
  groups["BenchPress"]['data'] = [[2048*286, 76.79], [1024*306, 83.62], [512*325, 88.27], [256*326, 91.47], [128*333, 95.53], [64*338, 97.30], [32*236, 99.13]]

  groups["BenchDirect"]['names'] = [2048, 1024, 512, 256, 128, 64, 32]
  groups["BenchPress"]['names'] = [2048, 1024, 512, 256, 128, 64, 32]

  time_speedup = [100*abs(round((x[0]-y[0])) / y[0]) for x, y in zip(groups["BenchDirect"]["data"], groups["BenchPress"]["data"])]
  acc_speedup = [100*abs(round((x[1]-y[1])) / y[1]) for x, y in zip(groups["BenchDirect"]["data"], groups["BenchPress"]["data"])]

  time_speedup = [[x, y] for x, y in zip([2048, 1024, 512, 256, 128, 64, 32], time_speedup)]
  acc_speedup = [[x, y] for x, y in zip([2048, 1024, 512, 256, 128, 64, 32], acc_speedup)]

  print(time_speedup)
  print(acc_speedup)

  plt.GroupScatterPlot(groups, plot_name="grewe")
  plt.GroupScatterPlot({"time_speedup": {'data': time_speedup, 'names': []}, "accuracy_improvement": {'data': acc_speedup, 'names': []}}, plot_name="grewe_speedup")

  ## Autophase
  groups["BenchDirect"]['data'] = [[262*2048, 41.02], [262*1024, 44.7], [512*267, 52.36], [256*262, 54.60], [128*254, 58.02], [64*230, 61.09], [32*164, 83.59]]
  groups["BenchPress"]['data'] = [[2048*292, 48.88], [1024*297, 50.84], [512*302, 57.38], [256*307, 57.63], [128*312, 71.32], [64*312, 74.27], [32*254, 83.59]]

  time_speedup = [100*abs(round((x[0]-y[0])) / y[0]) for x, y in zip(groups["BenchDirect"]["data"], groups["BenchPress"]["data"])]
  acc_speedup = [100*abs(round((x[1]-y[1])) / y[1]) for x, y in zip(groups["BenchDirect"]["data"], groups["BenchPress"]["data"])]

  time_speedup = [[x, y] for x, y in zip([2048, 1024, 512, 256, 128, 64, 32], time_speedup)]
  acc_speedup = [[x, y] for x, y in zip([2048, 1024, 512, 256, 128, 64, 32], acc_speedup)]

  print(time_speedup)
  print(acc_speedup)

  plt.GroupScatterPlot(groups, plot_name="autophase")
  plt.GroupScatterPlot({"time_speedup": {'data': time_speedup, 'names': []}, "accuracy_improvement": {'data': acc_speedup, 'names': []}}, plot_name="autophase_speedup")

  ## Instcount
  groups["BenchDirect"]['data'] = [[252*2048, 30.73], [257*1024, 34.36], [512*262, 36.32], [256*259, 39.89], [128*265, 41.96], [64*257, 46.21], [32*163, 48.33]]
  groups["BenchPress"]['data'] = [[2048*301, 32.63], [1024*307, 40.09], [512*302, 40.49], [256*307, 52.89], [128*307, 56.41], [64*312, 57.77], [32*208, 69.11]]

  time_speedup = [100*abs(round((x[0]-y[0])) / y[0]) for x, y in zip(groups["BenchDirect"]["data"], groups["BenchPress"]["data"])]
  acc_speedup = [100*abs(round((x[1]-y[1])) / y[1]) for x, y in zip(groups["BenchDirect"]["data"], groups["BenchPress"]["data"])]

  time_speedup = [[x, y] for x, y in zip([2048, 1024, 512, 256, 128, 64, 32], time_speedup)]
  acc_speedup = [[x, y] for x, y in zip([2048, 1024, 512, 256, 128, 64, 32], acc_speedup)]

  print(time_speedup)
  print(acc_speedup)

  plt.GroupScatterPlot(groups, plot_name="instcount")
  plt.GroupScatterPlot({"time_speedup": {'data': time_speedup, 'names': []}, "accuracy_improvement": {'data': acc_speedup, 'names': []}}, plot_name="instcount_speedup")
  """
  return groups

@public.evaluator
def MinScore(**kwargs) -> None:
  """
  Compare the closest sample per target benchmark
  for all different database groups.
  """
  if 'top_k' in kwargs:
    del kwargs['top_k']
  return KAverageScore(top_k = 1, unique_code = False, **kwargs)

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
  workspace_path = kwargs.get('workspace_path') / "analyze_beam_search" / feature_space
  workspace_path.mkdir(exist_ok = True, parents = True)

  def feats_to_list(feats: typing.Dict[str, float]) -> typing.Tuple[typing.List, typing.List]:
    k, v = list(feats.keys()), list(feats.values())
    k, v = zip(*sorted(zip(k, v)))
    k, v = list(k), list(v)
    return k, v

  stats = {}
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
      if dbg.group_name not in stats:
        stats[dbg.group_name] = {
          "zero_distance"    : 0,
          "total_epochs"     : 0,
          "best_distance"    : [],
          "singleshot_distance": [],
          "total_benchmarks" : len(benchmarks),
        }
      stats[dbg.group_name]["best_distance"].append(math.inf)
      for dp in data:
        if dp.generation_id not in score_gens:
          score_gens[dp.generation_id] = dp.sample_quality
          stats[dbg.group_name]['best_distance'][-1] = dp.sample_quality
          stats[dbg.group_name]['singleshot_distance'].append(dp.sample_quality)
        else:
          score_gens[dp.generation_id] = min(score_gens[dp.generation_id], dp.sample_quality)
          stats[dbg.group_name]['best_distance'][-1] = score_gens[dp.generation_id]

      stats[dbg.group_name]['total_epochs'] += len(list(score_gens.keys()))
      if stats[dbg.group_name]['best_distance'][-1] == 0:
        stats[dbg.group_name]['zero_distance'] += 1

      generations_score[dbg.group_name] = {
        'data': [[idx, v] for idx, v in score_gens.items()],
        'names': [x for x, _ in score_gens.items()]
      }
    ## Benchmark characterization.
    plotter.GrouppedRadar(
      groups    = radar_features,
      plot_name = "feeds_radar_{}_{}_{}".format(feature_space, benchmark.name, '-'.join([dbg.group_name.replace("BenchPress", "BP").replace("BenchDirect", "BD") for dbg in db_groups])),
      path      = workspace_path / "radar",
      title     = benchmark.name,
      # **plot_config if plot_config else {},
    )
    ## Score convergence per generation.
    plotter.GroupScatterPlot(
      groups    = generations_score,
      plot_name = "Beam_generation_{}_{}_{}".format(feature_space, benchmark.name, '-'.join([dbg.group_name.replace("BenchPress", "BP").replace("BenchDirect", "BD") for dbg in db_groups])),
      path      = workspace_path / "scatter",
      mode      = "lines+markers",
      title     = "{}, {}".format(feature_space, benchmark.name),
      **plot_config if plot_config else {},
    )
  plotter.GrouppedBars(
    groups = {
      '#zero_distanced': (
        list(stats.keys()),
        [x['zero_distance'] for x in stats.values()],
      )
    },
    plot_name = "zero_distances_{}_{}".format(feature_space, '-'.join([dbg.group_name.replace("BenchPress", "BP").replace("BenchDirect", "BD") for dbg in db_groups])),
    path = workspace_path / "stats",
    # **plot_config if plot_config else {},
  )
  plotter.GrouppedBars(
    groups = {
      '#total_epochs': (
        list(stats.keys()),
        [x['total_epochs'] for x in stats.values()],
      )
    },
    plot_name = "total_epochs_{}_{}".format(feature_space, '-'.join([dbg.group_name.replace("BenchPress", "BP").replace("BenchDirect", "BD") for dbg in db_groups])),
    path = workspace_path / "stats",
    **plot_config if plot_config else {},
  )
  # base_dist = distributions.GenericDistribution(
  #   samples = [int(x*10) for x in stats['Base']['best_distance']],
  #   log_path = workspace_path,
  #   set_name = "Base_best_dist_distr_{}".format(feature_space)
  # )
  # feat_dist = distributions.GenericDistribution(
  #   samples = [int(x*10) for x in stats['Feature_Head']['best_distance']],
  #   log_path = workspace_path,
  #   set_name = "FeatHead_best_dist_distr_{}".format(feature_space)
  # )
  # base_dist.plot()
  # feat_dist.plot()
  # (base_dist - feat_dist).plot()

  # single_base_dist = distributions.GenericDistribution(
  #   samples = [int(x*10) for x in stats['Base']['singleshot_distance']],
  #   log_path = workspace_path,
  #   set_name = "Base_single_dist_distr_{}".format(feature_space)
  # )
  # single_feat_dist = distributions.GenericDistribution(
  #   samples = [int(x*10) for x in stats['Feature_Head']['singleshot_distance']],
  #   log_path = workspace_path,
  #   set_name = "FeatHead_single_dist_distr_{}".format(feature_space)
  # )
  # single_base_dist.plot()
  # single_feat_dist.plot()
  # (single_base_dist - single_feat_dist).plot()
  return

@public.evaluator
def GenDistanceDistribution(**kwargs) -> None:
  """
  For a given beam search generation, calculate the distance distribution from the given target benchmark.

  Compare against multiple db_groups.
  """
  db_groups      = kwargs.get('db_groups')
  feature_space  = kwargs.get('feature_space')
  plot_config    = kwargs.get('plot_config')
  generation_id  = kwargs.get('generation_id')
  workspace_path = kwargs.get('workspace_path') / "gen_distance_distr" / feature_space
  workspace_path.mkdir(exist_ok = True, parents = True)

  """
  groups = {
    target: {
      db_group_name: [sample_score]
    }
  }
  """

  groups = {}
  for dbg in db_groups:
    ## Flattened list of scores distribution, sorted by target -> group
    if not dbg.db_type == evaluate_cand_database.SearchCandidateDatabase:
      raise ValueError("Beam search analysis requires SearchCandidateDatabase, but received {}", dbg.db_type)
    benchmarks = [x for x in dbg.get_data if x.generation_id == generation_id]
    for b in benchmarks:
      target = b.target_benchmark.split('\n')[0].replace("// ", "")
      if target not in groups:
        groups[target] = {dbg.group_name: []}
      elif dbg.group_name not in groups[target]:
        groups[target][dbg.group_name] = []
      groups[target][dbg.group_name] += [b.sample_score]*b.frequency

  stats = {}
  for target, groups in groups.items():
    distrs = []
    stats[target] = {}
    for name, data in groups.items():
      d = distributions.GenericDistribution([int(round(x)) for x in data if x < float('inf')], workspace_path, "{}-{}".format(target, name))
      d.plot()
      distrs.append(d)
      stats[target][name] = {}
      stats[target][name]['average'] = d.average
      stats[target][name]['median'] = d.median
      stats[target][name]['min'] = d.min
      stats[target][name]['max'] = d.max
    if len(distrs) == 2:
      diff = distrs[0] - distrs[1]
      stats[target]["likelihood"] = distrs[0] - distrs[1] < 0
      stats[target]["closer_than_the_minimum"] = distrs[0] <= distrs[1].min
      stats[target]["closer_than_2"] = distrs[0] <= distrs[1].get_sorted_index(2)
      stats[target]["closer_than_4"] = distrs[0] <= distrs[1].get_sorted_index(4)
      stats[target]["closer_than_8"] = distrs[0] <= distrs[1].get_sorted_index(8)
      stats[target]["closer_than_16"] = distrs[0] <= distrs[1].get_sorted_index(16)
      stats[target]["closer_than_32"] = distrs[0] <= distrs[1].get_sorted_index(32)
      diff.plot()

  l.logger().info(feature_space)
  l.logger().info("Avg closer than 1: {}".format(sum([x['closer_than_the_minimum'] for y, x in stats.items()]) / len(stats.keys())))
  l.logger().info("Avg closer than 2: {}".format(sum([x['closer_than_2'] for y, x in stats.items()]) / len(stats.keys())))
  l.logger().info("Avg closer than 4: {}".format(sum([x['closer_than_4'] for y, x in stats.items()]) / len(stats.keys())))
  l.logger().info("Avg closer than 8: {}".format(sum([x['closer_than_8'] for y, x in stats.items()]) / len(stats.keys())))
  l.logger().info("Avg closer than 16: {}".format(sum([x['closer_than_16'] for y, x in stats.items()]) / len(stats.keys())))
  l.logger().info("Avg closer than 32: {}".format(sum([x['closer_than_32'] for y, x in stats.items()]) / len(stats.keys())))
  with open(workspace_path / "stats.json", 'w') as outf:
    json.dump(stats, outf, indent = 2)
  return
