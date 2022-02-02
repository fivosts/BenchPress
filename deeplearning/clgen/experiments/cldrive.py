"""
Evaluation script for kernel execution using cldrive or similar drivers.
"""
import tqdm
import pickle
import math

from deeplearning.clgen.preprocessors import opencl
from deeplearning.clgen.corpuses import encoded
from deeplearning.clgen.samplers import samples_database
from deeplearning.clgen.util import plotter

def TopKCLDrive(**kwargs) -> None:
  """
  Collect top-K samples per database group for each target benchmark.
  """
  db_groups      = kwargs.get('db_groups')
  target         = kwargs.get('targets')
  feature_space  = kwargs.get('feature_space')
  top_k          = kwargs.get('top_k')
  unique_code    = kwargs.get('unique_code', False)
  plot_config    = kwargs.get('plot_config')
  workspace_path = kwargs.get('workspace_path')

  groups = {}
  gsize, lsize = [2**10, 2**15, 2**20], [2**10] # 1024 is max local size for GTX1080.

  # For each db group -> for each target -> k samples -> 1) benchmark.name 2) distance 3) label.
  for dbg in db_groups:
    l.logger().info("Running {} on cldrive".format(dbg.group_name))
    if not (dbg.db_type == samples_database.SamplesDatabase or dbg.db_type == encoded.EncodedContentFiles):
      raise ValueError("Scores require SamplesDatabase or EncodedContentFiles but received", dbg.db_type)

    if unique_code:
      get_data = lambda x: dbg.get_unique_data_features(x)
    else:
      get_data = lambda x: dbg.get_data_features(x)

    ## Unpack and collect benchmarks
    benchmarks = target.get_benchmarks(feature_space)
    for benchmark in tqdm.tqdm(benchmarks, total = len(benchmarks), desc = "Benchmarks"):

      closest_src = None
      for gs in gsize:
        for ls in lsize:
          if ls > gs:
            continue

          ## Set-up number of runs.
          nruns = 10**4
          if gs > 2**13:
            nruns = 10**3
          if gs > 2**15:
            nruns = 10**2

          ## Run cldrive on benchmark.
          benchmark_label = "TimeOut"
          bench_runs = nruns
          while benchmark_label == "TimeOut" and bench_runs > 0:
            try:
              benchmark_label = opencl.CLDriveLabel(benchmark.contents, num_runs = bench_runs, gsize = gs, lsize = ls, timeout = 200)
            except TimeoutError:
              bench_runs = bench_runs // 10
          if benchmark_label not in {"CPU", "GPU"}:
            continue

          ## Fix dictionary entry.
          config = "g{}-l{}".format(gs, ls)
          if config not in groups:
            groups[config] = {}
          if dbg.group_name not in groups[config]:
            groups[config][dbg.group_name] = ([], [], [], [])

          groups[config][dbg.group_name][0].append(
            {
              'benchmark_name'     : benchmark.name,
              'benchmark_label'    : benchmark_label,
              'benchmark_contents' : benchmark.contents
            }
          )

          ## Get unique contentfiles of database group.
          if closest_src is None:
            l.logger().info(benchmark.name)
            closest_src = SortedSrcDistances(get_data(feature_space), benchmark.features, feature_space)
          l.logger().info("global size: {}, local size: {}".format(gs, ls))
          l.logger().error("Benchmark label: {}".format(benchmark_label))

          cand_idx = 0
          for idx, (src, dist) in enumerate(closest_src):
            if cand_idx >= top_k:
              break
            label  = "TimeOut"
            c_runs = nruns
            while label == "TimeOut" and c_runs > 0:
              try:
                label = opencl.CLDriveLabel(src, num_runs = c_runs, gsize = gs, lsize = ls, timeout = 200)
              except TimeoutError:
                c_runs = c_runs // 10
            if label not in {"CPU", "GPU"}:
              continue
            l.logger().error("Label: {}, distance: {}".format(label, dist))
            if len(groups[config][dbg.group_name][1]) - 1 < idx:
              groups[config][dbg.group_name][1].append([dist])
              groups[config][dbg.group_name][2].append([label])
              groups[config][dbg.group_name][3].append([src])
            else:
              groups[config][dbg.group_name][1][idx].append(dist)
              groups[config][dbg.group_name][2][idx].append(label)
              groups[config][dbg.group_name][3][idx].append(src)
            cand_idx += 1
            # Some thoughts: Maybe a dedicated plot to show distribution of execution times, etc. ?
            # In here you basically need the label.
          # Compute target's distance from O(0,0)
          # target_origin_dist = math.sqrt(sum([x**2 for x in benchmark.features.values()]))
          # avg_dist = sum([x[1] for x in closest_src]) / top_k

          # groups[config][dbg.group_name][1].append(100 * ((target_origin_dist - avg_dist) / target_origin_dist))
  print(groups)
  with open("./data_{}.pkl".format(feature_space), 'wb') as inf:
    pickle.dump(groups, inf)
  raise NotImplementedError
  return