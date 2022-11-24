# Copyright (c) Foivos Tsimpourlas.
#
# BenchPress is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BenchPress is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
"""
API to communicate with legacy 'preamble.py' and 'model.py'
of Grewe's et al. predictive model (CGO 2013).

This API is used to convert modernized database groups
to the expected csv files by the script and also fill in
missing cldrive data.
"""
import sys
import pathlib
import typing
import math
import tqdm
import pandas as pd

from deeplearning.benchpress.corpuses import encoded
from deeplearning.benchpress.corpuses import benchmarks
from deeplearning.benchpress.experiments import public
from deeplearning.benchpress.experiments import workers
from deeplearning.benchpress.experiments import cldrive
from deeplearning.benchpress.experiments import clsmith
from deeplearning.benchpress.preprocessors import opencl
from deeplearning.benchpress.samplers import samples_database
from deeplearning.benchpress.util import crypto
from deeplearning.benchpress.util import monitors
from deeplearning.benchpress.util import logging as l
from deeplearning.benchpress.util import plotter
from deeplearning.benchpress.util import distributions

from deeplearning.benchpress.experiments.grewe import preamble

from absl import app, flags

FLAGS = flags.FLAGS

"""
1. You may insert database groups as usual to convert to csv
2. You need to introduce a systematic way to insert the amd/nvidia/clgen csv's from clgen's artifacts.
  a) Could be protobuf path arguments pointing to results workspace
"""

def DataFrameSchema() -> typing.List[str]:
  """
  Return index list of dataframe.
  """
  return [
    "benchmark",
    "dataset",
    "comp",
    "rational",
    "mem",
    "localmem",
    "coalesced",
    "atomic",
    "transfer",
    "wgsize",
    "F1:transfer/(comp+mem)",
    "F2:coalesced/mem",
    "F3:(localmem/mem)*avgws",
    "F4:comp/mem",
    "oracle",
    "runtime",
    "speedup",
    "penalty",
    "runtime_cpu",
    "ci_cpu",
    "ci_mean_cpu",
    "runtime_gpu",
    "ci_gpu",
    "ci_mean_gpu",
    "kernel_nlines",
    "kernel_size"
  ]

def ToDataFrameRow(name                 : str,
                   grewe_feats          : typing.Dict[str, float],
                   transferred_bytes    : int,
                   global_size          : int,
                   local_size           : int,
                   label                : str,
                   cpu_transfer_time_ns : int,
                   cpu_kernel_time_ns   : int,
                   gpu_transfer_time_ns : int,
                   gpu_kernel_time_ns   : int,
                   ) -> pd.DataFrame:
  """
  Convert a samples DB to a csv with the same columns found in paper's artifact.
  """
  return [
    name,
    global_size,
    grewe_feats['comp'],
    grewe_feats['rational'],
    grewe_feats['mem'],
    grewe_feats['localmem'],
    grewe_feats['coalesced'],
    grewe_feats['atomic'],
    transferred_bytes,
    local_size,
    transferred_bytes / (grewe_feats['comp'] + grewe_feats['mem']),
    grewe_feats["F2:coalesced/mem"],
    (grewe_feats['localmem'] / grewe_feats['mem']) * local_size,
    grewe_feats["F4:comp/mem"],
    label,
    min(cpu_transfer_time_ns + cpu_kernel_time_ns, gpu_transfer_time_ns + gpu_kernel_time_ns) / (10**6),
    max(cpu_transfer_time_ns + cpu_kernel_time_ns / gpu_transfer_time_ns + gpu_kernel_time_ns, gpu_transfer_time_ns + gpu_kernel_time_ns / cpu_transfer_time_ns + cpu_kernel_time_ns) / (10**6),
    min(cpu_transfer_time_ns + cpu_kernel_time_ns / gpu_transfer_time_ns + gpu_kernel_time_ns, gpu_transfer_time_ns + gpu_kernel_time_ns / cpu_transfer_time_ns + cpu_kernel_time_ns) / (10**6),
    (cpu_transfer_time_ns + cpu_kernel_time_ns) / 10**6,
    cpu_transfer_time_ns / 10**6,
    cpu_kernel_time_ns / 10**6,
    (gpu_transfer_time_ns + gpu_kernel_time_ns) / 10**6,
    gpu_transfer_time_ns / 10**6,
    gpu_kernel_time_ns / 10**6,
    0,
    0
  ]

def CSVPathToFrame(csv_path: pathlib.Path) -> pd.DataFrame:
  """
  Receive a csv path and return a dataframe.
  """
  return pd.read_csv(csv_path)

def DriveSource(src        : str,
                include    : str,
                group_name : str,
                feats      : typing.Dict[str, float],
                cldrive_db : cldrive.CLDriveExecutions,
                name       : str = None,
                no_cache   : bool = False,
                extra_args : typing.List[str] = [],
                ) -> typing.Generator:
  """
  For a given source code, drive to CLDrive and return a ready row.
  Args:
    src        : source code to process
    feats      : Grewe Feature vector of source code.
    cldrive_db : Caches cldrive executions of source code.
  """
  def median_of_list(lst: typing.List) -> int:
    """
    Return median of list
    """
    mid = len(lst) // 2
    if len(lst) == 0:
      return None
    elif len(lst) % 2 == 0:
      return (lst[mid] + lst[mid+1]) / 2 
    else:
      return lst[mid] / 2

  # for gsize in tqdm.tqdm([2**6, 2**7, 2**8, 2**10, 2**12, 2**14, 2**16, 2**18, 2**20], desc = "gsize", leave = False):
  for gsize in tqdm.tqdm([2**4, 2**8, 2**10, 2**12, 2**14, 2**16, 2**18, 2**20, 2**24, 2**28], desc = "gsize", leave = False):
    for lsize in tqdm.tqdm([2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8], desc = "lsize", leave = False):
      if lsize > gsize:
        continue

      sha = crypto.sha256_str(include + src + group_name + str(gsize) + str(lsize))
      if sha in cldrive_db.status_cache:
        if no_cache:
          cached = cldrive_db.update_and_get(
            src,
            feats,
            group_name,
            gsize,
            lsize,
            num_runs = 10000,
            timeout = 60,
            include = include,
            extra_args = extra_args
          )
        else:
          cached = cldrive_db.get_entry(
            src,
            group_name,
            gsize,
            lsize,
            include = include
          )
        if cached.status in {"CPU", "GPU"}:
          yield ToDataFrameRow(
            name                 = "{}.cl".format(sha) if name is None else name,
            grewe_feats          = feats,
            transferred_bytes    = cached.transferred_bytes,
            global_size          = gsize,
            local_size           = lsize,
            label                = cached.status,
            cpu_transfer_time_ns = median_of_list([int(float(x)) for x in cached.cpu_transfer_time_ns.split('\n') if x != 'nan']),
            cpu_kernel_time_ns   = median_of_list([int(float(x)) for x in cached.cpu_kernel_time_ns.split('\n') if x != 'nan']),
            gpu_transfer_time_ns = median_of_list([int(float(x)) for x in cached.gpu_transfer_time_ns.split('\n') if x != 'nan']),
            gpu_kernel_time_ns   = median_of_list([int(float(x)) for x in cached.gpu_kernel_time_ns.split('\n') if x != 'nan']),
          )
        else:
          yield None
      else:
        df, label = opencl.CLDriveDataFrame(
          src,
          header_file = include,
          num_runs = 10000,
          gsize = gsize,
          lsize = lsize,
          extra_args = extra_args,
          timeout = 60
        )
        cldrive_db.add_entry(
          src,
          feats,
          group_name,
          label,
          gsize,
          lsize,
          df,
          include = include
        )
        if label not in {"CPU", "GPU"}:
          yield None
        else:
          idx = 0
          transferred_bytes = float('NaN')
          while idx < len(df.transferred_bytes) and math.isnan(transferred_bytes):
            try:
              transferred_bytes = int(df.transferred_bytes[idx])
            except ValueError:
              idx += 1
          yield ToDataFrameRow(
            name                 = "{}.cl".format(sha) if name is None else name,
            grewe_feats          = feats,
            transferred_bytes    = transferred_bytes,
            global_size          = gsize,
            local_size           = lsize,
            label                = label,
            cpu_transfer_time_ns = df[df['device'].str.contains("CPU")].transfer_time_ns.median(),
            cpu_kernel_time_ns   = df[df['device'].str.contains("CPU")].kernel_time_ns.median(),
            gpu_transfer_time_ns = df[df['device'].str.contains("GPU")].transfer_time_ns.median(),
            gpu_kernel_time_ns   = df[df['device'].str.contains("GPU")].kernel_time_ns.median(),
          )

@public.evaluator
def GreweTopKCSV(**kwargs) -> None:
  """
  Sample top-K candidates for each db group to target, and store them to CSV.
  """
  db_groups     = kwargs.get('db_groups')
  cldrive_cache = kwargs.get('cldrive_cache', '')
  target        = kwargs.get('targets')
  top_k         = kwargs.get('top_k')
  unique_code   = kwargs.get('unique_code', False)
  workspace     = kwargs.get('workspace_path')
  tokenizer     = kwargs.get('tokenizer')

  cldrive_db = cldrive.CLDriveExecutions(url = "sqlite:///{}".format(pathlib.Path(cldrive_cache).resolve()), must_exist = False)

  for dbg in tqdm.tqdm(db_groups, desc = "DB Groups", leave = True):
    l.logger().info("Running {} on cldrive".format(dbg.group_name))
    if not (dbg.db_type == samples_database.SamplesDatabase or dbg.db_type == encoded.EncodedContentFiles):
      raise ValueError("Scores require SamplesDatabase or EncodedContentFiles but received", dbg.db_type)

    if dbg.db_type == clsmith.CLSmithDatabase:
      extra_args = ["-include{}".format(pathlib.Path(clsmith.CLSMITH_INCLUDE) / "CLSmith.h")]
    else:
      extra_args = []

    datapoints = []
    out_path = workspace / "{}.csv".format(dbg.group_name)

    if unique_code:
      get_data = lambda: dbg.get_unique_data_features("GreweFeatures", use_mp = False)
    else:
      get_data = lambda: dbg.get_data_features("GreweFeatures", use_mp = False)

    ## Unpack and collect benchmarks
    benchmarks = target.get_benchmarks("GreweFeatures")
    for benchmark in tqdm.tqdm(benchmarks, total = len(benchmarks), desc = "Benchmarks"):
      top_k_idx = 0
      top_k_bar = tqdm.tqdm(total = top_k, desc = "Top K cands", leave = False)
      for (src, incl, feats, dist) in tqdm.tqdm(workers.SortedSrcFeatsDistances(get_data(), benchmark.features, "GreweFeatures"), desc = "Sorted Data", leave = False):
        toggle = False
        if dbg.db_type == clsmith.CLSmithDatabase:
          src = "#include \"CLSmith.h\"\n" + src
        for row in DriveSource(src, incl, dbg.group_name, feats, cldrive_db, extra_args = extra_args):
          if row:
            toggle = True
            datapoints.append(row)
        if toggle:
          top_k_idx += 1
          top_k_bar.update(1)
        if top_k_idx >= top_k:
          break

    frame = pd.DataFrame(datapoints, columns = DataFrameSchema())
    frame.to_csv(out_path)
  return

@public.evaluator
def GreweCSV(**kwargs) -> None:
  """
  Convert database groups to CSV files that are supported by Grewe's predictive model.
  """
  db_groups   = kwargs.get('db_groups')
  cldrive_cache = kwargs.get('cldrive_cache', '')
  unique_code = kwargs.get('unique_code', False)
  workspace   = kwargs.get('workspace_path')
  tokenizer   = kwargs.get('tokenizer')

  cldrive_db = cldrive.CLDriveExecutions(url = "sqlite:///{}".format(pathlib.Path(cldrive_cache).resolve()), must_exist = False)

  for dbg in tqdm.tqdm(db_groups, desc = "DB Groups", leave = True):
    if not (dbg.db_type == samples_database.SamplesDatabase or dbg.db_type == encoded.EncodedContentFiles or dbg.db_type == clsmith.CLSmithDatabase):
      raise ValueError("Scores require SamplesDatabase or EncodedContentFiles but received", dbg.db_type)

    if dbg.db_type == clsmith.CLSmithDatabase:
      extra_args = ["-I{}".format(pathlib.Path(clsmith.CLSMITH_INCLUDE))]
    else:
      extra_args = []

    datapoints = []
    out_path = workspace / "{}.csv".format(dbg.group_name)

    if unique_code:
      get_data = lambda: dbg.get_unique_data_features("GreweFeatures", use_mp = False)
    else:
      get_data = lambda: dbg.get_data_features("GreweFeatures", use_mp = False)

    for (src, incl, feats) in tqdm.tqdm(get_data(), desc = "Src", leave = True):
      if dbg.db_type == clsmith.CLSmithDatabase:
        src = "#include \"CLSmith.h\"\n" + src
      for row in DriveSource(src, incl, dbg.group_name, feats, cldrive_db, extra_args = extra_args):
        if row:
          datapoints.append(row)
    frame = pd.DataFrame(datapoints, columns = DataFrameSchema())
    frame.to_csv(out_path)
  return

@public.evaluator
def TrainGrewe(**kwargs) -> None:
  """
  Collect CSV files in the same formate expected by 'preamble.py'
  and train Grewe et al. predictive model.
  """
  grewe_baseline = kwargs.get('grewe_baseline')
  csv_groups     = kwargs.get('csv_groups')
  plot_config    = kwargs.get('plot_config')
  workspace      = kwargs.get('workspace_path')

  speedups = {}
  accuracies = {}
  for group in csv_groups:
    R, base, enhanced, base_precision, base_recall, base_tnr, enhanced_precision, enhanced_recall, enhanced_tnr = preamble.plot_speedups_with_clgen(
      open(grewe_baseline, 'r'),
      open(group['path'], 'r'),
      synth_bench_name = group['name'],
    )
    if "GPGPU" not in speedups:
      speedups["GPGPU"] = base
      accuracies["GPGPU"] = {
        'precision': base_precision,
        'recall'   : base_recall,
        'tnr'      : base_tnr,
      }
    speedups["GPGPU+{}".format(group['name'])] = enhanced
    accuracies["GPGPU+{}".format(group['name'])] = {
      'precision': enhanced_precision,
      'recall'   : enhanced_recall,
      'tnr'      : enhanced_tnr,
    }
  l.logger().info("Predictive model speedup vs GPU static mapping for different datasets:")
  for k, v in speedups.items():
    l.logger().info("{}:  {}x speedup".format(k, round(v, 2)))

  plotter.MultiScatterLine(
    x = [[x for x in range(10)], [x for x in range(10)]],
    y = [preamble.bp_al, preamble.bp_pl],
    names = ["BenchPress_Active", "BenchPress_Passive"],
    plot_name = "Active_vs_Passive_speedup",
    path = workspace,
    **plot_config if plot_config else {}
  )
  return

@public.evaluator
def FeatureSpaceCovLabel(**kwargs) -> None:
  """
  For each baseline + ground truth, collect
  all Grewe datapoints from CSV and plot the feature
  space coverage. Points are colored based on label, CPU or GPU.
  """
  grewe_baseline = kwargs.get('grewe_baseline')
  csv_groups     = kwargs.get('csv_groups')
  plot_config    = kwargs.get('plot_config')
  workspace      = kwargs.get('workspace_path')

  base_df = CSVPathToFrame(grewe_baseline)
  base_map = {
    'CPU': base_df[base_df['oracle'] == 'CPU'].values.tolist(),
    'GPU': base_df[base_df['oracle'] == 'GPU'].values.tolist(),
  }
  tsne_mon = monitors.TSNEMonitor(
    cache_path = workspace,
    set_name = 'GPGPU',
  )
  for k in {'CPU', 'GPU'}:
    for dp in base_map[k]:
      sample = (
        dp[2:14],
        k,
      )
      tsne_mon.register(sample)
    tsne_mon.plot()

  for group in csv_groups:
    group_df = CSVPathToFrame(group['path'])
    group_map = {
      'CPU': group_df[group_df['oracle'] == 'CPU'].values.tolist(),
      'GPU': group_df[group_df['oracle'] == 'GPU'].values.tolist(),
    }
    tsne_mon = monitors.TSNEMonitor(
      cache_path = workspace,
      set_name = group['name'],
    )
    for k in {'CPU', 'GPU'}:
      for dp in group_map[k] + base_map[k]:
        sample = (
          dp[2:14],
          k,
        )
        tsne_mon.register(sample)
      tsne_mon.plot()
  return

@public.evaluator
def FeatureSpaceCovGroup(**kwargs) -> None:
  """
  For each baseline + ground truth, collect
  all Grewe datapoints from CSV and plot the feature
  space coverage.
  Points are colored based on group, everything is plotted
  in one figure.
  """
  grewe_baseline = kwargs.get('grewe_baseline')
  csv_groups     = kwargs.get('csv_groups')
  plot_config    = kwargs.get('plot_config')
  workspace      = kwargs.get('workspace_path')

  base_df = CSVPathToFrame(grewe_baseline)
  groups = {}
  csv_data = {
    'GPGPU_CPU': [(dp[10:14], "{}-{}-{}".format(dp[0], dp[1], dp[9])) for dp in base_df[base_df['oracle'] == 'CPU'].values.tolist()],
    'GPGPU_GPU': [(dp[10:14], "{}-{}-{}".format(dp[0], dp[1], dp[9])) for dp in base_df[base_df['oracle'] == 'GPU'].values.tolist()],
  }
  tsne_mon = monitors.TSNEMonitor(
    cache_path = workspace,
    set_name = 'Benchmarks_without_derived_split',
  )

  ranges = [
    [167, 250],
    [500, 533],
    [50, 167],
  ]

  print(len(base_df[base_df["F1:transfer/(comp+mem)"] == 64]) / len(base_df))
  for idx, f_dim in enumerate(range(11, 14)):
    samples = [int((dp[f_dim] * 100))  for dp in base_df.values.tolist()]
    d = distributions.GenericDistribution(samples, log_path = workspace, set_name = "GPGPU_distr_{}".format(f_dim))
    print("P[X inside range] = {}%".format(100 * (1 - (d < ranges[idx][0]) - (d > ranges[idx][1]))))
    d.plot()

  for k in ['GPGPU_CPU', 'GPGPU_GPU']:
    for dp, name in csv_data[k]:
      tsne_mon.register((dp, k, name))

  for group in csv_groups:
    # Run the predictive model and plot per group-predicted_label.
    # R, base, enhanced, base_precision, base_recall, base_tnr, enhanced_precision, enhanced_recall, enhanced_tnr = preamble.plot_speedups_with_clgen(
    #   open(grewe_baseline, 'r'),
    #   open(group['path'], 'r'),
    #   synth_bench_name = group['name'],
    # )

    # b_mask = R["training"] == "Grewe et al."
    # bs_mask = R["training"] == "w. {}".format(group['name'])

    # groups['GPGPU_correct'] = [[dp[13], dp[10]] for dp in R[b_mask][R[b_mask]['oracle'] == R[b_mask]['p']].values.tolist()]
    # groups['GPGPU_wrong'] = [[dp[13], dp[10]] for dp in R[b_mask][R[b_mask]['oracle'] != R[b_mask]['p']].values.tolist()]

    # groups['{}_correct'.format(group['name'])] = [[dp[13], dp[10]] for dp in R[bs_mask][R[bs_mask]['oracle'] == R[bs_mask]['p']].values.tolist()]
    # groups['{}_wrong'.format(group['name'])] = [[dp[13], dp[10]] for dp in R[bs_mask][R[bs_mask]['oracle'] != R[bs_mask]['p']].values.tolist()]

    group_df = CSVPathToFrame(group['path'])
    csv_data["{}_CPU".format(group['name'])] = [(dp[10:14], "{}-{}-{}".format(dp[0], dp[1], dp[9])) for dp in group_df[group_df['oracle'] == 'CPU'].values.tolist()]
    csv_data["{}_GPU".format(group['name'])] = [(dp[10:14], "{}-{}-{}".format(dp[0], dp[1], dp[9])) for dp in group_df[group_df['oracle'] == 'GPU'].values.tolist()]
    for k in ['{}_CPU'.format(group['name']), '{}_GPU'.format(group['name'])]:
      for dp, name in csv_data[k]:
        tsne_mon.register((dp, k, name))
  tsne_mon.plot()
#                                                                                         runtime    runtime_cpu                                 runtime_gpu
# npb-3.3-BT-exact_rhs5 16  4   4 8 5 2 0 768  [ 8 64  0.25  5                 0.5  ] CPU 0.036555  0.039456282295465 0.036221266726596 0.036555  0.028247  0.008308  0.039122  0.031148  0.007974  0 0
# npb-3.3-SP-rhs_norm   64  10  5 6 4 1 0 1024 [ 8 64  0.167 5.33333333333333  1.67 ] GPU 0.029257  0.032185415898677 0.027501519665707 0.030429  0.019423  0.011006  0.029257  0.021179  0.008078  0 0


# shoc-1.1.5-S3D-ratt2_kernel 16384 2426  0 702 0 0 0 262144  32  [83.8056265984655  0 0 3.46]  GPU 0.169562  0.48240488604774  0.174838721962068 0.487672  0.060578  0.427094  0.169562  0.055309  0.114253  0 0
# npb-3.3-LU-setbv3           64    14    3 4   0 0 0 1536    8   [85.3333333333333  0 0 3.5 ]  CPU 0.037712  0.041938274881693 0.034214350094968 0.037712  0.026837  0.010875  0.03844   0.031063  0.007377  0 0




  # norm_0, norm_1 = 0, 0
  # for k, v in groups.items():
  #   for dp in v:
  #     norm_0 = max(norm_0, dp[0])
  #     norm_1 = max(norm_1, dp[1])
  # for k, v in groups.items():
  #   for idx, dp in enumerate(v):
  #     groups[k][idx][0] = groups[k][idx][0] / norm_0
  #     groups[k][idx][1] = groups[k][idx][1] / norm_1

  # norm_0, norm_1 = 0, 0
  # for k, v in csv_data.items():
  #   for dp in v:
  #     norm_0 = max(norm_0, dp[0])
  #     norm_1 = max(norm_1, dp[1])
  # for k, v in csv_data.items():
  #   for idx, dp in enumerate(v):
  #     csv_data[k][idx][0] = csv_data[k][idx][0] / norm_0
  #     csv_data[k][idx][1] = csv_data[k][idx][1] / norm_1

  # for group in csv_groups:
  #   tsne_mon = monitors.TSNEMonitor(
  #     cache_path = workspace,
  #     set_name = 'GPGPU',
  #   )
  #   for k in ['GPGPU_correct', 'GPGPU_wrong']:
  #     for dp in groups[k]:
  #       tsne_mon.register((dp, k))
  #   tsne_mon.plot()

  #   tsne_mon = monitors.TSNEMonitor(
  #     cache_path = workspace,
  #     set_name = '{}'.format(group['name']),
  #   )
  #   for k in ['{}_correct'.format(group['name']), '{}_wrong'.format(group['name'])]:
  #     for dp in groups[k]:
  #       tsne_mon.register((dp, k))
  #   tsne_mon.plot()

  # plot_groups = {}
  # for k, v in groups.items():
  #   plot_groups[k] = {
  #     'data': v,
  #     'names': []
  #   }
  # plotter.GroupScatterPlot(
  #   groups    = plot_groups,
  #   plot_name = "test",
  #   path      = workspace,
  #   title     = "test",
  # )

  # plot_groups = {}
  # for k, v in csv_data.items():
  #   plot_groups[k] = {
  #     'data': v,
  #     'names': []
  #   }
  # plotter.GroupScatterPlot(
  #   groups    = plot_groups,
  #   plot_name = "test2",
  #   path      = workspace,
  #   title     = "test2",
  # )


  # for k, l in groups.items():
  #   for dp in l:
  #     tsne_mon.register((dp, k))
  # tsne_mon.plot()
  return

def fetch_gpgpu_cummins_benchmarks(gpgpu_path: pathlib.Path, cldrive_path: pathlib.Path, out_path: pathlib.Path) -> None:
  """
  Parse GPGPU folder, isolate and collect all kernel instances.
  Save to DataFrame.
  """
  if isinstance(gpgpu_path, str):
    gpgpu_path = pathlib.Path(gpgpu_path)
  if isinstance(cldrive_path, str):
    cldrive_path = pathlib.Path(cldrive_path)
  if isinstance(out_path, str):
    out_path = pathlib.Path(out_path)

  kernels = benchmarks.yield_cl_kernels(gpgpu_path)

  gpgpu_benchmarks = []
  for k in kernels:
    try:
      _ = opencl.Compile(k[1])
      b = benchmarks.benchmark_worker(k, "GreweFeatures")
      gpgpu_benchmarks.append(b)
    except ValueError:
      pass

  l.logger().info("Fetched {} GPGPU benchmarks. {} compiled successfully.".format(len(kernels), len(gpgpu_benchmarks)))

  datapoints = []
  cldrive_db = cldrive.CLDriveExecutions(url = "sqlite:///{}".format(pathlib.Path(cldrive_path).resolve()), must_exist = False)
  for k in gpgpu_benchmarks:
    name = '-'.join(str(k.path).split("gpgpu/")[-1].split('/'))
    for row in DriveSource(k.contents, "", "GPGPU_benchmarks", k.features, cldrive_db, name = name, no_cache = True):
      if row:
        datapoints.append(row)
  frame = pd.DataFrame(datapoints, columns = DataFrameSchema())
  frame.to_csv(out_path)
  return

def main(*args, **kwargs):
  fetch_gpgpu_cummins_benchmarks(sys.argv[1], sys.argv[2], sys.argv[3])
  return

if __name__ == "__main__":
  app.run(main)
  exit(0)
