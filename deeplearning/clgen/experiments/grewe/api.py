"""
API to communicate with legacy 'preamble.py' and 'model.py'
of Grewe's et al. predictive model (CGO 2013).

This API is used to convert modernized database groups
to the expected csv files by the script and also fill in
missing cldrive data.
"""
import sys
import pathlib
import tempfile
import typing
import math
import tqdm
import pandas as pd

from deeplearning.clgen.corpuses import encoded
from deeplearning.clgen.corpuses import benchmarks
from deeplearning.clgen.experiments import public
from deeplearning.clgen.experiments import workers
from deeplearning.clgen.experiments import cldrive
from deeplearning.clgen.experiments import clsmith
from deeplearning.clgen.preprocessors import opencl
from deeplearning.clgen.samplers import samples_database
from deeplearning.clgen.util import crypto
from deeplearning.clgen.util import logging as l
from deeplearning.clgen.experiments.grewe import preamble

from absl import app, flags

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

def DriveSource(src        : str,
                include    : str,
                group_name : str,
                feats      : typing.Dict[str, float],
                cldrive_db : cldrive.CLDriveExecutions,
                name       : str = None
                ) -> typing.Generator:
  """
  For a given source code, drive to CLDrive and return a ready row.
  Args:
    src        : source code to process
    feats      : Grewe Feature vector of source code.
    cldrive_db : Caches cldrive executions of source code.
  """
  # for gsize in tqdm.tqdm([2**6, 2**7, 2**8, 2**10, 2**12, 2**14, 2**16, 2**18, 2**20], desc = "gsize", leave = False):
  for gsize in tqdm.tqdm([2**10, 2**12, 2**14, 2**16, 2**18, 2**20], desc = "gsize", leave = False):
    for lsize in tqdm.tqdm([2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8], desc = "lsize", leave = False):
      if lsize > gsize:
        continue

      sha = crypto.sha256_str(include + src + group_name + str(gsize) + str(lsize))
      if sha in cldrive_db.status_cache:
        cached = cldrive_db.get_entry(src, group_name, gsize, lsize, include = include)
        if cached.status in {"CPU", "GPU"}:
          yield ToDataFrameRow(
            name                 = "{}.cl".format(sha) if name is None else name,
            grewe_feats          = feats,
            transferred_bytes    = cached.transferred_bytes,
            global_size          = gsize,
            local_size           = lsize,
            label                = cached.status,
            cpu_transfer_time_ns = sum([int(float(x)) for x in cached.cpu_transfer_time_ns.split('\n') if x != 'nan']) // len([x for x in cached.cpu_transfer_time_ns.split('\n') if x != 'nan']),
            cpu_kernel_time_ns   = sum([int(float(x)) for x in cached.cpu_kernel_time_ns.split('\n') if x != 'nan'])   // len([x for x in cached.cpu_kernel_time_ns.split('\n') if x != 'nan']),
            gpu_transfer_time_ns = sum([int(float(x)) for x in cached.gpu_transfer_time_ns.split('\n') if x != 'nan']) // len([x for x in cached.gpu_transfer_time_ns.split('\n') if x != 'nan']),
            gpu_kernel_time_ns   = sum([int(float(x)) for x in cached.gpu_kernel_time_ns.split('\n') if x != 'nan'])   // len([x for x in cached.gpu_kernel_time_ns.split('\n') if x != 'nan']),
          )
        else:
          yield None
      else:
        df, label = opencl.CLDriveDataFrame(src, header_file = include, num_runs = 1000, gsize = gsize, lsize = lsize, timeout = 60)
        cldrive_db.add_entry(src, group_name, label, gsize, lsize, df, include = include)
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
            cpu_transfer_time_ns = df[df['device'].str.contains("CPU")].transfer_time_ns.mean(),
            cpu_kernel_time_ns   = df[df['device'].str.contains("CPU")].kernel_time_ns.mean(),
            gpu_transfer_time_ns = df[df['device'].str.contains("GPU")].transfer_time_ns.mean(),
            gpu_kernel_time_ns   = df[df['device'].str.contains("GPU")].kernel_time_ns.mean(),
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
        for row in DriveSource(src, incl, dbg.group_name, feats, cldrive_db):
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

    datapoints = []
    out_path = workspace / "{}.csv".format(dbg.group_name)

    if unique_code:
      get_data = lambda: dbg.get_unique_data_features("GreweFeatures", use_mp = False)
    else:
      get_data = lambda: dbg.get_data_features("GreweFeatures", use_mp = False)

    for (src, incl, feats) in tqdm.tqdm(get_data(), desc = "Src", leave = True):
      for row in DriveSource(src, incl, dbg.group_name, feats, cldrive_db):
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

  try:
    tdir = FLAGS.local_filesystem
  except Exception:
    tdir = None

  csv_contents = open(grewe_baseline, 'r').readlines()
  for group in csv_groups:
    preamble.plot_speedups_with_clgen(
      open(grewe_baseline, 'r'),
      open(group['path'], 'r'),
      synth_bench_name = group['name'],
    )
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
    for row in DriveSource(k.contents, "", "GPGPU_benchmarks", k.features, cldrive_db, name = name):
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
