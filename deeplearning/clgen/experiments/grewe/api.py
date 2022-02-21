"""
API to communicate with legacy 'preamble.py' and 'model.py'
of Grewe's et al. predictive model (CGO 2013).

This API is used to convert modernized database groups
to the expected csv files by the script and also fill in
missing cldrive data.
"""
import pathlib
import typing
import tqdm
import pandas as pd

from deeplearning.clgen.experiments import public
from deeplearning.clgen.experiments import workers
from deeplearning.clgen.experiments import cldrive
from deeplearning.clgen.preprocessors import opencl
from deeplearning.clgen.samplers import samples_database
from deeplearning.clgen.util import crypto
from deeplearning.clgen.util import logging as l
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

def ToDataFrameRow(name              : str,
                   grewe_feats       : typing.Dict[str, float],
                   transferred_bytes : int,
                   global_size       : int,
                   local_size        : int,
                   label             : str,
                   cpu_transfer_time : int,
                   cpu_kernel_time   : int,
                   gpu_transfer_time : int,
                   gpu_kernel_time   : int,
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
    min(cpu_transfer_time + cpu_kernel_time, gpu_transfer_time + gpu_kernel_time),
    max(cpu_transfer_time + cpu_kernel_time / gpu_transfer_time + gpu_kernel_time, gpu_transfer_time + gpu_kernel_time / cpu_transfer_time + cpu_kernel_time),
    min(cpu_transfer_time + cpu_kernel_time / gpu_transfer_time + gpu_kernel_time, gpu_transfer_time + gpu_kernel_time / cpu_transfer_time + cpu_kernel_time),
    cpu_transfer_time + cpu_kernel_time,
    cpu_transfer_time,
    cpu_kernel_time,
    gpu_transfer_time + gpu_kernel_time,
    gpu_transfer_time,
    gpu_kernel_time,
    0,
    0
  ]

def DriveSource(src: str,
                feats: typing.Dict[str, float],
                cldrive_db: cldrive.CLDriveExecutions,
                ) -> typing.Generator:
  """
  For a given source code, drive to CLDrive and return a ready row.
  Args:
    src        : source code to process
    feats      : Grewe Feature vector of source code.
    cldrive_db : Caches cldrive executions of source code.
  """
  for gsize in tqdm.tqdm([2**6, 2**7, 2**8, 2**10, 2**12, 2**14, 2**16, 2**18, 2**20], desc = "gsize", leave = False):
    for lsize in tqdm.tqdm([2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8], desc = "lsize", leave = False):
      if lsize > gsize:
        continue

      sha = crypto.sha256_str(src + str(gsize) + str(lsize))
      if sha in cldrive_db.status_cache:
        cached = cldrive_db.get_entry(src, gsize, lsize)
        if cached.status in {"CPU", "GPU"}:
          yield ToDataFrameRow(
            name              = "{}.cl".format(sha),
            grewe_feats       = feats,
            transferred_bytes = cached.transferred_bytes,
            global_size       = gsize,
            local_size        = lsize,
            label             = cached.status,
            cpu_transfer_time = cached.cpu_transfer_time_ns,
            cpu_kernel_time   = cached.cpu_kernel_time_ns,
            gpu_transfer_time = cached.gpu_transfer_time_ns,
            gpu_kernel_time   = cached.gpu_kernel_time_ns,
          )
        else:
          yield None
      else:
        df, label = opencl.CLDriveDataFrame(src, num_runs = 400, gsize = gsize, lsize = lsize, timeout = 60)
        cldrive_db.add_entry(src, label, gsize, lsize, df)
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
            name              = "{}.cl".format(sha),
            grewe_feats       = feats,
            transferred_bytes = transferred_bytes,
            global_size       = gsize,
            local_size        = lsize,
            label             = label,
            cpu_transfer_time = df[df['device'].str.contains("CPU")].transfer_time_ns.mean(),
            cpu_kernel_time   = df[df['device'].str.contains("CPU")].kernel_time_ns.mean(),
            gpu_transfer_time = df[df['device'].str.contains("GPU")].transfer_time_ns.mean(),
            gpu_kernel_time   = df[df['device'].str.contains("GPU")].kernel_time_ns.mean(),
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
      get_data = lambda: dbg.get_unique_data_features("GreweFeatures")
    else:
      get_data = lambda: dbg.get_data_features("GreweFeatures")

    ## Unpack and collect benchmarks
    benchmarks = target.get_benchmarks("GreweFeatures")
    for benchmark in tqdm.tqdm(benchmarks, total = len(benchmarks), desc = "Benchmarks"):
      top_k_idx = 0
      top_k_bar = tqdm.tqdm(total = top_k, desc = "Top K cands", leave = True)
      for (src, _, feats, dist) in tqdm.tqdm(workers.SortedSrcFeatsDistances(get_data(), benchmark.features, "GreweFeatures"), desc = "Sorted Data", leave = False):
        toggle = False
        for row in DriveSource(src, feats, cldrive_db):
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

  cldrive_db = CLDriveExecutions(url = "sqlite:///{}".format(pathlib.Path(cldrive_cache).resolve()), must_exist = False)

  for dbg in tqdm.tqdm(db_groups, desc = "DB Groups", leave = True):
    if not (dbg.db_type == samples_database.SamplesDatabase or dbg.db_type == encoded.EncodedContentFiles):
      raise ValueError("Scores require SamplesDatabase or EncodedContentFiles but received", dbg.db_type)

    datapoints = []
    out_path = workspace / "{}.csv".format(dbg.group_name)

    if unique_code:
      get_data = lambda: dbg.get_unique_data_features("GreweFeatures")
    else:
      get_data = lambda: dbg.get_data_features("GreweFeatures")

    for (src, feats) in tqdm.tqdm(get_data(), desc = "Src", leave = True):
      for row in DriveSource(src, feats, cldrive_db):
        if row:
          datapoints.append(row)
    frame = pd.DataFrame(datapoints, columns = DataFrameSchema())
    frame.to_csv(out_path)
  return
