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
from deeplearning.clgen.preprocessors import opencl
from deeplearning.clgen.samplers import samples_database
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

def ToDataFrameRow(name: str,
                   grewe_feats: typing.Dict[str, float],
                   df: pd.DataFrame,
                   ) -> pd.DataFrame:
  """
  Convert a samples DB to a csv with the same columns found in paper's artifact.
  """
  cts, cks, gts, gks = df[df['device'].str.contains("CPU")].transfer_time_ns.mean(), df[df['device'].str.contains("CPU")].kernel_time_ns.mean(), df[df['device'].str.contains("GPU")].transfer_time_ns.mean(), df[df['device'].str.contains("GPU")].kernel_time_ns.mean()
  return [
    name,
    df.global_size[0],
    grewe_feats['comp'],
    grewe_feats['rational'],
    grewe_feats['mem'],
    grewe_feats['localmem'],
    grewe_feats['coalesced'],
    grewe_feats['atomic'],
    df[df['device'].str.contains("CPU")].transferred_bytes[0],
    df.local_size[0],
    df[df['device'].str.contains("CPU")].transferred_bytes[0] / (grewe_feats['comp'] + grewe_feats['mem']),
    grewe_feats["F2:coalesced/mem"],
    (grewe_feats['localmem'] / grewe_feats['mem']) * df.local_size[0],
    grewe_feats["F4:comp/mem"],
    "GPU" if cts + cks > gts + gks else "CPU",
    min(cts + cks, gts + gks),
    max(cts + cks / gts + gks, gts + gks / cts + cks),
    min(cts + cks / gts + gks, gts + gks / cts + cks),
    cts + cks,
    cts,
    cks,
    gts + gks,
    gts,
    gks,
    0,
    0
  ]

def DriveSource(src: str, feats: typing.Dict[str, float], idx: int) -> typing.Generator:
  """
  For a given source code, drive to CLDrive and return a ready row.
  """
  for gsize in tqdm.tqdm([2**6, 2**7, 2**8, 2**10, 2**12, 2**14, 2**16, 2**18, 2**20], desc = "gsize", leave = False):
    for lsize in tqdm.tqdm([2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8], desc = "lsize", leave = False):
      if lsize > gsize:
        continue

      data, label = opencl.CLDriveDataFrame(src, num_runs = 400, gsize = gsize, lsize = lsize, timeout = 60)
      if label not in {"CPU", "GPU"}:
        yield None
      else:
        yield ToDataFrameRow(
            name = "{}.cl".format(idx),
            grewe_feats = feats,
            df = data,
          )

@public.evaluator
def GreweTopKCSV(**kwargs) -> None:
  """
  Sample top-K candidates for each db group to target, and store them to CSV.
  """
  db_groups   = kwargs.get('db_groups')
  target      = kwargs.get('targets')
  top_k       = kwargs.get('top_k')
  unique_code = kwargs.get('unique_code', False)
  workspace   = kwargs.get('workspace_path')
  tokenizer   = kwargs.get('tokenizer')

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
      for idx, (src, _, feats, dist) in enumerate(tqdm.tqdm(workers.SortedSrcFeatsDistances(get_data(), benchmark.features, "GreweFeatures"), desc = "Sorted Data", leave = False)):
        toggle = False
        for row in DriveSource(src, feats, idx):
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
  unique_code = kwargs.get('unique_code', False)
  workspace   = kwargs.get('workspace_path')
  tokenizer   = kwargs.get('tokenizer')

  for dbg in tqdm.tqdm(db_groups, desc = "DB Groups", leave = True):
    if not (dbg.db_type == samples_database.SamplesDatabase or dbg.db_type == encoded.EncodedContentFiles):
      raise ValueError("Scores require SamplesDatabase or EncodedContentFiles but received", dbg.db_type)

    datapoints = []
    out_path = workspace / "{}.csv".format(dbg.group_name)

    if unique_code:
      get_data = lambda: dbg.get_unique_data_features("GreweFeatures")
    else:
      get_data = lambda: dbg.get_data_features("GreweFeatures")

    for idx, (src, feats) in enumerate(tqdm.tqdm(get_data(), desc = "Src", leave = True)):
      for row in DriveSource(src, feats, idx):
        if row:
          datapoints.append(row)
    frame = pd.DataFrame(datapoints, columns = DataFrameSchema())
    frame.to_csv(out_path)

  return
