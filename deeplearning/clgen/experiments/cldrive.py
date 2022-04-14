"""
Evaluation script for kernel execution using cldrive or similar drivers.
"""
import datetime
import sqlite3
import tqdm
import pickle
import math
import pathlib
import typing
import pandas as pd

import sqlalchemy as sql
from sqlalchemy.ext import declarative

from deeplearning.clgen.preprocessors import opencl
from deeplearning.clgen.corpuses import encoded
from deeplearning.clgen.samplers import samples_database
from deeplearning.clgen.util import plotter
from deeplearning.clgen.util import sqlutil
from deeplearning.clgen.util import crypto
from deeplearning.clgen.util import distributions
# from deeplearning.clgen.util import cldrive_server
from deeplearning.clgen.util import logging as l
from deeplearning.clgen.experiments import workers
from deeplearning.clgen.experiments import public

from absl import flags

Base = declarative.declarative_base()

FLAGS = flags.FLAGS

# flags.DEFINE_string(
#   "remote_cldrive_cache",
#   None,
#   "Set reachable address of cldrive cache. If None, cldrive cache is considered to reside in the local machine."
#   "If set, computation happens in local machine, caching in get is de-activated and computed samples are sent to remote."
# )

class Data(Base):
  __tablename__ = "sampling_results"
  """
    DB Table for concentrated validation results.
  """
  key     : str = sql.Column(sql.String(1024), primary_key=True)
  results : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)

class CLDriveSample(Base, sqlutil.ProtoBackedMixin):
  """
  A database row representing a CLDrive execution trace.
  """
  __tablename__    = "cldrive_samples"
  # entry id
  id                   : int = sql.Column(sql.Integer,    primary_key = True)
  # unique hash of cldrive execution.
  sha256               : str = sql.Column(sql.String(64), nullable = False, index = True)
  # Global size of execution
  global_size          : int = sql.Column(sql.Integer,   nullable = False)
  # Local size of execution
  local_size           : int = sql.Column(sql.Integer,   nullable = False)
  # Executed source code
  source               : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Name of dataset where this sample comes from.
  dataset              : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # cpu transfer time of kernel
  cpu_transfer_time_ns : str = sql.Column(sql.Integer,   nullable = False)
  # cpu execution time of kernel
  cpu_kernel_time_ns   : str = sql.Column(sql.Integer,   nullable = False)
  # gpu transfer time of kernel
  gpu_transfer_time_ns : str = sql.Column(sql.Integer,   nullable = False)
  # gpu execution time of kernel
  gpu_kernel_time_ns   : str = sql.Column(sql.Integer,   nullable = False)
  # amount of transferred bytes
  transferred_bytes    : int = sql.Column(sql.Integer,   nullable = False)
  # Whether cldrive executes correctly or not.
  status               : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Date
  date_added           : datetime.datetime = sql.Column(sql.DateTime, nullable=False)

  @classmethod
  def FromArgs(cls,
               id                   : int,
               global_size          : int,
               local_size           : int,
               source               : str,
               dataset              : str,
               cpu_transfer_time_ns : typing.List[int],
               cpu_kernel_time_ns   : typing.List[int],
               gpu_transfer_time_ns : typing.List[int],
               gpu_kernel_time_ns   : typing.List[int],
               transferred_bytes    : int,
               status               : str,
               ) -> typing.Dict[str, typing.Any]:
    return CLDriveSample(**{
      # "id"                   : id,
      "sha256"               : crypto.sha256_str(source + dataset + str(global_size) + str(local_size)),
      "global_size"          : global_size,
      "local_size"           : local_size,
      "source"               : source,
      "dataset"              : dataset,
      "cpu_transfer_time_ns" : '\n'.join([str(int(x)) for x in cpu_transfer_time_ns if x != 'nan']),
      "cpu_kernel_time_ns"   : '\n'.join([str(int(x)) for x in cpu_kernel_time_ns if x != 'nan']),
      "gpu_transfer_time_ns" : '\n'.join([str(int(x)) for x in gpu_transfer_time_ns if x != 'nan']),
      "gpu_kernel_time_ns"   : '\n'.join([str(int(x)) for x in gpu_kernel_time_ns if x != 'nan']),
      "transferred_bytes"    : transferred_bytes,
      "status"               : status,
      "date_added"           : datetime.datetime.utcnow(),
    })

class CLDriveExecutions(sqlutil.Database):
  """A database of CLDrive Execution samples."""

  @property
  def count(self):
    """Number of cldrive traces in DB."""
    with self.Session() as s:
      count = s.query(CLDriveSample).count()
    return count

  @property
  def status_cache(self):
    """Return list of tuples [hash, status]"""
    if self._status_cache is None:
      with self.Session() as s:
        self._status_cache = {f.sha256: f.status for f in s.query(CLDriveSample).yield_per(1000)}
    return self._status_cache

  # @property
  # def get_session(self):
  #   """
  #   Return the correct session for the cache.
  #   """
  #   if FLAGS.remote_cldrive_cache is None:
  #     return self.Session
  #   else:
  #     return self.remote_session

  def __init__(self, url: str, must_exist: bool = False):
    super(CLDriveExecutions, self).__init__(url, Base, must_exist = must_exist)
    self._status_cache = None
    # if FLAGS.remote_cldrive_cache is not None:
      # self.remote_session = cldrive_server.RemoteSession(FLAGS.remote_cldrive_cache)

  def add_entry(self, src: str, dataset: str, status: str, global_size: int, local_size: int, df: pd.DataFrame, include: str = "") -> None:
    """
    Adds execution entries from pandas dataframe.
    """
    sha = crypto.sha256_str(include + src + dataset + str(global_size) + str(local_size))
    try:
      with self.Session(commit = True) as session:
        entry = session.query(CLDriveSample).filter_by(sha256 = sha).first()
        if entry is None:
          if status in {"CPU", "GPU"}:
            idx = 0
            transferred_bytes = float('NaN')
            while idx < len(df.transferred_bytes) and math.isnan(transferred_bytes):
              try:
                transferred_bytes = int(df.transferred_bytes[idx])
              except ValueError:
                idx += 1
            try:
              session.add(
                CLDriveSample.FromArgs(
                  id          = self.count,
                  global_size = global_size,
                  local_size  = local_size,
                  source      = include + src,
                  dataset     = dataset,
                  cpu_transfer_time_ns = list(df[df['device'].str.contains("CPU")].transfer_time_ns),
                  cpu_kernel_time_ns   = list(df[df['device'].str.contains("CPU")].kernel_time_ns),
                  gpu_transfer_time_ns = list(df[df['device'].str.contains("GPU")].transfer_time_ns),
                  gpu_kernel_time_ns   = list(df[df['device'].str.contains("GPU")].kernel_time_ns),
                  transferred_bytes    = transferred_bytes,
                  status               = status,
                )
              )
            except ValueError as e:
              l.logger().warn(e)
              l.logger().warn("Skip adding entry,")
              pass
          else:
            session.add(
              CLDriveSample.FromArgs(
                id          = self.count,
                global_size = global_size,
                local_size  = local_size,
                source      = include + src,
                dataset     = dataset,
                cpu_transfer_time_ns = [],
                cpu_kernel_time_ns   = [],
                gpu_transfer_time_ns = [],
                gpu_kernel_time_ns   = [],
                transferred_bytes    = -1,
                status               = status,
              )
            )
          if self._status_cache is not None:
            assert sha not in self._status_cache, "{} should not be in DB".format(sha)
            self._status_cache[sha] = status
        elif status in {"CPU", "GPU"}:
          entry.cpu_transfer_time_ns = entry.cpu_transfer_time_ns + "\n" + '\n'.join([str(x) for x in df[df['device'].str.contains("CPU")].transfer_time_ns if x != 'nan'])
          entry.cpu_kernel_time_ns   = entry.cpu_kernel_time_ns   + "\n" + '\n'.join([str(x) for x in df[df['device'].str.contains("CPU")].kernel_time_ns if x != 'nan'])
          entry.gpu_transfer_time_ns = entry.gpu_transfer_time_ns + "\n" + '\n'.join([str(x) for x in df[df['device'].str.contains("GPU")].transfer_time_ns if x != 'nan'])
          entry.gpu_kernel_time_ns   = entry.gpu_kernel_time_ns   + "\n" + '\n'.join([str(x) for x in df[df['device'].str.contains("GPU")].kernel_time_ns if x != 'nan'])
        session.commit()
    except Exception as e:
      raise e
    return

  def get_entry(self,
                src         : str,
                dataset     : str,
                global_size : int,
                local_size  : int,
                include     : str = "",
                ) -> "CLDriveSample":
    """
    Fetch row from DB, if exists.
    """
    sha = crypto.sha256_str(include + src + dataset + str(global_size) + str(local_size))
    try:
      with self.Session() as session:
        entry = session.query(CLDriveSample).filter_by(sha256 = sha).first()
        if entry is not None:
          return entry
        else:
          return None
    except Exception as e:
      l.logger().error(e)
    return

  def update_and_get(self,
                     src         : str,
                     dataset     : str,
                     global_size : int,
                     local_size  : int,
                     num_runs    : int,
                     timeout     : int = 0,
                     include     : str = "",
                     extra_args  : typing.List[str] = [],
                     ) -> "CLDriveSample":
    """
    Add or update incoming entry by running CLDrive and pinging the database.
    """
    df, label = opencl.CLDriveDataFrame(
      src,
      header_file = include,
      num_runs    = num_runs,
      gsize       = global_size,
      lsize       = local_size,
      extra_args  = extra_args,
      timeout     = timeout,
    )
    self.add_entry(include + src, dataset, label, global_size, local_size, df)
    return self.get_entry(src, dataset, global_size, local_size)

  def get_valid_data(self, dataset: str = None) -> typing.List[CLDriveSample]:
    """
    Return all valid entries, labelled either as CPU or GPU.
    """
    with self.Session() as session:
      if dataset:
        return session.query(
          CLDriveSample
        ).filter(
          sql.and_(
            CLDriveSample.dataset == dataset,
            CLDriveSample.status.in_({"CPU", "GPU"})
          )
        ).yield_per(1000)
      else:
        return session.query(
          CLDriveSample
        ).filter(
          CLDriveSample.status.in_({"CPU", "GPU"})
        ).yield_per(1000)

  def get_execution_times_ms(self, src: str, dataset: str, global_size: int, local_size: int) -> typing.Tuple[typing.List[int], typing.List[int], typing.List[int], typing.List[int]]:
    """
    Search code by hash and return lists with all different execution times.
    """
    sha = crypto.sha256_str(src + dataset + str(global_size) + str(local_size))
    ctt, ckt, gtt, gkt = [], [], [], []
    with self.Session() as session:
      entry = session.query(CLDriveSample).filter_by(sha256 = sha).first()
      if entry is None:
        return None
      else:
        ctt = [int(x) // 1000 for x in entry.cpu_transfer_time_ns.split('\n')]
        ckt = [int(x) // 1000 for x in entry.cpu_kernel_time_ns.split('\n')]
        gtt = [int(x) // 1000 for x in entry.gpu_transfer_time_ns.split('\n')]
        gkt = [int(x) // 1000 for x in entry.gpu_kernel_time_ns.split('\n')]
    return ctt, ckt, gtt, gkt

def ComputeLabel(cpu_transfer : typing.List[int],
                 cpu_execute  : typing.List[int],
                 gpu_transfer : typing.List[int],
                 gpu_execute  : typing.List[int],
                 workspace    : pathlib.Path,
                 ) -> typing.Dict[str, float]:
  """
  Collects execution metrics of kernels, computes statistical
  distribution of execution times and returns optimal device
  to execute with certainty metrics.
  """
  cput_dist = distributions.GenericDistribution(cpu_transfer, workspace, "cpu_transfer_time")
  cpue_dist = distributions.GenericDistribution(cpu_execute, workspace, "cpu_execution_time")
  gput_dist = distributions.GenericDistribution(gpu_transfer, workspace, "gpu_transfer_time")
  gpue_dist = distributions.GenericDistribution(gpu_execute, workspace, "gpu_execution_time")

  ## P[CPUt + CPUe] and P[GPUt + GPUe].
  cpu_dist = cput_dist + cpue_dist
  gpu_dist = gput_dist + gpue_dist

  ## P[CPU - GPU]
  dist = cput_dist - gpu_dist

  return {
    "CPU": round(100 * (dist < 0), 2),
    "GPU": round(100 * (dist > 0), 2),
  }

@public.evaluator
def TopKCLDrive(**kwargs) -> None:
  """
  Collect top-K samples per database group for each target benchmark.
  """
  db_groups      = kwargs.get('db_groups')
  cldrive_cache  = kwargs.get('cldrive_cache', '')
  target         = kwargs.get('targets')
  feature_space  = kwargs.get('feature_space')
  top_k          = kwargs.get('top_k')
  unique_code    = kwargs.get('unique_code', False)
  plot_config    = kwargs.get('plot_config')
  workspace_path = kwargs.get('workspace_path')

  groups = {}
  gsize, lsize = [2**10, 2**15, 2**20], [2**10] # 1024 is max local size for GTX1080.

  cldrive_db = CLDriveExecutions(url = "sqlite:///{}".format(pathlib.Path(cldrive_cache).resolve()), must_exist = False)

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
    for idx, benchmark in enumerate(tqdm.tqdm(benchmarks, total = len(benchmarks), desc = "Benchmarks")):
      # if feature_space == "AutophaseFeatures" and (idx < 25 or benchmark.name == "particle_double.cl-1" or benchmark.name == "particle_double.cl-3" or benchmark.name == "particle_naive.cl" or benchmark.name == "particle_single.cl-1"):
      #   continue
      closest_src = None
      for gs in gsize:
        for ls in lsize:
          if ls > gs:
            continue
          ## Run cldrive on benchmark.
          benchmark_label = "TimeOut"
          nruns = 50
          bench_runs = nruns
          try:
            df, benchmark_label = opencl.CLDriveDataFrame(benchmark.contents, num_runs = bench_runs, gsize = gs, lsize = ls, timeout = 200)
          except TimeoutError:
            pass

          cldrive_db.add_entry(benchmark.contents, target.target, benchmark_label, gs, ls, df)
          if benchmark_label not in {"CPU", "GPU"}:
            continue

          times = cldrive_db.get_execution_times_ms(benchmark.contents, target.target, gs, ls)
          if times:
            ctt, ckt, gtt, gkt = times
            prob_labels = ComputeLabel(ctt, ckt, gtt, gkt, workspace_path)
          else:
            raise ValueError("Why can you not find a file you just inserted ?")

          ## Fix dictionary entry.
          config = "g{}-l{}".format(gs, ls)
          if config not in groups:
            groups[config] = {}
          if dbg.group_name not in groups[config]:
            groups[config][dbg.group_name] = ([], [], [], [])

          groups[config][dbg.group_name][0].append(
            {
              'benchmark_name'     : benchmark.name,
              'benchmark_label'    : "CPU:{}/GPU:{}".format(prob_labels['CPU'], prob_labels['GPU']),
              'benchmark_contents' : benchmark.contents
            }
          )

          ## Get unique contentfiles of database group.
          if closest_src is None:
            l.logger().info(benchmark.name)
            closest_src = workers.SortedSrcDistances(get_data(feature_space), benchmark.features, feature_space)
          l.logger().info("global size: {}, local size: {}".format(gs, ls))
          l.logger().error("Benchmark label: {}, prob: {}".format(benchmark_label, "CPU:{}/GPU:{}".format(prob_labels['CPU'], prob_labels['GPU'])))

          cand_idx = 0
          for idx, (src, incl, dist) in enumerate(closest_src):
            if cand_idx >= top_k:
              break
            label  = "TimeOut"
            c_runs = nruns
            try:
              df, label = opencl.CLDriveDataFrame(incl + src, num_runs = c_runs, gsize = gs, lsize = ls, timeout = 200)
            except TimeoutError:
              pass
            cldrive_db.add_entry(incl + src, dbg.group_name, label, gs, ls, df)

            if label not in {"CPU", "GPU"}:
              continue
            times = cldrive_db.get_execution_times_ms(incl + src, dbg.group_name, gs, ls)
            if times:
              ctt, ckt, gtt, gkt = times
              prob_labels = ComputeLabel(ctt, ckt, gtt, gkt, workspace_path)
            else:
              raise ValueError("Why can you not find a file you just inserted ?")

            l.logger().error("Label: {}, probs: {}, distance: {}".format(label, "CPU:{}/GPU:{}".format(prob_labels['CPU'], prob_labels['GPU']), dist))
            if len(groups[config][dbg.group_name][1]) - 1 < idx:
              groups[config][dbg.group_name][1].append([dist])
              groups[config][dbg.group_name][2].append(["CPU:{}/GPU:{}".format(prob_labels['CPU'], prob_labels['GPU'])])
              groups[config][dbg.group_name][3].append([incl + src])
            else:
              groups[config][dbg.group_name][1][idx].append(dist)
              groups[config][dbg.group_name][2][idx].append("CPU:{}/GPU:{}".format(prob_labels['CPU'], prob_labels['GPU']))
              groups[config][dbg.group_name][3][idx].append(incl + src)
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
  return
