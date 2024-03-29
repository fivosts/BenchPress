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
This module specifies the range of available
downstream tasks that the committee can be trained on.

The input and output features per downstream task are defined.
"""
import datetime
import typing
import tqdm

import sqlalchemy as sql
from sqlalchemy.ext import declarative

from deeplearning.benchpress.util import sqlutil
from deeplearning.benchpress.util import crypto

from absl import flags

FLAGS = flags.FLAGS

Base = declarative.declarative_base()

def GreweDataScheme() -> typing.List[str]:
  """
  Return schema of grewe predictive model data inputs.
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

class GreweInstance(Base, sqlutil.ProtoBackedMixin):
  """
  A database row representation for Grewe heuristic model training instance.
  """
  __tablename__    = "grewe_training_instances"
  # entry id
  id                 : int = sql.Column(sql.Integer,    primary_key = True)
  # Indexable hash
  sha256             : str = sql.Column(sql.String(64), nullable = False, index = True)
  # source code of the first occurence that created this row.
  src                : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Sampling epoch where training instance was first collected.
  sampling_epoch     : int = sql.Column(sql.Integer,   nullable = False)
  # Grewe features of kernel.
  features           : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Target grewe features of kernel.
  target_features    : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Distance from target.
  euclidean_distance : float = sql.Column(sql.Float, nullable = False)
  # Name of benchmark.
  benchmark          : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # For some reason, this is the global size.
  dataset            : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # #comp instructions.
  comp               : int = sql.Column(sql.Integer,   nullable = False)
  # #rational instructions.
  rational           : int = sql.Column(sql.Integer,   nullable = False)
  # #mem instructions.
  mem                : int = sql.Column(sql.Integer,   nullable = False)
  # #localmem instructions.
  localmem           : int = sql.Column(sql.Integer,   nullable = False)
  # #coalesced instructions.
  coalesced          : int = sql.Column(sql.Integer,   nullable = False)
  # #atomic instructions.
  atomic             : int = sql.Column(sql.Integer,   nullable = False)
  # amount of transferred bytes.
  transfer           : int = sql.Column(sql.Integer,   nullable = False)
  # work-group size as in local size.
  wgsize             : int = sql.Column(sql.Integer,   nullable = False)
  # F1:transfer/(comp+mem) score
  F1                 : float = sql.Column(sql.Float, nullable = False)
  # F2:coalesced/mem
  F2                 : float = sql.Column(sql.Float, nullable = False)
  # F3:(localmem/mem)*avgws
  F3                 : float = sql.Column(sql.Float, nullable = False)
  # F4:comp/mem
  F4                 : float = sql.Column(sql.Float, nullable = False)
  # Is CPU or GPU the best place to run this instance?
  oracle             : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Total execution time for optimal device
  runtime            : float = sql.Column(sql.Float, nullable = False)
  # How much faster is the faster device from the slower
  speedup            : float = sql.Column(sql.Float, nullable = False)
  # The inverse of speedup
  penalty            : float = sql.Column(sql.Float, nullable = False)
  # The runtime of CPU.
  runtime_cpu        : int = sql.Column(sql.Integer, nullable = False)
  # transfer time of CPU.
  ci_cpu             : int = sql.Column(sql.Integer, nullable = False)
  # kernel time of CPU.
  ci_mean_cpu        : int = sql.Column(sql.Integer, nullable = False)
  # The runtime of GPU.
  runtime_gpu        : int = sql.Column(sql.Integer, nullable = False)
  # transfer time of GPU.
  ci_gpu             : int = sql.Column(sql.Integer, nullable = False)
  # kernel time of GPU.
  ci_mean_gpu        : int = sql.Column(sql.Integer, nullable = False)
  # Number of source code lines of kernel.
  kernel_nlines      : int = sql.Column(sql.Integer, nullable = False)
  # Size of kernel in number of tokens
  kernel_size        : int = sql.Column(sql.Integer, nullable = False)
  # Date added
  date_added         : datetime.datetime = sql.Column(sql.DateTime, nullable=False)

  @classmethod
  def FromArgs(cls,
               sampling_epoch     : int,
               src                : str,
               grewe_feats        : typing.Dict[str, float],
               target_features    : typing.Dict[str, float],
               euclidean_distance : float,
               global_size        : int,
               local_size         : int,
               transferred_bytes  : int,
               oracle             : str,
               cpu_transfer_ns    : int,
               cpu_kernel_ns      : int,
               gpu_transfer_ns    : int,
               gpu_kernel_ns      : int,
               kernel_nlines      : int,
               kernel_size        : int,
               ) -> typing.Dict[str, typing.Any]:
    sha = crypto.sha256_str(
      src
      + str(grewe_feats)
      + str(target_features)
      + str(transferred_bytes)
      + str(local_size)
      + str(global_size)
      + str(oracle)
    )
    try:
      F1 = transferred_bytes / (grewe_feats['comp'] + grewe_feats['mem'])
    except ZeroDivisionError:
      F1 = 0.0
    try:
      F3 = (grewe_feats['localmem'] / grewe_feats['mem']) * local_size
    except ZeroDivisionError:
      F3 = 0.0
    return GreweInstance(**{
      "src"                : src,
      "sampling_epoch"     : sampling_epoch,
      "sha256"             : sha,
      "features"           : "\n".join(["{}:{}".format(k, v) for k, v in grewe_feats.items()]),
      "target_features"    : "\n".join(["{}:{}".format(k, v) for k, v in target_features.items()]),
      "euclidean_distance" : euclidean_distance,
      "benchmark"   : "{}-cl.A".format(sha),
      "dataset"     : global_size,
      "comp"        : grewe_feats['comp'],
      "rational"    : grewe_feats['rational'],
      "mem"         : grewe_feats['mem'],
      "localmem"    : grewe_feats['localmem'],
      "coalesced"   : grewe_feats['coalesced'],
      "atomic"      : grewe_feats['atomic'],
      "transfer"    : transferred_bytes,
      "wgsize"      : local_size,
      "F1"          : F1,
      "F2"          : grewe_feats["F2:coalesced/mem"],
      "F3"          : F3,
      "F4"          : grewe_feats["F4:comp/mem"],
      "oracle"      : oracle,
      "runtime"     : min(cpu_transfer_ns + cpu_kernel_ns, gpu_transfer_ns + gpu_kernel_ns),
      "speedup"     : max(cpu_transfer_ns + cpu_kernel_ns / gpu_transfer_ns + gpu_kernel_ns, gpu_transfer_ns + gpu_kernel_ns / cpu_transfer_ns + cpu_kernel_ns),
      "penalty"     : min(cpu_transfer_ns + cpu_kernel_ns / gpu_transfer_ns + gpu_kernel_ns, gpu_transfer_ns + gpu_kernel_ns / cpu_transfer_ns + cpu_kernel_ns),
      "runtime_cpu" : cpu_kernel_ns + cpu_transfer_ns,
      "ci_cpu"      : cpu_transfer_ns,
      "ci_mean_cpu" : cpu_kernel_ns,
      "runtime_gpu" : gpu_kernel_ns + gpu_transfer_ns,
      "ci_gpu"        : gpu_transfer_ns,
      "ci_mean_gpu"   : gpu_kernel_ns,
      "kernel_nlines" : kernel_nlines,
      "kernel_size"   : kernel_size,
      "date_added"    : datetime.datetime.utcnow(),
    })

  @classmethod
  def add_epoch(cls,
                batch           : typing.List[typing.Dict],
                sampling_epoch  : int,
                target_features : typing.Dict[str, float],
                tokenizer       : 'tokenizers.TokenizerBase',
                ) -> typing.List['GreweInstance']:
    instances = []
    for sample in batch:
      src = tokenizer.ArrayToCode(sample.sample)
      instances.append(GreweInstance.FromArgs(
        src                = src,
        sampling_epoch     = sampling_epoch,
        global_size        = sample.runtime_features['global_size'],
        grewe_feats        = sample.features,
        target_features    = target_features,
        euclidean_distance = sample.score,
        transferred_bytes  = int(sample.runtime_features['transferred_bytes']),
        local_size         = int(sample.runtime_features['local_size']),
        oracle             = sample.runtime_features['label'],
        cpu_transfer_ns    = int(sample.runtime_features['cpu_transfer_ns']),
        cpu_kernel_ns      = int(sample.runtime_features['cpu_kernel_ns']),
        gpu_transfer_ns    = int(sample.runtime_features['gpu_transfer_ns']),
        gpu_kernel_ns      = int(sample.runtime_features['gpu_kernel_ns']),
        kernel_nlines      = len(src.split('\n')),
        kernel_size        = len(src.split(' ')),
      ))
    return instances

class FeatureLessGreweInstance(Base, sqlutil.ProtoBackedMixin):
  """
  A database row representation for Grewe heuristic model training instance.
  """
  __tablename__    = "featureless_grewe_training_instances"
  # entry id
  id                 : int = sql.Column(sql.Integer,    primary_key = True)
  # Indexable hash
  sha256             : str = sql.Column(sql.String(64), nullable = False, index = True)
  # source code of the first occurence that created this row.
  src                : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Sampling epoch where training instance was first collected.
  sampling_epoch     : int = sql.Column(sql.Integer,   nullable = False)
  # Grewe features of kernel.
  features           : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Target grewe features of kernel.
  target_features    : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Distance from target.
  euclidean_distance : float = sql.Column(sql.Float, nullable = False)
  # Name of benchmark.
  benchmark          : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # For some reason, this is the global size.
  dataset            : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # #comp instructions.
  comp               : int = sql.Column(sql.Integer,   nullable = False)
  # #rational instructions.
  rational           : int = sql.Column(sql.Integer,   nullable = False)
  # #mem instructions.
  mem                : int = sql.Column(sql.Integer,   nullable = False)
  # #localmem instructions.
  localmem           : int = sql.Column(sql.Integer,   nullable = False)
  # #coalesced instructions.
  coalesced          : int = sql.Column(sql.Integer,   nullable = False)
  # #atomic instructions.
  atomic             : int = sql.Column(sql.Integer,   nullable = False)
  # amount of transferred bytes.
  transfer           : int = sql.Column(sql.Integer,   nullable = False)
  # work-group size as in local size.
  wgsize             : int = sql.Column(sql.Integer,   nullable = False)
  # F1:transfer/(comp+mem) score
  F1                 : float = sql.Column(sql.Float, nullable = False)
  # F2:coalesced/mem
  F2                 : float = sql.Column(sql.Float, nullable = False)
  # F3:(localmem/mem)*avgws
  F3                 : float = sql.Column(sql.Float, nullable = False)
  # F4:comp/mem
  F4                 : float = sql.Column(sql.Float, nullable = False)
  # Is CPU or GPU the best place to run this instance?
  oracle             : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Total execution time for optimal device
  runtime            : float = sql.Column(sql.Float, nullable = False)
  # How much faster is the faster device from the slower
  speedup            : float = sql.Column(sql.Float, nullable = False)
  # The inverse of speedup
  penalty            : float = sql.Column(sql.Float, nullable = False)
  # The runtime of CPU.
  runtime_cpu        : int = sql.Column(sql.Integer, nullable = False)
  # transfer time of CPU.
  ci_cpu             : int = sql.Column(sql.Integer, nullable = False)
  # kernel time of CPU.
  ci_mean_cpu        : int = sql.Column(sql.Integer, nullable = False)
  # The runtime of GPU.
  runtime_gpu        : int = sql.Column(sql.Integer, nullable = False)
  # transfer time of GPU.
  ci_gpu             : int = sql.Column(sql.Integer, nullable = False)
  # kernel time of GPU.
  ci_mean_gpu        : int = sql.Column(sql.Integer, nullable = False)
  # Number of source code lines of kernel.
  kernel_nlines      : int = sql.Column(sql.Integer, nullable = False)
  # Size of kernel in number of tokens
  kernel_size        : int = sql.Column(sql.Integer, nullable = False)
  # Date added
  date_added         : datetime.datetime = sql.Column(sql.DateTime, nullable=False)

  @classmethod
  def FromArgs(cls,
               sampling_epoch     : int,
               src                : str,
               features           : typing.Dict[str, float],
               grewe_feats        : typing.Dict[str, float],
               target_features    : typing.Dict[str, float],
               euclidean_distance : float,
               global_size        : int,
               local_size         : int,
               transferred_bytes  : int,
               oracle             : str,
               cpu_transfer_ns    : int,
               cpu_kernel_ns      : int,
               gpu_transfer_ns    : int,
               gpu_kernel_ns      : int,
               kernel_nlines      : int,
               kernel_size        : int,
               ) -> typing.Dict[str, typing.Any]:
    sha = crypto.sha256_str(
      src
      + str(features)
      + str(grewe_feats)
      + str(target_features)
      + str(transferred_bytes)
      + str(local_size)
      + str(global_size)
      + str(oracle)
    )
    try:
      F1 = transferred_bytes / (grewe_feats['comp'] + grewe_feats['mem'])
    except ZeroDivisionError:
      F1 = 0.0
    try:
      F3 = (grewe_feats['localmem'] / grewe_feats['mem']) * local_size
    except ZeroDivisionError:
      F3 = 0.0
    return FeatureLessGreweInstance(**{
      "src"                : src,
      "sampling_epoch"     : sampling_epoch,
      "sha256"             : sha,
      "features"           : "\n".join(["{}:{}".format(k, v) for k, v in features.items()]),
      "target_features"    : "\n".join(["{}:{}".format(k, v) for k, v in target_features.items()]),
      "euclidean_distance" : euclidean_distance,
      "benchmark" : "{}-cl.A".format(sha),
      "dataset"   : global_size,
      "comp"      : grewe_feats['comp'],
      "rational"  : grewe_feats['rational'],
      "mem"       : grewe_feats['mem'],
      "localmem"  : grewe_feats['localmem'],
      "coalesced" : grewe_feats['coalesced'],
      "atomic"    : grewe_feats['atomic'],
      "transfer"  : transferred_bytes,
      "wgsize"    : local_size,
      "F1"        : F1,
      "F2"        : grewe_feats["F2:coalesced/mem"],
      "F3"        : F3,
      "F4"        : grewe_feats["F4:comp/mem"],
      "oracle"      : oracle,
      "runtime"     : min(cpu_transfer_ns + cpu_kernel_ns, gpu_transfer_ns + gpu_kernel_ns),
      "speedup"     : max(cpu_transfer_ns + cpu_kernel_ns / gpu_transfer_ns + gpu_kernel_ns, gpu_transfer_ns + gpu_kernel_ns / cpu_transfer_ns + cpu_kernel_ns),
      "penalty"     : min(cpu_transfer_ns + cpu_kernel_ns / gpu_transfer_ns + gpu_kernel_ns, gpu_transfer_ns + gpu_kernel_ns / cpu_transfer_ns + cpu_kernel_ns),
      "runtime_cpu" : cpu_kernel_ns + cpu_transfer_ns,
      "ci_cpu"      : cpu_transfer_ns,
      "ci_mean_cpu" : cpu_kernel_ns,
      "runtime_gpu" : gpu_kernel_ns + gpu_transfer_ns,
      "ci_gpu"      : gpu_transfer_ns,
      "ci_mean_gpu" : gpu_kernel_ns,
      "kernel_nlines" : kernel_nlines,
      "kernel_size"   : kernel_size,
      "date_added"    : datetime.datetime.utcnow(),
    })

  @classmethod
  def add_epoch(cls,
                batch           : typing.List[typing.Tuple[typing.Dict, typing.Dict]],
                sampling_epoch  : int,
                target_features : typing.Dict[str, float],
                tokenizer       : 'tokenizers.TokenizerBase',
                ) -> typing.List['FeatureLessGreweInstance']:
    instances = []
    for sample, grewe_feats in batch:
      src = tokenizer.ArrayToCode(sample.sample)
      instances.append(FeatureLessGreweInstance.FromArgs(
        src                = src,
        sampling_epoch     = sampling_epoch,
        global_size        = sample.runtime_features['global_size'],
        features           = sample.features,
        grewe_feats        = grewe_feats,
        target_features    = target_features,
        euclidean_distance = sample.score,
        transferred_bytes  = int(sample.runtime_features['transferred_bytes']),
        local_size         = int(sample.runtime_features['local_size']),
        oracle             = sample.runtime_features['label'],
        cpu_transfer_ns    = int(sample.runtime_features['cpu_transfer_ns']),
        cpu_kernel_ns      = int(sample.runtime_features['cpu_kernel_ns']),
        gpu_transfer_ns    = int(sample.runtime_features['gpu_transfer_ns']),
        gpu_kernel_ns      = int(sample.runtime_features['gpu_kernel_ns']),
        kernel_nlines      = len(src.split('\n')),
        kernel_size        = len(src.split(' ')),
      ))
    return instances

class DownstreamData(sqlutil.Database):
  """Database for downstream task yielded data."""
  @property
  def count(self) -> int:
    """
    Count the number of rows in given table.
    """
    with self.Session() as s:
      return s.query(self.task_type).count()

  @property
  def sampling_epoch(self) -> int:
    """
    Return the current sample epoch.
    If DB is empty then this is 0. Otherwise it is the max+1,
    given a full sample epoch is populated at the same time.
    """
    if self.count == 0:
      return 0
    else:
      with self.Session() as s:
        return 1 + max([int(x.sampling_epoch) for x in s.query(self.task_type).all()])

  def __init__(self, url: str, task_type: typing.Callable, must_exist: bool = False):
    super(DownstreamData, self).__init__(url, Base, must_exist = must_exist)
    self.task_type = task_type
    return

  def add_epoch(self,
                batch           : typing.List[typing.Union[typing.Tuple, typing.Dict]],
                sampling_epoch  : int,
                target_features : typing.Dict[str, float],
                tokenizer       : 'tokenizers.TokenizerBase',
                ) -> None:
    """
    Add new row entry in downstream data DB.
    """
    instances = self.task_type.add_epoch(batch, sampling_epoch, target_features, tokenizer)
    with self.Session(commit = True) as ses:
      for instance in instances:
        entry = ses.query(self.task_type).filter_by(sha256 = instance.sha256).first()
        if entry is None:
          ses.add(instance)
      ses.commit()
    return
