"""
This module specifies the range of available
downstream tasks that the committee can be trained on.

The input and output features per downstream task are defined.
"""
import sqlite3
import pathlib
import pickle
import math
import functools
import typing
import tqdm
import multiprocessing
import time
import numpy as np

import sqlalchemy as sql
from sqlalchemy.ext import declarative

from absl import app, flags

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

class GrewePredictive(Base, sqlutil.ProtoBackedMixin):
  """
  A database row representation for Grewe heuristic model training instance.
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
  # Code features, possibly directly derived from extractos.
  features             : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
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
               grewe_features       : typing.Dict[str, float],
               dataset              : str,
               cpu_transfer_time_ns : typing.List[int],
               cpu_kernel_time_ns   : typing.List[int],
               gpu_transfer_time_ns : typing.List[int],
               gpu_kernel_time_ns   : typing.List[int],
               transferred_bytes    : int,
               status               : str,
               ) -> typing.Dict[str, typing.Any]:
    return CLDriveSample(**{
      "sha256"               : crypto.sha256_str(source + dataset + str(global_size) + str(local_size)),
      "global_size"          : global_size,
      "local_size"           : local_size,
      "source"               : source,
      "features"             : grewe_features,
      "dataset"              : dataset,
      "cpu_transfer_time_ns" : '\n'.join([str(int(x)) for x in cpu_transfer_time_ns if x != 'nan']),
      "cpu_kernel_time_ns"   : '\n'.join([str(int(x)) for x in cpu_kernel_time_ns if x != 'nan']),
      "gpu_transfer_time_ns" : '\n'.join([str(int(x)) for x in gpu_transfer_time_ns if x != 'nan']),
      "gpu_kernel_time_ns"   : '\n'.join([str(int(x)) for x in gpu_kernel_time_ns if x != 'nan']),
      "transferred_bytes"    : transferred_bytes,
      "status"               : status,
      "date_added"           : datetime.datetime.utcnow(),
    })
