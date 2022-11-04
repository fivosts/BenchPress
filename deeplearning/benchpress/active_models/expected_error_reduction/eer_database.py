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
"""A module for databases of active learning query by committee samples."""
import datetime
import typing
import progressbar
import pathlib
import sqlite3

import sqlalchemy as sql
from sqlalchemy.ext import declarative

from deeplearning.benchpress.util import crypto
from deeplearning.benchpress.util import distrib
from deeplearning.benchpress.util import environment
from deeplearning.benchpress.util import sqlutil
from deeplearning.benchpress.util import logging as l

from absl import flags

FLAGS = flags.FLAGS

Base = declarative.declarative_base()

class EERResults(Base):
  __tablename__ = "eer)results"
  """
    DB Table for concentrated validation results.
  """
  key     : str = sql.Column(sql.String(1024), primary_key=True)
  results : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)

class EERSample(Base, sqlutil.ProtoBackedMixin):
  """A database row representing a AL model sample.

  """
  __tablename__    = "eer_samples"
  # entry id
  id               : int = sql.Column(sql.Integer,    primary_key = True)
  # unique hash of sample text
  sha256           : str = sql.Column(sql.String(64), nullable = False, index = True)
  # Sample step iteration ID.
  sample_epoch     : int = sql.Column(sql.Integer, nullable = False)
  # model's train step that generated the sample
  train_step       : int = sql.Column(sql.Integer, nullable = False)
  # Original input where the feed came from
  src              : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Runtime features of input.
  runtime_features : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Input to the model.
  input_features   : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Predicted label
  prediction       : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Date
  date_added       : datetime.datetime = sql.Column(sql.DateTime, nullable=False)

  @classmethod
  def FromArgs(cls,
               id               : int,
               sample_epoch     : int,
               train_step       : int,
               src              : str,
               runtime_features : typing.Dict[str, float],
               input_features   : typing.Dict[str, float],
               prediction       : str,
               ) -> 'EERSample':
    str_input_features   = '\n'.join(["{}:{}".format(k, v) for k, v in input_features.items()])
    str_runtime_features = '\n'.join(["{}:{}".format(k, v) for k, v in runtime_features.items()])
    sha256 = crypto.sha256_str(
      str(train_step)
      + src
      + str_runtime_features
      + str_input_features
      + prediction
    )
    return EERSample(
      id               = id,
      sha256           = sha256,
      sample_epoch     = sample_epoch,
      train_step       = train_step,
      src              = src,
      runtime_features = str_runtime_features,
      input_features   = str_input_features,
      prediction       = prediction,
      date_added       = datetime.datetime.utcnow(),
    )

class EERSamples(sqlutil.Database):
  """A database of Query-by-Committee samples."""

  def __init__(self, url: str, must_exist: bool = False, is_replica: bool = False):
    if environment.WORLD_RANK == 0 or is_replica:
      super(EERSamples, self).__init__(url, Base, must_exist = must_exist)
    if environment.WORLD_SIZE > 1 and not is_replica:
      # Conduct engine connections to replicated preprocessed chunks.
      self.base_path = pathlib.Path(url.replace("sqlite:///", "")).resolve().parent
      hash_id = self.base_path.name
      try:
        tdir = pathlib.Path(FLAGS.local_filesystem).resolve() / hash_id / "node_committee_samples"
      except Exception:
        tdir = pathlib.Path("/tmp").resolve() / hash_id / "node_committee_samples"
      try:
        tdir.mkdir(parents = True, exist_ok = True)
      except Exception:
        pass
      self.replicated_path = tdir / "samples_{}.db".format(environment.WORLD_RANK)
      self.replicated = EERSamples(
        url = "sqlite:///{}".format(str(self.replicated_path)),
        must_exist = must_exist,
        is_replica = True
      )
      distrib.barrier()
    return

  @property
  def sample_count(self):
    """Number of samples in DB."""
    with self.get_session() as s:
      count = s.query(EERSample).count()
    return count

  @property
  def get_data(self):
    """Return all database in list format"""
    with self.get_session() as s:
      return s.query(EERSample).all()

  @property
  def cur_sample_epoch(self):
    """Return the most recent checkpointed current sample step."""
    if self.sample_count > 0:
      with self.get_session() as s:
        return max([int(x.sample_epoch) for x in s.query(EERSample).all()])
    else:
      return 0

  @property
  def get_session(self):
    """
    get proper DB session.
    """
    if environment.WORLD_SIZE == 1 or environment.WORLD_RANK == 0:
      return self.Session
    else:
      return self.replicated.Session

  def add_samples(self, sample_epoch: int, samples: typing.Dict[str, typing.Any]) -> None:
    """
    If not exists, add sample to Samples table.
    """
    hash_cache = set()
    offset_idx = self.sample_count
    with self.get_session(commit = True) as s:
      for sample in samples:
        sample_entry = EERSample.FromArgs(
          id               = offset_idx,
          sample_epoch     = sample_epoch,
          train_step       = sample['train_step'],
          src              = sample['src'],
          runtime_features = sample['runtime_features'],
          input_features   = sample['input_features'],
          prediction       = sample['prediction'],
        )
        exists = s.query(EERSample).filter_by(sha256 = sample_entry.sha256).first()
        if not exists and sample_entry.sha256 not in hash_cache:
          s.add(sample_entry)
          hash_cache.add(sample_entry.sha256)
          offset_idx += 1
      s.commit()
    return
