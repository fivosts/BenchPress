# coding=utf-8
# Copyright 2022 Foivos Tsimpourlas and Chris Cummins.
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
"""A module for databases of CLgen samples."""
import contextlib
import pathlib
import datetime
import typing
import progressbar
import tqdm
import sqlite3
from google.cloud import bigquery

import sqlalchemy as sql
from sqlalchemy.ext import declarative
from deeplearning.benchpress.util import sqlutil

from absl import app, flags
from deeplearning.benchpress.util import monitors
from deeplearning.benchpress.util import logging as l

FLAGS = flags.FLAGS

flags.DEFINE_string(
  "bq_database",
  None,
  "Insert path of BigQuery's database."
)

flags.DEFINE_integer(
  "chunkify",
  None,
  "Select chunkifying factor to split BQ database into sub-databases to perform pseudo-distributed preprocessing."
)

Base = declarative.declarative_base()

class bqData(Base):
  __tablename__ = "data"
  """
    DB Table for concentrated validation results.
  """
  key   : str = sql.Column(sql.String(1024), primary_key=True)
  value : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)

  @staticmethod
  def bqSchema() -> typing.List[bigquery.SchemaField]:
    return [
      bigquery.SchemaField("key",   "STRING", mode = "REQUIRED"),
      bigquery.SchemaField("value", "STRING", mode = "REQUIRED"),
    ]

class bqFile():
  """
    A database entry representing a CLgen validation trace.
  """
  id             : int = sql.Column(sql.String(64),    primary_key = True)
  repo_name      : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  ref            : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  path           : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  size           : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  content        : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  date_added     : datetime.datetime = sql.Column(sql.DateTime, nullable=False)

  @classmethod
  def FromArgs(cls,
               row: bigquery.Row
               ) -> typing.Dict[str, typing.Any]:

    return {
      "id"             : row['id'],
      "repo_name"      : row['repo_name'],
      "ref"            : row['ref'],
      "path"           : row['path'],
      "size"           : row['size']           if row['size']           else "None",
      "content"        : row['content']        if row['content']        else "None",
      "date_added"     : datetime.datetime.utcnow(),
    }

  @staticmethod
  def bqSchema() -> typing.List[bigquery.SchemaField]:
    return [
      bigquery.SchemaField("id",             "STRING",  mode = "REQUIRED"),
      bigquery.SchemaField("repo_name",      "STRING",  mode = "REQUIRED"),
      bigquery.SchemaField("ref",            "STRING",  mode = "REQUIRED"),
      bigquery.SchemaField("path",           "STRING",  mode = "REQUIRED"),
      bigquery.SchemaField("size",           "INTEGER", mode = "REQUIRED"),
      bigquery.SchemaField("content",        "STRING",  mode = "REQUIRED"),
    ]

  def ToJSONDict(self) -> typing.Dict[str, typing.Any]:
    return {
      "id"             : self.id,
      "repo_name"      : self.repo_name,
      "ref"            : self.ref,
      "path"           : self.path,
      "size"           : self.size,
      "content"        : self.content,
      "date_added"     : str(self.date_added.strftime("%m/%d/%Y, %H:%M:%S")),
    }

class bqMainFile(Base, bqFile):
  """Abstract representation of main queried files."""
  __tablename__  = "main_files"

class bqOtherFile(Base, bqFile):
  """Abstract representation of other-to-main-language queried files."""
  __tablename__  = "other_files"

class bqRepo(Base):
  """
    A database entry representing a CLgen validation trace.
  """
  __tablename__  = "repositories"
  id             : int = sql.Column(sql.Integer, primary_key = True)
  repo_name      : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  ref            : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  date_added     : datetime.datetime = sql.Column(sql.DateTime, nullable=False)

  @classmethod
  def FromArgs(cls,
               id: int,
               row: bigquery.Row
               ) -> typing.Dict[str, typing.Any]:
    return {
      "id"             : id,
      "repo_name"      : row['repo_name'],
      "ref"            : row['ref'],
      "date_added"     : datetime.datetime.utcnow(),
    }

  @staticmethod
  def bqSchema() -> typing.List[bigquery.SchemaField]:
    return [
      bigquery.SchemaField("repo_name", "STRING", mode = "REQUIRED"),
      bigquery.SchemaField("ref",       "STRING", mode = "REQUIRED"),
    ]

  def ToJSONDict(self) -> typing.Dict[str, typing.Any]:
    return {
      "id"             : self.id,
      "repo_name"      : self.repo_name,
      "ref"            : self.ref,
      "date_added"     : str(self.date_added.strftime("%m/%d/%Y, %H:%M:%S")),
    }

class bqDatabase(sqlutil.Database):
  """A database of BigQuery contentfiles."""

  def __init__(self, url: str, must_exist: bool = False):
    super(bqDatabase, self).__init__(url, Base, must_exist = must_exist)

  def main_files_batch(self, limit: int, offset: int, exclude_id: typing.Set[str] = set()) -> typing.List[bqMainFile]:
    with self.Session() as s:
      total_batch = s.query(bqMainFile).limit(limit).offset(offset).all()
      batch = [f for f in total_batch if f.id not in exclude_id]
      if exclude_id:
        for vid in [f.id for f in total_batch if f.id in exclude_id]:
          exclude_id.remove(vid)
      return batch

  ##### Main file properties
  @property
  def main_files(self) -> typing.List[bqMainFile]:
    with self.Session() as s:
      return s.query(bqMainFile).yield_per(100000)

  @property
  def mainfile_entries(self) -> typing.Set[typing.Tuple[str, str]]:
    with self.Session() as s:
      return set(s.query(bqMainFile.repo_name, bqMainFile.path).all())

  @property
  def main_ids(self) -> typing.Set[str]:
    with self.Session() as s:
      return set(x[0] for x in s.query(bqMainFile.id).all())

  @property
  def mainfile_count(self) -> int:
    with self.Session() as s:
      return s.query(bqMainFile).count()

  @property
  def main_repo_count(self) -> int:
    with self.Session() as s:
      return s.query(bqMainFile.repo_name, bqMainFile.ref).distinct().count()

  ##### Other file properties
  @property
  def other_files(self) -> typing.List[bqOtherFile]:
    with self.Session() as s:
      return s.query(bqOtherFile).all()

  @property
  def otherfile_entries(self) -> typing.Set[typing.Tuple[str, str]]:
    with self.Session() as s:
      return set(s.query(bqOtherFile.repo_name, bqOtherFile.path).all())

  @property
  def other_ids(self) -> typing.Set[str]:
    with self.Session() as s:
      return set(x[0] for x in s.query(bqOtherFile.id).all())
  
  @property
  def otherfile_count(self) -> int:
    with self.Session() as s:
      return s.query(bqOtherFile).count()

  @property
  def other_repo_count(self) -> int:
    with self.Session() as s:
      return s.query(bqOtherFile.repo_name, bqOtherFile.ref).distinct().count()

  ##### Repository table properties
  @property
  def loadRepos(self) -> typing.Set[typing.Tuple[str, str]]:
    with self.Session() as s:
      return set((e.repo_name, e.ref) for e in s.query(bqRepo))

  @property
  def repo_count(self) -> int:
    """
    Get number of repos in bqRepo table.
    """
    with self.Session() as s:
      return s.query(bqRepo).count()

  ##### Data
  @property
  def data(self) -> bqData:
    """
    Get bqData entry from table.
    """
    with self.Session() as s:
      return s.query(bqData).first()

def chunkify_db(bq_db: bqDatabase, chunks: int, prefix: str) -> None:
  out_dbs = [bqDatabase(url = "sqlite:///{}_{}.db".format(prefix, idx)) for idx in range(chunks)]

  total_files = bq_db.mainfile_count
  chunk_size = total_files // chunks

  idx = 0
  for db_idx, db in enumerate(out_dbs):
    l.logger().info("Writing db_{}...".format(db_idx))
    batch = bq_db.main_files_batch(limit = chunk_size + (chunks if db_idx == chunks - 1 else 0), offset = idx)
    with db.Session() as s:
      bar = progressbar.ProgressBar(max_value = len(batch))
      l.logger().info(len(batch))
      for f in bar(batch):
        s.add(bqMainFile(**bqMainFile.FromArgs(f.ToJSONDict())))
        idx += 1
      l.logger().info("commit")
      s.commit()
  return

def file_size_distribution(db: bqDatabase) -> None:
  """
  Plot the distribution of size of each entry.
  """
  m = monitors.FrequencyMonitor(".", "bq_size_distrib")
  with db.Session() as s:
    for x in tqdm.tqdm(s.query(bqMainFile.size).all(), total = db.mainfile_count):
      try:
        y = int(str(x[0]))
        m.register(y)
      except Exception:
        pass
  m.plot()
  print(m.getData())
  return

def reduce_database_by_size(db: bqDatabase, out_db: bqDatabase) -> None:
  """
  Reduce BQ database by files that are too massive anyway.
  """
  with db.Session() as s:
    data = []
    for x in s.query(bqMainFile).all():
      try:
        if int(str(x.size)) < 10**6:
          data.append(x)
      except ValueError:
        pass
  l.logger().info("BQ Database reduced from {} to {} files".format(db.mainfile_count, len(data)))
  with out_db.Session(commit = True) as s:
    for dp in data:
      s.add(bqMainFile(
          id = dp.id,
          repo_name = "",
          ref = "",
          path = "",
          size = dp.size,
          content = dp.content,
          date_added = datetime.datetime.utcnow(),
        )
      )
    s.commit()
  return

def initMain(*args, **kwargs):
  """
  Setup module's operations.
  """
  l.initLogger(name = "bigQuery_database")
  # file_size_distribution(bqDatabase(url = "sqlite:///{}".format("/private/home/foivos/clgen_c_github.db", must_exist = True)))
  reduce_database_by_size(
    bqDatabase(url = "sqlite:///{}".format("/private/home/foivos/clgen_c_github.db", must_exist = True)),
    bqDatabase(url = "sqlite:///{}".format("/private/home/foivos/reduced1M_clgen_c_github.db", must_exist = False)),
  )
  return
  if FLAGS.chunkify or FLAGS.chunkify < 2:
    if not FLAGS.bq_database:
      raise ValueError("You must set a path for bq_database")
    bq_db_path = pathlib.Path(FLAGS.bq_database).resolve()
    if not bq_db_path.exists():
      raise FileNotFoundError(bq_db_path)
    bq_db = bqDatabase(url = "sqlite:///{}".format(str(bq_db_path)), must_exist = True)
    chunkify_db(bq_db, FLAGS.chunkify, prefix = "{}/{}".format(bq_db_path.parent, bq_db_path.stem))
  else:
    l.logger().warn("Chunkify has not been set or has been set to less than 2. Nothing to do, exiting...")
  return

if __name__ == "__main__":
  app.run(initMain)
  sys.exit(0)
