"""A module for databases of CLgen samples."""
import contextlib
import datetime
import typing
import sqlite3
from google.cloud import bigquery

import sqlalchemy as sql
from sqlalchemy.ext import declarative
from deeplearning.clgen.util import sqlutil

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
      files = s.query(bqMainFile).limit(limit).offset(offset).filter(bqMainFile.id not in exclude_id).all()
      if exclude_id:
        for vid in s.query(bqMainFile.id).limit(limit).offset(offset).filter(bqMainFile.id in exclude_id).all():
          exclude_id.remove(vid)
      return files

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
