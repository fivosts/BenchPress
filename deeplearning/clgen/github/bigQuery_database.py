"""A module for databases of CLgen samples."""
import contextlib
import datetime
import typing
import sqlite3
from google.cloud import bigquery

import sqlalchemy as sql
from sqlalchemy.ext import declarative
from labm8.py import sqlutil

Base = declarative.declarative_base()

class bqData(Base):
  __tablename__ = "bq_data"
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
  id             : int = sql.Column(sql.Integer,    primary_key = True)
  sha256         : str = sql.Column(sql.String(64), nullable = False, index = True)
  repo_name      : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  ref            : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  path           : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  size           : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  content        : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  date_added     : datetime.datetime = sql.Column(sql.DateTime, nullable=False)

  @classmethod
  def FromArgs(cls,
               id: int,
               row: bigquery.Row
               ) -> typing.Dict[str, typing.Any]:

    return {
      "id"             : id,
      "sha256"         : row['id'],
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
      "sha256"         : self.sha256,
      "repo_name"      : self.repo_name,
      "ref"            : self.ref,
      "path"           : self.path,
      "size"           : self.size,
      "content"        : self.content,
      "date_added"     : str(self.date_added.strftime("%m/%d/%Y, %H:%M:%S")),
    }

class bqMainFile(Base, bqFile):
  """Abstract representation of main queried files."""
  __tablename__  = "bq_main_contentfiles"

class bqOtherFile(Base, bqFile):
  """Abstract representation of other-to-main-language queried files."""
  __tablename__  = "bq_other_contentfiles"

class bqHeaderFile(Base, bqFile):
  """Abstract representation of header file includes."""
  __tablename__  = "bq_header_contentfiles"

class bqRepo(Base):
  """
    A database entry representing a CLgen validation trace.
  """
  __tablename__  = "bq_repofiles"
  id             : int = sql.Column(sql.Integer,    primary_key = True)
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

  @property
  def count(self) -> typing.Tuple[int, int]:
    """
    Get number of repositories in bqRepo
    and number of all contentfiles (inc. main, other & header)
    """
    return (self.repo_count, self.file_count)

  @property
  def file_count(self) -> int:
    """
    Get total number of contentfiles in DB.
    """
    with self.Session() as s:
      return (s.query(bqMainFile).count() +
              s.query(bqOtherFile).count() +
              s.query(bqHeaderFile).count()
             )

  @property
  def repo_count(self) -> int:
    """
    Get number of repos in bqRepo table.
    """
    with self.Session() as s:
      return s.query(bqRepo).count()

  @property
  def data(self) -> bqData:
    """
    Get bqData entry from table.
    """
    with self.Session() as s:
      return s.query(bqData).first()

  @property
  def repo_entries(self) -> typing.Set[str]:
    """
    Get all repository/ref entries in bqRepo table in string format.
    Returns a set of joint strings.
    """
    with self.Session() as s:
      repos = s.query(bqRepo)
      return set("{}, {}".format(e.repo_name, e.ref) for e in s.query(bqRepo))

  @property
  def main_repo_entries(self) -> typing.Set[str]:
    """
    Get distinct repository/ref list from bqMainFile table.
    """
    with self.Session() as s:
      q = s.query(bqMainFile).with_entities(bqMainFile.repo_name, bqMainFile.ref)
      return set("{}, {}".format(e.repo_name, e.ref) for e in q.yield_per(10000).enable_eagerloads(False))

  @property
  def other_repo_entries(self) -> typing.Set[str]:
    """
    Get distinct repository/ref list from bqOtherFile table.
    """
    with self.Session() as s:
      q = s.query(bqOtherFile).with_entities(bqOtherFile.repo_name, bqOtherFile.ref)
      return set("{}, {}".format(e.repo_name, e.ref) for e in q.yield_per(10000).enable_eagerloads(False))

  @property
  def header_repo_entries(self) -> typing.Set[str]:
    """
    Get distinct repository/ref list from bqHeaderFile table.
    """
    with self.Session() as s:
      q = s.query(bqHeaderFile).with_entities(bqHeaderFile.repo_name, bqHeaderFile.ref)
      return set("{}, {}".format(e.repo_name, e.ref) for e in q.yield_per(1000000).enable_eagerloads(False))

  @property
  def main_sha(self) -> typing.Set[str]:
    """
    Returns set of all distinct sha256 entries from main files.
    """
    with self.Session() as s:
      return set(s.query(bqMainFile.sha256).all())

  @property
  def other_sha(self) -> typing.Set[str]:
    """
    Returns set of all distinct sha256 entries from main files.
    """
    with self.Session() as s:
      return set(s.query(bqOtherFile.sha256).all())

  @property
  def header_sha(self) -> typing.Set[str]:
    """
    Returns set of all distinct sha256 entries from main files.
    """
    with self.Session() as s:
      return set(s.query(bqHeaderFile.sha256).all())
  
