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

class bqFile(Base, sqlutil.ProtoBackedMixin):
  """
    A database entry representing a CLgen validation trace.
  """
  __tablename__  = "bq_contentfiles"
  id             : int = sql.Column(sql.Integer,    primary_key = True)
  sha256         : str = sql.Column(sql.String(64), nullable = False, index = True)
  repo_name      : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  ref            : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  path           : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  mode           : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = True)
  symlink_target : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = True)
  size           : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = True)
  content        : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = True)
  binary         : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = True)
  copies         : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
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
      "mode"           : row['mode'],
      "symlink_target" : row['symlink_target'],
      "size"           : row['size'],
      "content"        : row['content'],
      "binary"         : row['binary'],
      "copies"         : row['copies'],
      "date_added"     : datetime.datetime.utcnow(),
    }

  @staticmethod
  def bqSchema() -> typing.List[bigquery.SchemaField]:
    return [
      bigquery.SchemaField("id",             "STRING",  mode = "REQUIRED"),
      bigquery.SchemaField("repo_name",      "STRING",  mode = "REQUIRED"),
      bigquery.SchemaField("ref",            "STRING",  mode = "REQUIRED"),
      bigquery.SchemaField("path",           "STRING",  mode = "REQUIRED"),
      bigquery.SchemaField("mode",           "STRING",  mode = "REQUIRED"),
      bigquery.SchemaField("symlink_target", "STRING",  mode = "REQUIRED"),
      bigquery.SchemaField("size",           "INTEGER", mode = "REQUIRED"),
      bigquery.SchemaField("content",        "STRING",  mode = "REQUIRED"),
      bigquery.SchemaField("binary",         "BOOLEAN", mode = "REQUIRED"),
      bigquery.SchemaField("copies",         "INTEGER", mode = "REQUIRED"),
    ]

  def ToJSONDict(self) -> typing.Dict[str, typing.Any]:
    return {
      "id"             : self.id,
      "sha256"         : self.sha256,
      "repo_name"      : self.repo_name,
      "ref"            : self.ref,
      "path"           : self.path,
      "mode"           : self.mode,
      "symlink_target" : self.symlink_target,
      "size"           : self.size,
      "content"        : self.content,
      "binary"         : self.binary,
      "copies"         : self.copies,
      "date_added"     : str(self.date_added.strftime("%m/%d/%Y, %H:%M:%S")),
    }

class bqRepo(Base, sqlutil.ProtoBackedMixin):
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
  def count(self):
    return (self.repo_count, self.file_count)

  @property
  def file_count(self):
    with self.Session() as s:
      file_count = s.query(bqFile).count()
    return file_count

  @property
  def repo_count(self):
    with self.Session() as s:
      repo_count = s.query(bqRepo).count()
    return repo_count