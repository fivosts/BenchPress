"""A module for databases of CLgen samples."""
import contextlib
import datetime
import typing
import sqlite3

import sqlalchemy as sql
from sqlalchemy.ext import declarative
from absl import flags

from deeplearning.clgen.util import crypto
from labm8.py import sqlutil

FLAGS = flags.FLAGS

Base = declarative.declarative_base()

class ValResults(Base):
  __tablename__ = "bq_data"
  """
    DB Table for concentrated validation results.
  """
  key   : str = sql.Column(sql.String(1024), primary_key=True)
  value : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)

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
  mode           : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  symlink_target : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  size           : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  content        : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  binary         : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  copies         : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  date_added     : datetime.datetime = sql.Column(sql.DateTime, nullable=False)

  @classmethod
  def FromArgs(cls, 
               id: int,
               row
               ) -> typing.Dict[str, typing.Any]:

    return {
      "id"             : id,
      "sha256"         : row['id'],
      "repo_name"      : row['repo_name'],
      "ref"            : row['ref'],
      "path"           : row['path'],
      "mode"           : row['mode']           if row['mode']           else "None",
      "symlink_target" : row['symlink_target'] if row['symlink_target'] else "None",
      "size"           : row['size']           if row['size']           else "None",
      "content"        : row['content']        if row['content']        else "None",
      "binary"         : row['binary']         if row['binary']         else "None",
      "copies"         : row['copies']         if row['copies']         else "None",
      "date_added"     : datetime.datetime.utcnow(),
    }

class bqDatabase(sqlutil.Database):
  """A database of BigQuery contentfiles."""

  def __init__(self, url: str, must_exist: bool = False):
    super(bqDatabase, self).__init__(url, Base, must_exist = must_exist)

  @property
  def count(self):
    with self.Session() as s:
      count = s.query(bqFile).count()
    return count