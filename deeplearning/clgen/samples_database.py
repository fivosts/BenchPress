"""A module for databases of CLgen samples."""
import contextlib
import datetime
import typing
import sqlite3

import sqlalchemy as sql
from sqlalchemy.ext import declarative
from absl import flags

from deeplearning.clgen import sample_observers
from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen.util import crypto
from labm8.py import sqlutil

FLAGS = flags.FLAGS

Base = declarative.declarative_base()

class Sample(Base, sqlutil.ProtoBackedMixin):
  """A database row representing a CLgen sample.

  This is the clgen.Sample protocol buffer in SQL format.
  """
  __tablename__    = "samples"
  id                   : int = sql.Column(sql.Integer,    primary_key = True)
  sha256               : str = sql.Column(sql.String(64), nullable = False, index = True)
  train_step           : int = sql.Column(sql.Integer,    nullable = False)
  encoded_text         : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  sample_feed          : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  text                 : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  num_tokens           : int = sql.Column(sql.Integer,    nullable = False)
  categorical_sampling : str = sql.Column(sql.String(8),  nullable = False)
  sample_time_ms       : int = sql.Column(sql.Integer,    nullable = False)
  date_added           : datetime.datetime = sql.Column(sql.DateTime, nullable=False)

  @classmethod
  def FromProto(cls, id: int, proto: model_pb2.Sample) -> typing.Dict[str, typing.Any]:
    return {
      "id"                   : id,
      "sha256"               : crypto.sha256_str(proto.text),
      "train_step"           : proto.train_step,
      "encoded_text"         : proto.encoded_text,
      "sample_feed"          : proto.sample_feed,
      "text"                 : proto.text,
      "num_tokens"           : proto.num_tokens,
      "categorical_sampling" : proto.categorical_sampling,
      "sample_time_ms"       : proto.sample_time_ms,
      "date_added"           : datetime.datetime.strptime(proto.date_added, "%m/%d/%Y, %H:%M:%S"),
    }

class SamplesDatabase(sqlutil.Database):
  """A database of CLgen samples."""

  def __init__(self, url: str, must_exist: bool = False):
    super(SamplesDatabase, self).__init__(url, Base, must_exist = must_exist)

  @property
  def count(self):
    with self.Session() as s:
      count = s.query(Sample).count()
    return count