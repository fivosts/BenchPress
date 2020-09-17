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
  # entry id
  id                     : int = sql.Column(sql.Integer,    primary_key = True)
  # unique hash of sample text
  sha256                 : str = sql.Column(sql.String(64), nullable = False, index = True)
  # model's train step that generated the sample
  train_step             : int = sql.Column(sql.Integer,    nullable = False)
  # encoded sample text
  encoded_text           : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Starting feed of model
  sample_feed            : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # String-format generated text
  text                   : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Array of the actual generated tokens
  sample_indices         : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Encoded generated tokens
  encoded_sample_indices : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Length of total sequence in number of tokens
  num_tokens             : int = sql.Column(sql.Integer,    nullable = False)
  # Whether the generated sample compiles or not.
  compile_status         : str = sql.Column(sql.String(8),  nullable = False)
  # If Bernoulli distribution was used during samplinng
  categorical_sampling   : str = sql.Column(sql.String(8),  nullable = False)
  # Time
  sample_time_ms         : int = sql.Column(sql.Integer,    nullable = False)
  # Date
  date_added             : datetime.datetime = sql.Column(sql.DateTime, nullable=False)

  @classmethod
  def FromProto(cls, id: int, proto: model_pb2.Sample) -> typing.Dict[str, typing.Any]:
    return {
      "id"                     : id,
      "sha256"                 : crypto.sha256_str(proto.text),
      "train_step"             : proto.train_step,
      "encoded_text"           : proto.encoded_text,
      "sample_feed"            : proto.sample_feed,
      "text"                   : proto.text,
      "sample_indices"         : proto.sample_indices,
      "encoded_sample_indices" : proto.encoded_sample_indices,
      "num_tokens"             : proto.num_tokens,
      "compile_status"         : proto.compile_status,
      "categorical_sampling"   : proto.categorical_sampling,
      "sample_time_ms"         : proto.sample_time_ms,
      "date_added"             : datetime.datetime.strptime(proto.date_added, "%m/%d/%Y, %H:%M:%S"),
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