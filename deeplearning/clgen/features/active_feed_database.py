"""A module for databases of CLgen samples."""
import contextlib
import datetime
import typing
import sqlite3
import numpy as np
import sqlalchemy as sql
from sqlalchemy.ext import declarative
from absl import flags

from deeplearning.clgen.samplers import sample_observers
from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen.util import crypto
from labm8.py import sqlutil

FLAGS = flags.FLAGS

Base = declarative.declarative_base()

class ActiveFeedHistory(Base):
  __tablename__ = "data"
  """
    DB Table for concentrated online/active sampling results.
  """
  key     : str = sql.Column(sql.String(1024), primary_key=True)
  results : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)

class ActiveInput(Base, sqlutil.ProtoBackedMixin):
  """
  A database for all original inputs used for active learning.
  """
  __tablename__    = "input_feeds"
  # entry id
  id             : int = sql.Column(sql.Integer,    primary_key = True)
  # unique hash of sample text
  sha256         : str = sql.Column(sql.String(64), nullable = False, index = True)
  # Text original input
  input_feed     : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Encoded original input
  encoded_feed   : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Feature vector of input_feed
  input_features : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Actual length of sample, excluding pads.
  num_tokens     : int = sql.Column(sql.Integer, nullable = False)
  # Date
  date_added     : datetime.datetime = sql.Column(sql.DateTime, nullable = False)

  @classmethod
  def FromArgs(cls,
               tokenizer,
               id             : int,
               input_feed     : np.array,
               input_features : typing.Dict[str, float],
               ) -> typing.TypeVar("ActiveInput"):
    """Construt ActiveFeed table entry from argumentns."""
    str_input_feed = tokenizer.tokensToString(input_feed, ignore_token = tokenizer.padToken)
    if tokenizer.padToken in input_feed:
      num_tokens = np.where(sample == tokenizer.padToken)[0][0]
    else:
      num_tokens = len(input_feed)

    return ActiveInput(
      id             = id,
      sha256         = crypto.sha256_str(str_masked_input_ids),
      input_feed     = str_input_feed,
      encoded_feed   = ','.join([str(x) for x in input_feed]),
      input_features = '\n'.join(["{}:{}".format(k, v) for k, v in input_features.items()]),
      num_tokens     = int(num_tokens),
      date_added     = datetime.datetime.utcnow(),
    )


class ActiveFeed(Base, sqlutil.ProtoBackedMixin):
  """A database row representing a CLgen sample.

  This is the clgen.Sample protocol buffer in SQL format.
  """
  __tablename__    = "active_feeds"
  # entry id
  id               : int = sql.Column(sql.Integer,    primary_key = True)
  # unique hash of sample text
  sha256           : str = sql.Column(sql.String(64), nullable = False, index = True)
  # Text original input
  input_feed       : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Encoded original input
  encoded_feed     : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Feature vector of input_feed
  input_features   : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Resulting encoded array with masks
  masked_input_ids : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Array of lengths of holes for given instance
  hole_lengths     : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Array of starting ids of hole instances in feed.
  hole_start_ids   : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Output sample
  sample           : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Actual length of sample, excluding pads.
  num_tokens       : int = sql.Column(sql.Integer, nullable = False)
  # Sample's vector of features.
  output_features  : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Whether the generated sample is of good quality or not.
  sample_quality   : float = sql.Column(sql.Float,  nullable = False)
  # Whether sample compiles or not
  compile_status   : bool = sql.Column(sql.Boolean, nullable = False)
  # Number of generation for sample
  generation_id    : int  = sql.Column(sql.Integer, nullable = False)
  # Date
  date_added       : datetime.datetime = sql.Column(sql.DateTime, nullable = False)

  @classmethod
  def FromArgs(cls,
               tokenizer,
               id               : int,
               input_feed       : np.array,
               input_features   : typing.Dict[str, float],
               masked_input_ids : np.array,
               hole_instances   : typing.TypeVar("sequence_masking.MaskedLMInstance"),
               sample           : np.array,
               output_features  : typing.Dict[str, float],
               sample_quality   : float,
               compile_status   : bool,
               generation_id    : int,
               ) -> typing.TypeVar("ActiveFeed"):
    """Construt ActiveFeed table entry from argumentns."""
    str_input_feed       = tokenizer.tokensToString(input_feed,       ignore_token = tokenizer.padToken)
    str_masked_input_ids = tokenizer.tokensToString(masked_input_ids, ignore_token = tokenizer.padToken)
    str_sample           = tokenizer.tokensToString(sample,           ignore_token = tokenizer.padToken)

    num_tokens = len(sample)
    if tokenizer.padToken in sample:
      num_tokens = np.where(sample == tokenizer.padToken)[0][0]

    return ActiveFeed(
      id               = id,
      sha256           = crypto.sha256_str(str_masked_input_ids + str_sample),
      input_feed       = str_input_feed,
      encoded_feed     = ','.join([str(x) for x in input_feed]),
      input_features   = '\n'.join(["{}:{}".format(k, v) for k, v in input_features.items()]),
      masked_input_ids = str_masked_input_ids,
      hole_lengths     = ','.join(str(lm.hole_length) for lm in hole_instances),
      hole_start_ids   = ','.join(str(lm.pos_index) for lm in hole_instances),
      sample           = str_sample,
      num_tokens       = int(num_tokens),
      output_features  = '\n'.join(["{}:{}".format(k, v) for k, v in output_features.items()]) if output_features else "None",
      sample_quality   = sample_quality,
      compile_status   = compile_status,
      generation_id    = generation_id,
      date_added       = datetime.datetime.utcnow(),
    )

class ActiveFeedDatabase(sqlutil.Database):
  """A database of CLgen samples."""

  def __init__(self, url: str, must_exist: bool = False):
    super(ActiveFeedDatabase, self).__init__(url, Base, must_exist = must_exist)

  @property
  def input_count(self):
    """Number of input feeds in DB."""
    with self.Session() as s:
      count = s.query(ActiveInput).count()
    return count

  @property
  def active_count(self):
    """Number of active samples in DB."""
    with self.Session() as s:
      count = s.query(ActiveFeed).count()
    return count
