"""A module for databases of CLgen samples."""
import contextlib
import datetime
import typing
import sqlite3

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
  # Actual length neglecting pads
  kernel_length    : int = sql.Column(sql.Integer, nullable = False)
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
  # Sample's vector of features.
  output_features  : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Whether the generated sample is of good quality or not.
  sample_quality   : int = sql.Column(sql.Integer,  nullable = False)
  # Date
  date_added       : datetime.datetime = sql.Column(sql.DateTime, nullable = False)

  @classmethod
  def FromArgs(cls,
               atomizer,
               id               : int,
               input_feed       : str,
               input_features   : str,
               masked_input_ids : np.array,
               hole_instances   : typing.TypeVar("sequence_masking.MaskedLMInstance"),
               sample           : np.array,
               output_features  : str,
               sample_quality   : bool,
               ) -> typing.TypeVar("ActiveFeed"):
    str_masked_input_ids = atomizer.DeatomizeIndices(masked_input_ids)
    str_sample = atomizer.DeatomizeIndices(sample)
    ActiveFeed(
      id               = id,
      sha256           = crypto.sha256_str(str_masked_input_ids + str_sample),
      input_feed       = input_feed,
      encoded_feed     = ','.join(atomizer.AtomizeString(input_feed)),
      input_features   = input_features,
      masked_input_ids = ','.join(masked_input_ids),
      hole_lengths     = ','.join(lm.hole_length for l in hole_instances),
      hole_start_ids   = ','.join(lm.pos_index for l in hole_instances),
      sample           = atomizer.DeatomizeIndices(sample),
      output_features  = output_features,
      sample_quality   = sample_quality,
      date_added       = datetime.datetime.strptime(proto.date_added, "%m/%d/%Y, %H:%M:%S"),
    )

class ActiveFeedDatabase(sqlutil.Database):
  """A database of CLgen samples."""

  def __init__(self, url: str, must_exist: bool = False):
    super(ActiveFeedDatabase, self).__init__(url, Base, must_exist = must_exist)

  @property
  def count(self):
    """Number of samples in DB."""
    with self.Session() as s:
      count = s.query(ActiveFeed).count()
    return count
