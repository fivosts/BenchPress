"""A module for databases of active learning query by committee samples."""
import datetime
import typing
import progressbar
import sqlite3

import sqlalchemy as sql
from sqlalchemy.ext import declarative

from deeplearning.clgen.util import crypto
from deeplearning.clgen.util import sqlutil
from deeplearning.clgen.util import logging as l

Base = declarative.declarative_base()

class QBCResults(Base):
  __tablename__ = "qbc_results"
  """
    DB Table for concentrated validation results.
  """
  key     : str = sql.Column(sql.String(1024), primary_key=True)
  results : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)

class CommitteeConfig(Base, sqlutil.ProtoBackedMixin):
  """
  A table where each row presents the configuration of a committee member.
  """
  __tablename__ = "committee_members"
  # entry id
  id            : int = sql.Column(sql.Integer, primary_key = True)
  # Assigned member ID
  member_id     : int = sql.Column(sql.Integer, nullable = False, index = True)
  # Name of member
  member_name   : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Type of AI architecture (supervised, unsupervised etc.)
  type          : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # configuration specs
  configuration : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Date
  date_added    : datetime.datetime = sql.Column(sql.DateTime, nullable=False)

  @classmethod
  def FromArgs(cls,
               id            : int,
               member_id     : int,
               member_name   : str,
               type          : str,
               configuration : str
               ) -> 'CommitteeConfig':
    return CommitteeConfig(
      id            = id,
      member_id     = member_id,
      member_name   = member_name,
      type          = type,
      configuration = configuration,
      date_added    = datetime.datetime.utcnow(),
    )

class CommitteeSample(Base, sqlutil.ProtoBackedMixin):
  """A database row representing a CLgen sample.

  This is the clgen.Sample protocol buffer in SQL format.
  """
  __tablename__    = "qbc_samples"
  # entry id
  id                     : int = sql.Column(sql.Integer,    primary_key = True)
  # unique hash of sample text
  sha256                 : str = sql.Column(sql.String(64), nullable = False, index = True)
  # model's train step that generated the sample
  train_step             : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Original input where the feed came from
  static_features        : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Starting feed of model
  runtime_features       : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # String-format generated text
  input_features         : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Predictions of committee
  member_predictions     : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Amount of entropy
  entropy                : float = sql.Column(sql.Float, nullable=False)
  # Date
  date_added             : datetime.datetime = sql.Column(sql.DateTime, nullable=False)

  @classmethod
  def FromArgs(cls,
               id                 : int,
               train_step         : typing.Dict[str, int],
               static_features    : typing.Dict[str, float],
               runtime_features   : typing.Dict[str, float],
               input_features     : typing.Dict[str, float],
               member_predictions : typing.Dict[str, str],
               entropy            : float,
               ) -> 'CommitteeSample':
    str_train_step         = '\n'.join(["{}:{}".format(k, v) for k, v in train_step.items()])
    str_static_features    = '\n'.join(["{}:{}".format(k, v) for k, v in static_features.items()])
    str_runtime_features   = '\n'.join(["{}:{}".format(k, v) for k, v in runtime_features.items()])
    str_input_features     = '\n'.join(["{}:{}".format(k, v) for k, v in input_features.items()])
    str_member_predictions = '\n'.join(["{}:{}".format(k, v) for k, v in member_predictions.items()])
    sha256 = crypto.sha256_str(
      str_train_step
      + str_static_features
      + str_runtime_features
      + str_input_features
      + str_member_predictions
    )
    return CommitteeSample(
      id                 = id,
      sha256             = sha256,
      train_step         = str_train_step,
      static_features    = str_static_features,
      runtime_features   = str_runtime_features,
      input_features     = str_input_features,
      member_predictions = str_member_predictions,
      entropy            = entropy,
      date_added         = datetime.datetime.utcnow(),
    )

class CommitteeSamples(sqlutil.Database):
  """A database of Query-by-Committee samples."""

  def __init__(self, url: str, must_exist: bool = False):
    super(CommitteeSamples, self).__init__(url, Base, must_exist = must_exist)

  @property
  def member_count(self):
    """Number of committee members in DB."""
    with self.Session() as s:
      count = s.query(CommitteeConfig).count()
    return count

  @property
  def sample_count(self):
    """Number of samples in DB."""
    with self.Session() as s:
      count = s.query(CommitteeSample).count()
    return count

  @property
  def get_data(self):
    """Return all database in list format"""
    with self.Session() as s:
      return s.query(CommitteeSample).all()

  def add_member(self, member_id: int, member_name: str, type: str, configuration: str) -> None:
    """
    Add committee member if not exists.
    """
    with self.Session(commit = True) as s:
      exists = s.query(CommitteeConfig).filter_by(member_id = member_id).first()
      if not exists:
        s.add(CommitteeConfig.FromArgs(self.member_count, member_id, member_name, type, configuration))
        s.commit()
    return

  def add_samples(self, samples: typing.Dict[str, typing.Any]) -> None:
    """
    If not exists, add sample to Samples table.
    """
    hash_cache = set()
    offset_idx = 0
    with self.Session(commit = True) as s:
      for sample in samples:
        sample_entry = CommitteeSample.FromArgs(
          id                 = self.sample_count + offset_idx,
          train_step         = sample['train_step'],
          static_features    = sample['static_features'],
          runtime_features   = sample['runtime_features'],
          input_features     = sample['input_features'],
          member_predictions = sample['member_predictions'],
          entropy            = sample['entropy'],
        )
        exists = s.query(CommitteeSample).filter_by(sha256 = sample_entry.sha256).first()
        if not exists and sample.sha256 not in hash_cache:
          s.add(sample_entry)
          hash_cache.add(sample.sha256)
          offset_idx += 1
      s.commit()
    return
