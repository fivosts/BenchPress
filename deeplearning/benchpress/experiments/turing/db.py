# coding=utf-8
# Copyright 2023 Foivos Tsimpourlas.
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
"""
Evaluation script for kernel execution using cldrive or similar drivers.
"""
import datetime
import sqlite3
import typing

import sqlalchemy as sql
from sqlalchemy.ext import declarative

from deeplearning.benchpress.util import sqlutil
from deeplearning.benchpress.util import crypto
from deeplearning.benchpress.util import logging as l

from absl import flags

Base = declarative.declarative_base()

FLAGS = flags.FLAGS

class Data(Base):
  __tablename__ = "sampling_results"
  """
  DB Table for concentrated validation results.
  """
  key     : str = sql.Column(sql.String(1024), primary_key=True)
  results : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)

class QuizResult(Base, sqlutil.ProtoBackedMixin):
  """
  A database row representing a single quiz result.
  """
  __tablename__ = "quiz_result"
  # entry id
  id         : int = sql.Column(sql.Integer,    primary_key = True)
  # dataset name
  dataset    : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # code
  code       : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Human or Robot
  label      : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Prediction from user.
  prediction : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # User ID that made prediction.
  used_id    : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Ip of the user.
  user_ip    : str = sql.Column(sql.Integer,   nullable = False)
  # User was software engineer ?
  engineer : bool = sql.Column(sql.Boolean, unique = False, nullable = False)
  # Date the quiz was performed.
  date_added : datetime.datetime = sql.Column(sql.DateTime, nullable=False)

  @classmethod
  def FromArgs(cls,
               dataset    : int,
               code       : int,
               label      : str,
               prediction : typing.Dict[str, float],
               user_id    : str,
               user_ip    : typing.List[int],
               engineer   : bool,
               date_added : datetime.datetime,
               ) -> "QuizResult":
    return QuizResult(**{
      "dataset"    : dataset,
      "code"       : code,
      "label"      : label,
      "prediction" : prediction,
      "used_id"    : user_id,
      "user_ip"    : user_ip,
      "engineer"   : engineer,
      "date_added" : date_added,
    })

class UserSession(Base, sqlutil.ProtoBackedMixin):
  """
  A database with statistics, indexed by the unique User ID.
  """
  __tablename__ = "user_stats"
  # entry id
  id               : int = sql.Column(sql.Integer,    primary_key = True)
  # unique hash of cldrive execution.
  user_id          : str = sql.Column(sql.String(64), nullable = False, index = True)
  # Ips of one user.
  user_ip          : sql.Column(sql.ext.mutable.MutableDict.as_mutable(sql.dialects.postgresql.JSONB), nullable = False)
  # Software engineer or not ?
  engineer         : bool = sql.Column(sql.Boolean, unique = False, nullable = False)
  # Save the schedule for that user
  schedule         : sql.Column(sql.ext.mutable.MutableDict.as_mutable(sql.dialects.postgresql.JSONB), nullable = False)
  # Frequency distribution of encountered datasets
  dataset_distr    : sql.Column(sql.ext.mutable.MutableDict.as_mutable(sql.dialects.postgresql.JSONB), nullable = False)
  # Frequency distribution of oracle labels
  label_distr      : sql.Column(sql.ext.mutable.MutableDict.as_mutable(sql.dialects.postgresql.JSONB), nullable = False)
  # Predicted label distribution per dataset
  prediction_distr : sql.Column(sql.ext.mutable.MutableDict.as_mutable(sql.dialects.postgresql.JSONB), nullable = False)
  # Total predictions made
  num_predictions  : int = sql.Column(sql.Integer,   nullable = False)
  # Accumulated session for this user.
  session          : sql.Column(sql.ext.mutable.MutableDict.as_mutable(sql.dialects.postgresql.JSONB), nullable = False)
  # Date the quiz was performed.
  date_added       : datetime.datetime = sql.Column(sql.DateTime, nullable=False)

  @classmethod
  def FromArgs(cls,
               user_id          : str,
               user_ip          : typing.List[str],
               engineer         : bool,
               schedule         : typing.List[str],
               dataset_distr    : typing.Dict[str, int] = {},
               label_distr      : typing.Dict[str, int] = {},
               prediction_distr : typing.Dict[str, int] = {},
               num_predictions  : int = 0,
               session          : typing.List[typing.Dict[str, typing.Any]] = [],
               ) -> 'UserSession':
    return UserSession(**{
      "user_id"          : user_id,
      "user_ip"          : user_ip,
      "engineer"         : engineer,
      "schedule"         : schedule,
      "dataset_distr"    : dataset_distr,
      "label_distr"      : label_distr,
      "prediction_distr" : prediction_distr,
      "session"          : session,
      "num_predictions"  : num_predictions,
      "date_added"       : datetime.datetime.utcnow(),
    })

class TuringSession(Base, sqlutil.ProtoBackedMixin):
  """
  A database with high level statistics of all sessions.
  """
  __tablename__ = "turing_session"
  # entry id
  id               : int = sql.Column(sql.Integer,    primary_key = True)
  # Total number of participants by unique ids.
  num_user_ids     : int = sql.Column(sql.Integer, nullable = False)
  # A list of all user IDs.
  user_ids         : sql.Column(sql.ext.mutable.MutableDict.as_mutable(sql.dialects.postgresql.JSONB), nullable = False)
  # Ips of one user.
  num_user_ips     : int = sql.Column(sql.Integer, nullable = False)
  # A list of all user IPs.
  user_ips         : sql.Column(sql.ext.mutable.MutableDict.as_mutable(sql.dialects.postgresql.JSONB), nullable = False)
  # Engineers distribution
  engineer_distr   : sql.Column(sql.ext.mutable.MutableDict.as_mutable(sql.dialects.postgresql.JSONB), nullable = False)
  # Total predictions made per engineer and non engineer
  num_predictions  : sql.Column(sql.ext.mutable.MutableDict.as_mutable(sql.dialects.postgresql.JSONB), nullable = False)
  # Predictions distribution per engineer and non engineer per dataset with accuracies.
  prediction_distr : sql.Column(sql.ext.mutable.MutableDict.as_mutable(sql.dialects.postgresql.JSONB), nullable = False)
  # Date of assigned session.
  date_added       : datetime.datetime = sql.Column(sql.DateTime, nullable=False)

  @classmethod
  def FromArgs(cls,
               num_user_ids     : int = 0,
               user_ids         : typing.List[str] = [],
               num_user_ips     : int = 0,
               user_ips         : typing.List[str] = [],
               engineer_distr   : typing.Dict[int] = [],
               num_predictions  : int = 0,
               prediction_distr : typing.Dict[str, typing.Dict[str, typing.Any]] = {},
               ) -> "TuringSession":
    return TuringSession(**{
      "num_user_ids"     : num_user_ids,
      "user_ids"         : user_ids,
      "num_user_ips"     : num_user_ips,
      "user_ips"         : user_ips,
      "engineer_distr"   : engineer_distr,
      "num_predictions"  : num_predictions,
      "prediction_distr" : prediction_distr,
      "date_added"       : datetime.datetime.utcnow(),
    })

class TuringDB(sqlutil.Database):
  """A database of CLDrive Execution samples."""

  @property
  def count_users(self):
    """Number of cldrive traces in DB."""
    with self.Session() as s:
      count = s.query(UserSession).count()
    return count

  @property
  def count_quiz(self):
    """Number of cldrive traces in DB."""
    with self.Session() as s:
      count = s.query(QuizResult).count()
    return count

  def __init__(self, url: str, must_exist: bool = False):
    super(TuringDB, self).__init__(url, Base, must_exist = must_exist)
    self._status_cache = None

  def init_session(self) -> None:
    """
    TuringSession table must have only one entry.
    If no entries exist, initialize one.
    """
    with self.Session(commit = True) as s:
      exists = s.query(TuringSession).scalar() is not None
      if not exists:
        s.add(TuringSession.FromArgs())

  def init_user(self, user_id: str, user_ip: str, engineer: bool, schedule: typing.List[str]) -> None:
    """
    Initialize user by initial details, if doesn't exist already.
    """
    with self.Session(commit = True) as s:
      user = s.query(UserSession).filter_by(user_id = user_id).first()
      if not user:
        s.add(UserSession.FromArgs(
            user_id  = user_id,
            user_ip  = [user_ip],
            engineer = engineer,
          )
        )
    return

  def add_quiz(self,
               dataset    : int,
               code       : int,
               label      : str,
               prediction : typing.Dict[str, float],
               user_id    : str,
               user_ip    : typing.List[int],
               engineer   : bool,
               date_added : datetime.datetime,
               ) -> None:
    """
    Add new quiz instance to DB
    """
    with self.Session() as s:
      s.add(QuizResult.FromArgs(
          dataset = dataset,
          code       = code,
          label      = label,
          prediction = prediction,
          user_id    = user_id,
          user_ip    = user_ip,
          engineer   = engineer,
          date_added = date_added,
        )
      )
    return
