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
import json

import sqlalchemy as sql
from sqlalchemy.ext import declarative
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.mutable import MutableDict

from deeplearning.benchpress.util import sqlutil
from deeplearning.benchpress.util import crypto
from deeplearning.benchpress.util import logging as l

from absl import flags

Base = declarative.declarative_base()

FLAGS = flags.FLAGS

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
  user_id    : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
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
               ) -> "QuizResult":
    return QuizResult(**{
      "dataset"    : dataset,
      "code"       : code,
      "label"      : label,
      "prediction" : prediction,
      "user_id"    : user_id,
      "user_ip"    : user_ip,
      "engineer"   : engineer,
      "date_added" : datetime.datetime.utcnow(),
    })

class UserSession(Base, sqlutil.ProtoBackedMixin):
  """
  A database with statistics, indexed by the unique User ID.
  """
  __tablename__ = "user_session"
  # entry id
  id               : int = sql.Column(sql.Integer, primary_key = True)
  # unique hash of cldrive execution.
  user_id          : str = sql.Column(sql.String(64), nullable = False, index = True)
  # Ips of one user.
  user_ip          : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Software engineer or not ?
  engineer         : bool = sql.Column(sql.Boolean, unique = False, nullable = False)
  # Save the schedule for that user
  schedule         : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Frequency distribution of encountered datasets
  dataset_distr    : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Frequency distribution of oracle labels
  label_distr      : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Predicted labels distribution per dataset
  prediction_distr : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Total predictions made
  num_predictions  : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Accumulated session for this user.
  session          : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Date the quiz was performed.
  date_added       : datetime.datetime = sql.Column(sql.DateTime, nullable=False)

  @classmethod
  def FromArgs(cls,
               user_id          : str,
               engineer         : bool,
               schedule         : typing.List[str],
               user_ip          : typing.List[str] = [],
               dataset_distr    : typing.Dict[str, int] = {},
               label_distr      : typing.Dict[str, int] = {"human": 0, "robot": 0},
               prediction_distr : typing.Dict[str, typing.Dict[str, typing.Any]] = {},
               num_predictions  : typing.Dict[str, int] = {},
               session          : typing.List[typing.Dict[str, typing.Any]] = [],
               ) -> 'UserSession':
    l.logger().critical(prediction_distr)
    return UserSession(**{
      "user_id"          : user_id,
      "user_ip"          : json.dumps(user_ip, indent = 2),
      "engineer"         : engineer,
      "schedule"         : json.dumps(schedule, indent = 2),
      "dataset_distr"    : json.dumps(dataset_distr, indent = 2),
      "label_distr"      : json.dumps(label_distr, indent = 2),
      "prediction_distr" : json.dumps(prediction_distr, indent = 2),
      "session"          : json.dumps(session, indent = 2),
      "num_predictions"  : json.dumps(num_predictions, indent = 2),
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
  user_ids         : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Ips of one user.
  num_user_ips     : int = sql.Column(sql.Integer, nullable = False)
  # A list of all user IPs.
  user_ips         : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Engineers distribution
  engineer_distr   : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Total predictions made per engineer and non engineer
  num_predictions  : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Predictions distribution per engineer and non engineer per dataset with accuracies.
  prediction_distr : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Date of assigned session.
  date_added       : datetime.datetime = sql.Column(sql.DateTime, nullable=False)

  @classmethod
  def FromArgs(cls,
               num_user_ids     : int = 0,
               user_ids         : typing.List[str] = [],
               num_user_ips     : int = 0,
               user_ips         : typing.List[str] = [],
               engineer_distr   : typing.Dict[str, int] = {"engineer": 0, "non-engineer": 0},
               num_predictions  : typing.Dict[str, int] = {"engineer": {}, "non-engineer": {}},
               prediction_distr : typing.Dict[str, typing.Dict[str, typing.Any]] = {"engineer": {}, "non-engineer": {}},
               ) -> "TuringSession":
    return TuringSession(**{
      "num_user_ids"     : num_user_ids,
      "user_ids"         : json.dumps(user_ids, indent = 2),
      "num_user_ips"     : num_user_ips,
      "user_ips"         : json.dumps(user_ips, indent = 2),
      "engineer_distr"   : json.dumps(engineer_distr, indent = 2),
      "num_predictions"  : json.dumps(num_predictions, indent = 2),
      "prediction_distr" : json.dumps(prediction_distr, indent = 2),
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

  def get_user_accuracy(self, user_id: str, min_attempts: int) -> float:
    """
    Get accuracy of player, so long they've done at least min attempts.
    """
    with self.Session() as s:
      user = s.query(UserSession).filter_by(user_id = user_id).first()
      correct, total = 0, 0
      for dataset, data in json.loads(user.prediction_distr).items():
        for label, amount in data["predictions"].items():
          total += amount
          if label == data["label"]:
            correct += amount
    if total >= min_attempts:
      return correct / total, total
    else:
      return None, total

  def get_prediction_distr(self) -> typing.Dict[str, typing.Any]:
    """
    Return turing_session.prediction_distr
    """
    with self.Session() as s:
      return s.query(TuringSession.prediction_distr)

  def is_engineer(self, user_id: str) -> bool:
    """
    Return bool value of engineer status.
    """
    with self.Session() as s:
      user = s.query(UserSession).filter_by(user_id = user_id).first()
      if user:
        return user.engineer
      else:
        return None

  def get_schedule(self, user_id: str) -> typing.List[str]:
    """
    Return assigned schedule.
    """
    with self.Session() as s:
      user = s.query(UserSession).filter_by(user_id = user_id).first()
      if user:
        return json.loads(user.schedule)
      else:
        return None

  def init_session(self) -> None:
    """
    TuringSession table must have only one entry.
    If no entries exist, initialize one.
    """
    with self.Session(commit = True) as s:
      exists = s.query(TuringSession).scalar() is not None
      if not exists:
        s.add(TuringSession.FromArgs())

  def update_session(self, **kwargs) -> None:
    """
    Update session table with any new kwargs
    """
    with self.Session(commit = True) as s:
      session = s.query(TuringSession).first()
      if session is None:
        self.init_session()
      for key, value in kwargs.items():
        if key == "user_ids":
          usr_ids = json.loads(session.user_ids)
          if value not in usr_ids:
            session.user_ids = json.dumps(usr_ids + [value], indent = 2)
            session.num_user_ids += 1
        elif key == "user_ips":
          usr_ips = json.loads(session.user_ips)
          if value not in usr_ips:
            session.user_ips = json.dumps(usr_ips + [value], indent = 2)
            session.num_user_ips += 1
        elif key == "engineer_distr":
          eng_dist = json.loads(session.engineer_distr)
          if value:
            eng_dist["engineer"] += 1
          else:
            eng_dist["non-engineer"] += 1
          session.engineer_distr = json.dumps(eng_dist, indent = 2)
        elif key == "num_predictions":
          pred_dist = json.loads(session.num_predictions)
          engineer = "engineer" if kwargs.get("engineer") else "non-engineer"
          dname, freq = value
          if dname not in pred_dist[engineer]:
            pred_dist[engineer][dname] = freq
          else:
            pred_dist[engineer][dname] += freq
          session.num_predictions = json.dumps(pred_dist, indent = 2)
        elif key == "prediction_distr":
          cur_distr = json.loads(session.prediction_distr)
          for eng, attrs in value.items():
            engineer = "engineer" if eng else "non-engineer"
            for dname, attrs2 in attrs.items():
              if dname not in cur_distr[engineer]:
                cur_distr[engineer][dname] = {
                  "label": attrs2["label"],
                  "predictions": {
                    "human": 0,
                    "robot": 0,
                  }
                }
              cur_distr[engineer][dname]["predictions"][attrs2["predictions"]] += 1
          session.prediction_distr = json.dumps(cur_distr, indent = 2)
    return

  def update_user(self, user_id: str, **kwargs) -> None:
    """
    Add or update existing user.
    """
    with self.Session(commit = True) as s:
      user = s.query(UserSession).filter_by(user_id = user_id).first()
      if user is None:
        s.add(UserSession.FromArgs(
            user_id = user_id,
            **kwargs
          )
        )
        session = s.query(TuringSession).first()
        is_engineer = kwargs.get("engineer")
        cur_eng_dist = json.loads(session.engineer_distr)
        if is_engineer:
          cur_eng_dist["engineer"] += 1
        else:
          cur_eng_dist["non-engineer"] += 1
        session.engineer_dist = json.dumps(cur_eng_dist, indent = 2)
      else:
        for key, value in kwargs.items():
          if key == "user_ip":
            usr_ip = json.loads(user.user_ip)
            if value not in usr_ip:
              user.user_ip = json.dumps(usr_ip + [value], indent = 2)
          elif key == "engineer":
            l.logger().warn("Engineer has already been set to {}. I am not updating that.".format(user.engineer))
          elif key == "schedule":
            user.schedule = json.dumps(value, indent = 2)
          elif key == "dataset_distr":
            cur_distr = json.loads(user.dataset_distr)
            for k, v in value.items():
              if k not in cur_distr:
                cur_distr[k] = v
              else:
                cur_distr[k] += v
            user.dataset_distr = json.dumps(cur_distr, indent = 2)
          elif key == "label_distr":
            cur_distr = json.loads(user.label_distr)
            for k, v in value.items():
              cur_distr[k] += v
            user.label_distr = json.dumps(cur_distr, indent = 2)
          elif key == "prediction_distr":
            cur_distr = json.loads(user.prediction_distr)
            for dname, attrs in value.items():
              if dname not in cur_distr:
                cur_distr[dname] = {
                  "label": attrs["label"],
                  "predictions": {
                    "human": 0,
                    "robot": 0,
                  }
                }
              for k, v in attrs["predictions"].items():
                cur_distr[dname]["predictions"][k] += v
            user.prediction_distr = json.dumps(cur_distr, indent = 2)
          elif key == "num_predictions":
            cur_num_preds = json.loads(user.num_predictions)
            for k, v in value.items():
              if k not in cur_num_preds:
                cur_num_preds[k] = v
              else:
                cur_num_preds[k] += v
            user.num_predictions = json.dumps(cur_num_preds, indent = 2)
          elif key == "session":
            user.session = json.dumps(json.loads(user.session) + value, indent = 2)
    return

  def add_quiz(self,
               dataset    : int,
               code       : int,
               label      : str,
               prediction : typing.Dict[str, float],
               user_id    : str,
               user_ip    : typing.List[int],
               engineer   : bool,
               schedule   : typing.List[str],
               ) -> int:
    """
    Add new quiz instance to DB
    """
    with self.Session(commit = True) as s:
      s.add(QuizResult.FromArgs(
          dataset = dataset,
          code       = code,
          label      = label,
          prediction = prediction,
          user_id    = user_id,
          user_ip    = user_ip,
          engineer   = engineer,
        )
      )
    self.update_user(
      user_id = user_id,
      dataset_distr = {dataset: 1},
      label_distr = {label: 1, "human" if label == "robot" else "robot" : 0},
      engineer = engineer,
      schedule = schedule,
      prediction_distr = {
        dataset: {
          "label": label,
          "predictions": {prediction: 1, "human" if prediction == "robot" else "robot" : 0},
        }
      },
      num_predictions = {dataset: 1},
      session = [{
        "dataset"    : dataset,
        "code"       : code,
        "label"      : label,
        "prediction" : prediction,
      }]
    )
    self.update_session(
      engineer = engineer,
      num_predictions = [dataset, 1],
      prediction_distr = {
        engineer: {
          dataset: {
            "label": label,
            "predictions": prediction
          }
        }
      }
    )
    return 0
