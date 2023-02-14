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
import pathlib
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
from absl import app as absl_app

Base = declarative.declarative_base()

FLAGS = flags.FLAGS

flags.DEFINE_string(
  "out_results_db",
  None,
  "Set path to out results DB."
)

flags.DEFINE_string(
  "in_results_db",
  None,
  "Set comma-separated paths for input DBs to be merged."
)

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

  def get_quizzes(self) -> typing.List[QuizResult]:
    """
    Return a list of all quizzes.
    """
    with self.Session() as s:
      return s.query(QuizResult).all()

  def get_users(self) -> typing.List[UserSession]:
    """
    Return a list of all user sessions.
    """
    with self.Session() as s:
      return s.query(UserSession).all()

  def get_session(self) -> TuringSession:
    """
    Return DB's session.
    """
    with self.Session() as s:
      return s.query(TuringSession).first()

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
      return json.loads(s.query(TuringSession.prediction_distr).all()[0][0])

  def get_user_prediction_distr(self) -> typing.Dict[str, typing.List[typing.Dict[str, typing.Any]]]:
    """
    Group users to eng/non-eng, each category has a list of prediction_distr one per user.
    """
    with self.Session() as s:
      return {
        "engineer": [json.loads(x[0]) for x in s.query(UserSession.prediction_distr).filter_by(engineer = 1).all()],
        "non-engineer": [json.loads(x[0]) for x in s.query(UserSession.prediction_distr).filter_by(engineer = 0).all()],
      }

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

def merge_quiz(in_dbs: typing.List[TuringDB], out_db: TuringDB) -> None:
  data = []
  for db in in_dbs:
    data += db.get_quizzes()
  with out_db.Session(commit = True) as s:
    for dp in data:
      s.add(
        QuizResult(
          **{
            "dataset"    : dp.dataset,
            "code"       : dp.code,
            "label"      : dp.label,
            "prediction" : dp.prediction,
            "user_id"    : dp.user_id,
            "user_ip"    : dp.user_ip,
            "engineer"   : dp.engineer,
            "date_added" : dp.date_added,
          }
        )
      )
  return

def merge_user(in_dbs: typing.List[TuringDB], out_db: TuringDB) -> None:
  data = []
  for db in in_dbs:
    data += db.get_users()
  with out_db.Session(commit = True) as s:
    for dp in data:
      s.add(
        UserSession(
          **{
            "user_id"          : dp.user_id,
            "user_ip"          : dp.user_ip,
            "engineer"         : dp.engineer,
            "schedule"         : dp.schedule,
            "dataset_distr"    : dp.dataset_distr,
            "label_distr"      : dp.label_distr,
            "prediction_distr" : dp.prediction_distr,
            "session"          : dp.session,
            "num_predictions"  : dp.num_predictions,
            "date_added"       : dp.date_added,
          }
        )
      )
  return

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



def merge_session(in_dbs: typing.List[TuringDB], out_db: TuringDB) -> None:
  data = None
  for db in in_dbs:
    new_s = db.get_session()
    if data is None:
      data = new_s
    else:
      data.num_user_ids = data.num_user_ids + new_s.num_user_ids
      data.user_ids = json.dumps(json.loads(data.user_ids) + json.loads(new_s.user_ids))
      data.num_user_ips = data.num_user_ips + new_s.num_user_ips
      data.user_ips = json.dumps(json.loads(data.user_ips) + json.loads(new_s.user_ips))
      ## engineer_distr
      e1, e2 = json.loads(data.engineer_distr), json.loads(new_s.engineer_distr)
      e1['engineer'] += e2['engineer']
      e1['non-engineer'] += e2['non-engineer']
      data.engineer_distr = json.dumps(e1)
      ## num_predictions.
      e1, e2 = json.loads(data.num_predictions), json.loads(new_s.num_predictions)
      x1, x2 = json.loads(data.prediction_distr), json.loads(new_s.prediction_distr)
      out = {}
      out2 = {}
      keys = {"GitHub", "BenchPress_directed", "CLgen", "CLSmith", "BenchPress"}
      for l in {"engineer", "non-engineer"}:
        out[l] = {}
        out2[l] = {}
        for k in keys:
          out[l][k] = 0
          out2[l][k] = {
            "label": "human" if k == "GitHub" else "robot",
            "predictions": {
              "human": 0,
              "robot": 0,
            }
          }
          if k in e1[l]:
            out[l][k] += e1[l][k]
            out2[l][k]["predictions"]["human"] += x1[l][k]["predictions"]["human"]
            out2[l][k]["predictions"]["robot"] += x1[l][k]["predictions"]["robot"]
          if k in e2[l]:
            out[l][k] += e2[l][k]
            out2[l][k]["predictions"]["human"] += x2[l][k]["predictions"]["human"]
            out2[l][k]["predictions"]["robot"] += x2[l][k]["predictions"]["robot"]
      data.num_predictions = json.dumps(out)
      data.prediction_distr = json.dumps(out2)
  with out_db.Session(commit = True) as s:
    s.add(
      TuringSession(
        **{
          "num_user_ids"     : data.num_user_ids,
          "user_ids"         : data.user_ids,
          "num_user_ips"     : data.num_user_ips,
          "user_ips"         : data.user_ips,
          "engineer_distr"   : data.engineer_distr,
          "num_predictions"  : data.num_predictions,
          "prediction_distr" : data.prediction_distr,
          "date_added"       : data.date_added,
        }
      )
    )
  return

def merge_results(in_dbs: typing.List[TuringDB], out_db: TuringDB):
  merge_quiz(in_dbs, out_db)
  merge_user(in_dbs, out_db)
  merge_session(in_dbs, out_db)
  return

def main(*args, **kwargs) -> None:
  if FLAGS.out_results_db is None:
    raise ValueError("Please set out results db path")
  if FLAGS.in_results_db is None:
    raise ValueError("Please set path for input DBs")
  out_db = TuringDB(url = "sqlite:///{}".format(pathlib.Path(FLAGS.out_results_db).resolve()))
  in_dbs = [TuringDB(url = "sqlite:///{}".format(pathlib.Path(p).resolve()), must_exist = True) for p in FLAGS.in_results_db.split(',')]
  merge_results(in_dbs, out_db)

if __name__ == "__main__":
  absl_app.run(main)
