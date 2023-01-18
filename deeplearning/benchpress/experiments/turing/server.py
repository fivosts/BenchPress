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
import portpicker
import pathlib
import waitress
import subprocess
import typing
import flask
import uuid
import datetime

import numpy as np

from absl import app as absl_app

from deeplearning.benchpress.experiments.turing import db
from deeplearning.benchpress.util import logging as l


"""
TODO list:

1) Encrypt cookies.
2) Append to session database.
3) Append to user database.
4) Append to quiz database.
5) Make all urls viewable as home.
"""

app = flask.Flask(__name__)

class FlaskHandler(object):
  def __init__(self):
    self.databases = None
    self.workspace = None
    self.results_db = None
    self.schedule = None
    return

  def set_params(self, databases: typing.Dict[str, typing.Tuple[str, typing.List[str]]], workspace: pathlib.Path) -> None:
    self.databases = databases
    self.workspace = workspace
    self.session_db = db.TuringDB(url = "sqlite:///{}".format(workspace / "turing_results.db"))
    self.session_db.init_session()
    return

handler = FlaskHandler()

@app.route('/quiz')
def quiz():
  """
  Give a quiz.
  """
  l.logger().info("quiz")
  ## Get schedule from cookies.
  schedule = flask.request.cookies.get("schedule").split(',')
  ## Pop database.
  db_name = schedule.pop(0)
  label, data = handler.databases[db_name]["label"], handler.databases[db_name]["code"]
  ## Sample datapoint.
  question = data[np.random.RandomState().randint(0, len(data) - 1)]
  ## RR-add to the end.
  schedule.append(db_name)
  ## Update cookies.
  resp = flask.make_response(flask.render_template("quiz.html", data = [db_name, label, question]))
  resp.set_cookie("schedule", ','.join(schedule))
  return resp

@app.route('/start')
def start():
  """
  Ask if person knows software.
  """
  ## Create a round robin schedule of held databases.
  l.logger().info("Start")
  schedule = flask.request.cookies.get("schedule")
  print("Cookie schedule: ", schedule)
  if schedule is None:
    schedule = list(handler.databases.keys())
    np.random.RandomState().shuffle(schedule)
    resp = flask.make_response(flask.render_template("start.html"))
    resp.set_cookie("schedule", ','.join(schedule))
  else:
    # my schedule cookie is cached, so go straight to quiz.
    return quiz()

@app.route('/submit', methods = ["POST"])
def submit():
  l.logger().info("Submit")
  if "start" in flask.request.form:
    software = flask.request.cookies.get("engineer")
    l.logger().error("Software cookie: {}".format(software))
    if software is None:
      return start()
    else:
      return quiz()
  return flask.render_template("index.html")

@app.route('/')
def index():
  """
  Render the home page of the test.
  """
  l.logger().info("Index")
  ## Create response
  resp = flask.make_response(flask.render_template("index.html"))
  ## Load user id, or create a new one if no cookie exists.
  user_id = flask.request.cookies.get("user_id")
  if user_id is None:
    user_id = uuid.uuid4()
    resp.set_cookie("user_id", str(user_id))
  ## Assign a new IP anyway.
  user_ip = "XX.XX.XX.XX"
  resp.set_cookie("user_ip", str(user_ip))
  ## Update session database with new user.
  handler.session_db.update_session(user_ids = user_id, user_ips = user_ip, date_added = datetime.datetime.utcnow())
  return resp


"""
  num_user_ids     : int = sql.Column(sql.Integer, nullable = False)
  # A list of all user IDs.
  user_ids         : sql.Column(MutableDict.as_mutable(JSONB), nullable = False)
  # Ips of one user.
  num_user_ips     : int = sql.Column(sql.Integer, nullable = False)
  # A list of all user IPs.
  user_ips         : sql.Column(MutableDict.as_mutable(JSONB), nullable = False)
  # Engineers distribution
  engineer_distr   : sql.Column(MutableDict.as_mutable(JSONB), nullable = False)
  # Total predictions made per engineer and non engineer
  num_predictions  : sql.Column(MutableDict.as_mutable(JSONB), nullable = False)
  # Predictions distribution per engineer and non engineer per dataset with accuracies.
  prediction_distr : sql.Column(MutableDict.as_mutable(JSONB), nullable = False)
  # Date of assigned session.
  date_added       : datetime.datetime = sql.Column(sql.DateTime, nullable=False)


"""




def serve(databases: typing.Dict[str, typing.Tuple[str, typing.List[str]]],
          http_port: int = None,
          host_address: str = '0.0.0.0'
          ) -> None:
  """
  Serving function for Turing test quiz dashboard.
  Receive a list of databases. Each entry specifies:
    a) Name of database
    b) Data
    c) Human or Robot
  """
  try:
    if http_port is None:
      http_port = portpicker.pick_unused_port()
    ## Setup handler.
    handler.set_params(databases, pathlib.Path("./").resolve())
    ## Pretty print hostname.
    hostname = subprocess.check_output(
      ["hostname", "-i"],
      stderr = subprocess.STDOUT,
    ).decode("utf-8").replace("\n", "").split(' ')
    if len(hostname) == 2:
      ips = "ipv4: {}, ipv6: {}".format(hostname[1], hostname[0])
    else:
      ips = "ipv4: {}".format(hostname[0])
    l.logger().warn("Server Public IP: {}:{}".format(ips, http_port))

    waitress.serve(app, host = host_address, port = http_port)
  except KeyboardInterrupt:
    return
  except Exception as e:
    raise e
  return




def main(*args, **kwargs):

  serve(
    databases = {
      "BenchPress": {
        "label": "robot",
        "code" : ["src_A", "src_B"],
      },
      "GitHub": {
        "label": "human",
        "code" : ["src_C", "src_D"],
      }
    },
    http_port = 40822
  )


if __name__ == "__main__":
  absl_app.run(main)