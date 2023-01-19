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
    self.session_db = None
    self.schedule = None
    return

  def set_params(self, databases: typing.Dict[str, typing.Tuple[str, typing.List[str]]], workspace: pathlib.Path) -> None:
    self.databases = databases
    self.workspace = workspace
    self.session_db = db.TuringDB(url = "sqlite:///{}".format(workspace / "turing_results.db"))
    self.session_db.init_session()
    return
  
  def get_cookie(self, key: str) -> typing.Any:
    resp = flask.request.cookies.get(key)
    if resp is None:
      return resp
    elif key == "schedule":
      return resp.split(',')
    elif key == "engineer":
      return bool(resp)
    elif key in {"user_id", "user_ip"}:
      return str(resp)
    else:
      raise ValueError(key)

  def set_cookie(self, resp, **kwargs) -> None:
    for key, val in kwargs.items():
      if key == "schedule":
        pr_val = ','.join(val)
      elif key in {"user_id", "user_ip", "engineer"}:
        pr_val = str(val)
      else:
        raise ValueError(key)
      resp.set_cookie(key, pr_val)
    return

handler = FlaskHandler()

@app.route('/submit_quiz', methods = ["POST"])
def submit_quiz():
  """
  Capture quiz submission and redirect.
  """
  ## Save entry to databases right here.
  prediction = "human" if "human" in flask.request.form else "robot"
  print(prediction)
  handler.session_db.add_quiz()
  return flask.redirect(flask.url_for('quiz'))

@app.route('/quiz')
def quiz():
  """
  Give a quiz.
  """
  l.logger().info("quiz")
  ## Get schedule from cookies.
  schedule = handler.get_cookie("schedule")
  ## Pop database.
  db_name = schedule.pop(0)
  label, data = handler.databases[db_name]["label"], handler.databases[db_name]["code"]
  ## Sample datapoint.
  question = data[np.random.RandomState().randint(0, len(data) - 1)]
  ## RR-add to the end.
  schedule.append(db_name)
  ## Update cookies.
  resp = flask.make_response(flask.render_template("quiz.html", data = [db_name, label, question]))
  handler.set_cookie(resp, schedule = schedule)
  return resp

@app.route('/submit_engineer', methods = ["POST"])
def submit_engineer():
  """
  Read input from engineer survey question.
  """
  l.logger().critical(flask.request.form)
  user_id  = handler.get_cookie("user_id")
  user_ip  = handler.get_cookie("user_ip")
  schedule = handler.get_cookie("schedule")
  engineer = "yes" in flask.request.form
  ## TODO: Save the engineer information associated with user id.
  handler.session_db.update_user(
    user_id = user_id,
    user_ip = user_ip,
    schedule = schedule,
    engineer = engineer,
  )
  handler.session_db.update_session(
    user_ids = str(user_id),
    user_ips = user_ip,
    engineer_distr = engineer,
  )
  resp = flask.redirect(flask.url_for('quiz'))
  handler.set_cookie(resp, engineer = engineer)
  return resp

@app.route('/start')
def start():
  """
  Ask if person knows software.
  """
  ## Create a round robin schedule of held databases.
  l.logger().info("Start")
  user_id  = handler.get_cookie("user_id")
  user_ip  = handler.get_cookie("user_ip")
  schedule = handler.get_cookie("schedule")
  print("Cookie schedule: ", schedule)
  resp = flask.make_response(flask.render_template("start.html"))
  if schedule is None:
    schedule = list(handler.databases.keys())
    np.random.RandomState().shuffle(schedule)
    handler.set_cookie(resp, schedule = schedule)
  handler.session_db.update_session(
    user_ips = user_ip,
  )
  return resp

@app.route('/submit', methods = ["POST"])
def submit():
  l.logger().info("Submit")
  if "start" in flask.request.form:
    engineer = handler.get_cookie("engineer")
    l.logger().error("Software cookie: {}".format(engineer))
    if engineer is None:
      return flask.redirect(flask.url_for('start'))
    else:
      return flask.redirect(flask.url_for('quiz'))
  return flask.redirect(flask.url_for('index'))

@app.route('/')
def index():
  """
  Render the home page of the test.
  """
  l.logger().info("Index")
  ## Create response
  resp = flask.make_response(flask.render_template("index.html"))
  ## Load user id, or create a new one if no cookie exists.
  user_id = handler.get_cookie("user_id")
  if user_id is None:
    # Create user ID.
    user_id = uuid.uuid4()
    handler.set_cookie(resp, user_id = user_id)

  ## Assign a new IP anyway.
  user_ip = flask.request.remote_addr
  handler.set_cookie(resp, user_ip = user_ip)
  return resp

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