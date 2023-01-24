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
import json
import sys

import numpy as np

from absl import app as absl_app

from deeplearning.benchpress.experiments.turing import db
from deeplearning.benchpress.util import logging as l

app = flask.Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = -1

class FlaskHandler(object):
  def __init__(self):
    self.databases = None
    self.workspace = None
    self.session_db = None
    self.schedule = None
    self.user_cache = None
    return

  def set_params(self, databases: typing.Dict[str, typing.Tuple[str, typing.List[str]]], workspace: pathlib.Path) -> None:
    self.databases = databases
    self.workspace = workspace
    self.session_db = db.TuringDB(url = "sqlite:///{}".format(workspace / "turing_results.db"))
    self.session_db.init_session()
    self.user_cache = {}
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
    elif key == "quiz_cache":
      try:
        return json.loads(resp)
      except Exception as e:
        l.logger().error(resp)
        raise e
    else:
      raise ValueError(key)

  def set_cookie(self, resp, **kwargs) -> None:
    extra_args = {}
    for key, val in kwargs.items():
      if key == "schedule":
        pr_val = ','.join(val)
      elif key in {"user_id", "user_ip", "engineer"}:
        pr_val = str(val)
      elif key == "quiz_cache":
        expires = kwargs.get("expires")
        if expires is not None:
          extra_args["expires"] = expires
        pr_val = json.dumps(val)
      else:
        continue
      resp.set_cookie(key, pr_val, **extra_args)
    return

handler = FlaskHandler()

@app.route('/submit_quiz', methods = ["POST"])
def submit_quiz():
  """
  Capture quiz submission and redirect.
  """
  ## Save entry to databases right here.
  l.logger().error("Submit quiz.")
  prediction = "human" if "human" in flask.request.form else "robot"
  user_id  = handler.get_cookie("user_id")
  if user_id is None or user_id not in handler.user_cache:
    return flask.redirect(flask.url_for('index'))

  user_ip = handler.user_cache[user_id].get("user_ip", None)
  engineer = handler.user_cache[user_id].get("engineer", None)
  schedule = handler.user_cache[user_id].get("schedule", None)
  quiz_cache = handler.user_cache[user_id].get("quiz_cache", None)

  try:
    handler.session_db.add_quiz(
      dataset    = quiz_cache["dataset"],
      code       = quiz_cache["code"],
      label      = quiz_cache["label"],
      prediction = prediction,
      user_id    = user_id,
      user_ip    = user_ip,
      engineer   = engineer,
      schedule   = schedule,
    )
  except TypeError as e:
    print(e)
    raise e
    l.logger().error("Caught exception.")
    return flask.redirect(flask.url_for('quiz'))
  ## Clear cache for current quiz.
  del handler.user_cache[user_id]["quiz_cache"]
  return flask.redirect(flask.url_for('quiz'))

@app.route('/submit_quiz', methods = ["GET", "PUT"])
def submit_quiz_override():
  l.logger().error("quiz override.")
  return flask.redirect(flask.url_for('quiz'))

@app.route('/quiz')
def quiz():
  """
  Give a quiz.
  
  Cookies:
    gets:
      schedule
      user_id
      user_ip
      engineer
    sets:
      cached_session (All data for a single quiz result.)
  """
  l.logger().info("quiz")
  ## Read cache. IF quiz exists in cache, force user to answer this.
  user_id = handler.get_cookie("user_id")
  l.logger().warn(handler.user_cache)
  set_user_id = False
  if user_id is None or user_id not in handler.user_cache:
    return flask.redirect(flask.url_for('index'))
  quiz_cache = handler.user_cache[user_id].get("quiz_cache", None)
  if quiz_cache is not None:
    l.logger().error("Cached quiz.")
    resp = flask.make_response(
      flask.render_template(
        "quiz.html",
        data = quiz_cache["code"]
      )
    )
  else:
    l.logger().error("New quiz.")
    ## Avoid new users going directly to quiz URL.
    ## Get schedule from cookies.
    schedule = handler.user_cache[user_id].get("schedule", None)
    ## Introduce a little bit of randomness to dataset selection.
    dropout = np.random.RandomState().random()
    if dropout <= 0.3:
      ## Pick a random dataset instead.
      dropout = True
      dataset = schedule[np.random.RandomState().randint(0, len(schedule) - 1)]
    else:
      ## Pop database.
      dropout = False
      dataset = schedule.pop(0)
    label, data = handler.databases[dataset]["label"], handler.databases[dataset]["code"]
    ## Sample datapoint.
    code = data[np.random.RandomState().randint(0, len(data) - 1)]
    if not dropout:
      ## RR-add to the end.
      schedule.append(dataset)
    ## Update cookies.
    resp = flask.make_response(
      flask.render_template(
        "quiz.html",
        data = code
      )
    )
    handler.user_cache[user_id]["schedule"] = schedule
    handler.user_cache[user_id]["quiz_cache"] = {"dataset": dataset, "code": code, "label": label}
  if set_user_id:
    handler.set_cookie(resp, user_id = user_id)
  return resp

@app.route('/submit_engineer', methods = ["POST"])
def submit_engineer():
  """
  Read input from engineer survey question.

  Cookies:
    gets:
      user_id
      user_ip
      schedule
    sets:
      engineer
  """
  l.logger().critical("submit engineer")
  user_id  = handler.get_cookie("user_id")
  if user_id is None:
    user_id = str(uuid.uuid4())
    handler.user_cache[user_id] = {}
    handler.set_cookie(resp, user_id = user_id)
  engineer = handler.user_cache[user_id].get("engineer", None)
  if engineer is not None:
    l.logger().critical("skip engineer")
    return flask.redirect(flask.url_for('index'))
  user_ip = handler.user_cache[user_id].get("user_ip", None)
  schedule = handler.user_cache[user_id].get("schedule", None)
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
  handler.user_cache[user_id]["engineer"] = engineer
  return flask.redirect(flask.url_for('quiz'))

@app.route('/submit_engineer', methods = ["GET", "PUT"])
def submit_engineer_override():
  l.logger().critical("submit engineer override")
  return flask.redirect(flask.url_for('index'))

@app.route('/start')
def start():
  """
  Ask if person knows software. Drops here if engineer not in cookies.

  Cookies:
    gets:
      user_id
      user_ip
      schedule
      engineer (for the purpose of redirecting to index. Avoid re-answering.)
    sets:
      schedule
  """
  ## Create a round robin schedule of held databases.
  l.logger().info("Start")
  user_id  = handler.get_cookie("user_id")
  if user_id is None or user_id not in handler.user_cache:
    return flask.redirect(flask.url_for('index'))
  engineer = handler.user_cache[user_id].get("engineer", None)
  l.logger().error(engineer)
  l.logger().critical(handler.user_cache)
  if engineer is not None:
    return flask.redirect(flask.url_for('index'))
  user_ip = handler.user_cache[user_id].get("user_ip", None)
  schedule = handler.user_cache[user_id].get("schedule", None)

  print("Cookie schedule: ", schedule)
  if schedule is None:
    schedule = list(handler.databases.keys())
    np.random.RandomState().shuffle(schedule)
    handler.user_cache[user_id]["schedule"] = schedule
  handler.session_db.update_session(
    user_ips = user_ip,
  )
  return flask.make_response(flask.render_template("start.html"))

@app.route('/score')
def score():
  """
  Check player's current accuracy.
  """
  user_id = handler.get_cookie("user_id")
  if user_id is None or user_id not in handler.user_cache:
    return flask.redirect(flask.url_for('index'))
  last_total = handler.user_cache[user_id].get("last_total", 0)
  accuracy, total = handler.session_db.get_user_accuracy(user_id = user_id, min_attempts = 10)
  if accuracy is None or (0 < total - last_total < 10):
    return flask.make_response(flask.render_template("score_null.html"))
  else:
    handler.user_cache[user_id]["last_total"] = total
    return flask.make_response(flask.render_template("score.html", data = "{}%".format(int(100 * accuracy))))

@app.route('/submit', methods = ["POST"])
def submit():
  """
  START submit button in homepage.

  Cookies:
    gets:
      engineer
  """
  l.logger().info("Submit")
  user_id = handler.get_cookie("user_id")
  if user_id is None or user_id not in handler.user_cache:
    l.logger().critical(user_id)
    l.logger().critical(handler.user_cache)
    return flask.redirect(flask.url_for('index'))
  if "start" in flask.request.form:
    engineer = handler.user_cache[user_id].get("engineer", None)
    l.logger().error("Software cookie: {}".format(engineer))
    if engineer is None:
      resp = flask.make_response(flask.redirect(flask.url_for('start')))
    else:
      resp = flask.make_response(flask.redirect(flask.url_for('quiz')))
  else:
    resp = flask.make_response(flask.redirect(flask.url_for('index')))
  return resp

@app.route('/')
def index():
  """
  Render the home page of the test.

  Cookies:
    gets:
      user_id
    sets:
      user_id
      user_ip
  """
  l.logger().info("Index")
  ## Create response
  resp = flask.make_response(flask.render_template("index.html"))
  ## Load user id, or create a new one if no cookie exists.
  user_id = handler.get_cookie("user_id")
  if user_id is None:
    # Create user ID.
    user_id = str(uuid.uuid4())
    handler.set_cookie(resp, user_id = user_id)
    handler.user_cache[user_id] = {}
  else:
    is_engineer = handler.session_db.is_engineer(user_id = user_id)
    schedule = handler.session_db.get_schedule(user_id = user_id)
    if is_engineer is not None:
      handler.user_cache[user_id] = {
        'engineer': is_engineer,
        'schedule': schedule,
      }
    else:
      handler.user_cache[user_id] = {}
  ## Assign a new IP anyway.
  user_ip = flask.request.remote_addr
  handler.user_cache[user_id]["user_ip"] = user_ip
  return resp

def serve(databases: typing.Dict[str, typing.Tuple[str, typing.List[str]]],
          workspace_path: pathlib.Path,
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
    handler.set_params(databases, workspace_path)
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

    waitress.serve(app, host = host_address, port = http_port, threads = 32)
  except KeyboardInterrupt:
    return
  except Exception as e:
    raise e
  return

def main(*args, **kwargs):

  dbs = json.load(open(sys.argv[1], 'r'))
  serve(
    databases      = dbs,
    workspace_path = pathlib.Path(sys.argv[2]).resolve(),
    http_port      = 40822,
  )

if __name__ == "__main__":
  absl_app.run(main)

## ./benchpress ./get_human_likely/human_or_robot.json ./get_human_likely/test