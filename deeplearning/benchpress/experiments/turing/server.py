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
import queue
import multiprocessing
import waitress
import subprocess
import json
import typing
import requests
import time
import flask
import heapq

import numpy as np
from absl import flags

from deeplearning.benchpress.util.turing import db
from deeplearning.benchpress.util import logging as l

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
    self.results_db = db.TuringDB(url = "sqlite:///{}".format(workspace / "turing_results.db"))
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
  db = schedule.pop(0)
  name, hor, data = db
  ## Sample datapoint.
  question = data[np.random.RandomState().randint(0, len(data) - 1)]
  ## RR-add to the end.
  schedule.append(db)
  ## Update cookies.
  resp = flask.render_template("quiz.html", data = [name, hor, question])
  resp.set_cookies("schedule", ','.join([schedule]))
  return resp

@app.route('/start')
def start():
  """
  Ask if person knows software.
  """
  ## Create a round robin schedule of held databases.
  l.logger().info("Start")
  schedule = flask.request.cookies.get("schedule")
  if schedule is None:
    schedule = np.random.shuffle([x for x, _, _ in handler.databases])
    resp = flask.make_response(flask.render_template("start.html"))
    resp.set_cookie("schedule", ','.join([schedule]))
  return flask.render_template("start.html")

@app.route('/start', methods = ['POST'])
def index():
  """
  Receive input from yes/no software question.
  """
  l.logger().info("Start POST")
  text = flask.request.form['text']
  processed_text = text.upper()
  return processed_text

@app.route('/')
def index():
  """
  Main status page of turing test dashboard.
  """
  l.logger().info("Index")
  return flask.render_template("index.html")

@app.route('/', methods = ['POST'])
def index():
  """
  Get input from "START" button.
  """
  l.logger().info("Index POST")
  software = flask.request.cookies.get("engineer")
  button = flask.request.form['start']
  print(button)
  if software is None:
    # start()
    return processed_text
  else:
    ## Go directly to quizzing.
    quiz()
  return "OK\n", 200

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
    handler.set_params(databases)
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
  }
)