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

from absl import flags

from deeplearning.benchpress.util import logging as l
from deeplearning.benchpress.util import environment

app = flask.Flask(__name__)

class FlaskHandler(object):
  def __init__(self):
    self.read_queue   = None
    self.write_queues = None
    self.reject_queue = None
    self.peers        = None
    self.backlog      = None
    return

  def set_params(self, read_queue, write_queues, reject_queues, manager, work_flag):
    self.read_queue    = read_queue
    self.write_queues  = write_queues
    self.work_flag     = work_flag
    self.reject_queues = reject_queues
    self.my_address    = "http://{}:{}".format(FLAGS.http_server_ip_address, FLAGS.http_port)
    self.peers         = ["http://{}".format(s) for s in FLAGS.http_server_peers]
    self.master_node   = True if self.peers else False
    self.manager       = manager
    self.backlog       = []
    return

handler = FlaskHandler()

@app.route('/')
def index():
  """
  Main status page of turing test dashboard.
  """
  ## DO I need to handle any data here ?
  return flask.render_template("index.html")

@app.route('/', methods = ['POST'])
def index():
  """
  Main status page of turing test dashboard.
  """
  text = flask.request.form['text']
  processed_text = text.upper()
  return processed_text


def start_server_process() -> typing.Tuple[multiprocessing.Process, multiprocessing.Value, multiprocessing.Queue, typing.Dict, typing.Dict]:
  """
  This is an easy wrapper to start server from parent routine.
  Starts a new process or thread and returns all the multiprocessing
  elements needed to control the server.
  """
  m = multiprocessing.Manager()
  rq, wqs, rjqs = multiprocessing.Queue(), m.dict(), m.dict()
  wf = multiprocessing.Value('i', False)
  p = multiprocessing.Process(
    target = http_serve,
    kwargs = {
      'read_queue'    : rq,
      'write_queues'  : wqs,
      'reject_queues' : rjqs,
      'work_flag'     : wf,
      'manager'       : m,
    }
  )
  p.daemon = True
  p.start()
  return p, wf, rq, wqs, rjqs