# coding=utf-8
# Copyright 2022 Foivos Tsimpourlas.
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

FLAGS = flags.FLAGS

flags.DEFINE_boolean(
  "use_http_server",
  False,
  "Select to use http server in the app. If you set to True, the app will know how to use it with respect to the requested task."
)

flags.DEFINE_integer(
  "http_port",
  40822,
  "Define port this current server listens to."
)

flags.DEFINE_string(
  "http_server_ip_address",
  "cc1.inf.ed.ac.uk",
  "Set the target IP address of the host http server."
)

flags.DEFINE_list(
  "http_server_peers",
  [],
  "Set comma-separated http address <dns_name:port> to load balance on secondary nodes."
)

flags.DEFINE_string(
  "host_address",
  "localhost",
  "Specify address where http server will be set."
)

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

@app.route('/write_message', methods=['PUT'])
def write_message(): # Expects serialized json file, one list of dictionaries..
  """
  This function receives new kernels that need to be computed.

  Collect a json file with data and send to computation..

  Example command:
    curl -X PUT http://localhost:PORT/write_message \
         --header "Content-Type: application/json" \
         -d @/path/to/json/file.json
  """
  source = flask.request.headers.get("Server-Name")
  if source is None:
    return "Source address not provided.", 404

  if source not in handler.write_queues:
    handler.write_queues[source] = handler.manager.list()
  if source not in handler.reject_queues:
    handler.reject_queues[source] = handler.manager.list()

  data = flask.request.json

  if not isinstance(data, list):
    return "ERROR: JSON Input has to be a list of dictionaries. One for each entry.\n", 400

  if handler.master_node:
    # 1. Read the pending queue from all peer nodes.
    # A min heap is created that stores server nodes with their queue size.
    heap = []
    for add in handler.peers:
      status, sc = client_read_queue_size(add)
      size = status['read_queue_size']
      if sc < 200:
        l.logger().error("{}, {}".format(size, sc))
      else:
        heap.append([size, add])
    heap.append([handler.read_queue.qsize(), handler.my_address])
    heapq.heapify(heap)
    # 2. Create the schedule: dict[node_address -> list of workload]
    schedule = {}
    for entry in data:
      # For every kernel to be computed.
      # Pop the server with the least load.
      min_load = heapq.heappop(heap)
      size, address = min_load
      if address not in schedule:
        schedule[address] = []
      schedule[address].append(entry)
      heapq.heappush(heap, [size+1, address])
    # 3. For each compute node other than myself, do a write_message request.
    for node, workload in schedule.items():
      # If I need to add to my workload, just add to queue.
      if node == handler.my_address:
        for entry in workload:
          handler.read_queue.put([source, entry])
      # Otherwise run a request
      else:
        client_put_request(workload, address = node, servername = source)
  else:
    for entry in data:
      handler.read_queue.put([source, entry])

  return 'OK\n', 200

@app.route('/read_message', methods = ['GET'])
def read_message() -> bytes:
  """
  Publish all the predicted results of the write_queue.
  Before flushing the write_queue, save them into the backlog.

  Example command:
    curl -X GET http://localhost:PORT/read_message
  """
  source = flask.request.headers.get("Server-Name")
  if source not in handler.write_queues:
    l.logger().warn("Source {} not in write_queues: {}".format(source, ', '.join(handler.write_queues.keys())))
    ret = []
  else:
    ret = [r for r in handler.write_queues[source]]
    handler.write_queues[source] = handler.manager.list()
    handler.backlog += [[source, r] for r in ret]

  if handler.master_node:
    queue = handler.peers
    while queue:
      peer = queue.pop(0)
      sc = client_status_request()[1]
      if sc < 300:
        ret += client_get_request(address = peer, servername = source)
      else:
        queue.append(peer)
        time.sleep(2)
  return bytes(json.dumps(ret), encoding="utf-8"), 200

@app.route('/read_rejects', methods = ['GET'])
def read_rejects() -> bytes:
  """
  Publish all the predicted results of the write_queue.
  Before flushing the write_queue, save them into the backlog.

  Example command:
    curl -X GET http://localhost:PORT/read_rejects
  """
  source = flask.request.headers.get("Server-Name")
  if source not in handler.reject_queues:
    l.logger().warn("Source {} not in reject_queues: {}".format(source, ', '.join(handler.reject_queues.keys())))
    ret = []
  else:
    ret = [r for r in handler.reject_queues[source]]

  if handler.master_node:
    for peer in handler.peers:
      ret += client_get_rejects(address = peer, servername = source)
  return bytes(json.dumps(ret), encoding="utf-8"), 200

@app.route('/read_reject_labels', methods = ['GET'])
def read_reject_labels() -> bytes:
  """
  Get labels of rejected OpenCL kernels.

  Example command:
    curl -X GET http://localhost:PORT/read_reject_labels
  """
  labels = {}
  source = flask.request.headers.get("Server-Name")
  if source is None:
    return "Server-Name is undefined", 404
  if source not in handler.reject_queues:
    l.logger().warn("Source {} not in reject_queues: {}".format(source, ', '.join(handler.reject_queues.keys())))
    ret = []
  else:
    ret = [r for r in handler.reject_queues[source]]
  
  for c in ret:
    if c['runtime_features']['label'] not in labels:
      labels[c['runtime_features']['label']] = 1
    else:
      labels[c['runtime_features']['label']] += 1

  if handler.master_node:
    for peer in handler.peers:
      peer_labels = client_read_reject_labels(address = peer, servername = source)
      for lab, frq in peer_labels.items():
        if lab not in labels:
          labels[lab] = frq
        else:
          labels[lab] += frq
  return bytes(json.dumps(labels), encoding="utf-8"), 200

@app.route('/read_queue_size', methods = ['GET'])
def read_queue_size() -> bytes:
  """
  Read size of pending workload in read_queue for current compute node.
  """
  return handler.read_queue.qsize(), 200

@app.route('/get_backlog', methods = ['GET'])
def get_backlog() -> bytes:
  """
  In case a client side error has occured, proactively I have stored
  the whole backlog in memory. To retrieve it, call this method.

  Example command:
    curl -X GET http://localhost:PORT/get_backlog
  """
  backlog = handler.backlog
  if handler.master_node:
    for peer in handler.peers:
      backlog += client_get_backlog(address = peer)
  return bytes(json.dumps(backlog), encoding = "utf-8"), 200

@app.route('/status', methods = ['GET'])
def status():
  """
  Read the workload status of the http server.
  """
  source = flask.request.headers.get("Server-Name")
  if source is None:
    return "Server-Name is undefined", 404

  status = {
    'read_queue'        : 'EMPTY' if handler.read_queue.empty() else 'NOT_EMPTY',
    'write_queue'       : 'EMPTY' if source not in handler.write_queues or len(handler.write_queues[source]) == 0 else 'NOT_EMPTY',
    'reject_queue'      : 'EMPTY' if source not in handler.reject_queues or len(handler.reject_queues[source]) == 0 else 'NOT_EMPTY',
    'work_flag'         : 'WORKING' if handler.work_flag.value else 'IDLE',
    'read_queue_size'   : handler.read_queue.qsize(),
    'write_queue_size'  : -1 if source not in handler.write_queues else len(handler.write_queues[source]),
    'reject_queue_size' : -1 if source not in handler.reject_queues else len(handler.reject_queues[source]),
  }

  if handler.master_node:
    for peer in handler.peers:
      peer_status, sc = client_status_request(address = peer, servername = source)
      if sc < 200:
        l.logger().error("Error at {} /status".format(peer))
      status['read_queue']   = 'EMPTY' if peer_status['read_queue']   == 'EMPTY' and status['read_queue']   == 'EMPTY' else 'NOT_EMPTY'
      status['write_queue']  = 'EMPTY' if peer_status['write_queue']  == 'EMPTY' and status['write_queue']  == 'EMPTY' else 'NOT_EMPTY'
      status['reject_queue'] = 'EMPTY' if peer_status['reject_queue'] == 'EMPTY' and status['reject_queue'] == 'EMPTY' else 'NOT_EMPTY'
      status['work_flag']    = 'IDLE'  if peer_status['work_flag']    == 'IDLE'  and status['work_flag']    == 'IDLE'  else 'WORKING'
      status['read_queue_size']   += peer_status['read_queue_size']
      status['write_queue_size']  += peer_status['write_queue_size']
      status['reject_queue_size'] += peer_status['reject_queue_size']

  if status['read_queue'] == 'EMPTY' and status['write_queue'] == 'EMPTY':
    return bytes(json.dumps(status), encoding = 'utf-8'), 200 + (100 if handler.work_flag.value else 0)
  elif status['read_queue'] == 'EMPTY' and status['write_queue'] == 'NOT_EMPTY':
    return bytes(json.dumps(status), encoding = 'utf-8'), 201 + (100 if handler.work_flag.value else 0)
  elif status['read_queue'] == 'NOT_EMPTY' and status['write_queue'] == 'EMPTY':
    return bytes(json.dumps(status), encoding = 'utf-8'), 202 + (100 if handler.work_flag.value else 0)
  elif status['read_queue'] == 'NOT_EMPTY' and status['write_queue'] == 'NOT_EMPTY':
    return bytes(json.dumps(status), encoding = 'utf-8'), 203 + (100 if handler.work_flag.value else 0)

@app.route('/ping', methods = ['PUT'])
def ping():
  """
  A peer compute node receives a ping from master node before initializing the compute network.
  """
  source = flask.request.headers.get("Server-Name")
  if source is None:
    return "Server-Name is undefined", 404

  data = flask.request.json
  handler.peers = [x for x in data['peers'] if x != handler.my_address] + [data['master']]
  return ",".join(handler.peers), 200

@app.route('/', methods = ['GET', 'POST', 'PUT'])
def index():
  """
  In case a client side error has occured, proactively I have stored
  the whole backlog in memory. To retrieve it, call this method.

  Example command:
    curl -X GET http://localhost:PORT/get_backlog
  """
  multi_status = {
    'read_queue'      : 'EMPTY' if handler.read_queue.empty() else 'NOT_EMPTY',
    'read_queue_size' : handler.read_queue.qsize(),
    'work_flag'       : 'WORKING' if handler.work_flag.value else 'IDLE',
  }
  it = set(handler.write_queues.keys())
  it.update(set(handler.reject_queues.keys()))
  multi_status['out_servers'] = {}
  for hn in it:
    status = {
      'write_queue'       : 'EMPTY' if hn in handler.write_queues and len(handler.write_queues[hn]) == 0 else 'NOT_EMPTY',
      'reject_queue'      : 'EMPTY' if hn in handler.reject_queues and len(handler.reject_queues[hn]) == 0 else 'NOT_EMPTY',
      'write_queue_size'  : len(handler.write_queues[hn]) if hn in handler.write_queues else 0,
      'reject_queue_size' : len(handler.reject_queues[hn]) if hn in handler.reject_queues else 0,
    }
    multi_status['out_servers'][hn] = status
  return flask.render_template("index.html", data = multi_status)

def http_serve(read_queue    : multiprocessing.Queue,
               write_queues  : 'multiprocessing.Dict',
               reject_queues : 'multiprocessing.Dict',
               work_flag     : multiprocessing.Value,
               manager       : multiprocessing.Manager,
               ) -> None:
  """
  Run http server for read and write workload queues.
  """
  try:
    port = FLAGS.http_port
    if port is None:
      port = portpicker.pick_unused_port()
    handler.set_params(read_queue, write_queues, reject_queues, manager, work_flag)
    hostname = subprocess.check_output(
      ["hostname", "-i"],
      stderr = subprocess.STDOUT,
    ).decode("utf-8").replace("\n", "").split(' ')
    if len(hostname) == 2:
      ips = "ipv4: {}, ipv6: {}".format(hostname[1], hostname[0])
    else:
      ips = "ipv4: {}".format(hostname[0])
    l.logger().warn("Server Public IP: {}:{}".format(ips, port))

    if handler.master_node:
      l.logger().info("This is master compute server at {}.".format(handler.my_address))
      l.logger().info("Idling until I ensure all peer compute servers are responding:\n{}".format('\n'.join(handler.peers)))
      queue = [[p, 0] for p in handler.peers]
      while queue:
        cur = queue.pop(0)
        _, sc = ping_peer_request(cur[0], handler.peers, handler.my_address)
        if sc != 200:
          queue.append([cur[0], cur[1] + 1])
        else:
          l.logger().info("Successfully connected to {}, {} attempts".format(cur[0], cur[1]))
        time.sleep(5)
      l.logger().info("Successfully connected to all peers")
    waitress.serve(app, host = FLAGS.host_address, port = port)
  except KeyboardInterrupt:
    return
  except Exception as e:
    raise e
  return

##########################
# Client request methods #
##########################

def ping_peer_request(peer: str, peers: typing.List[str], master_node: str) -> int:
  """
  Master compute node peers a peer compute node to check if it's alive.
  If so, also pass the information of all peers that must be alive
  inside the compute network.
  """
  try:
    r = requests.put(
          "{}/ping".format(peer),
          data = json.dumps({'peers': peers, 'master': master_node}),
          headers = {
            "Content-Type": "application/json",
            "Server-Name": environment.HOSTNAME}
          )
  except Exception as e:
    l.logger().warn("PUT status Request at {}/ping has failed.".format(peer))
    print(e)
    return None, 404
  return r.content, r.status_code


def client_status_request(address: str = None, servername: str = None) -> typing.Tuple[typing.Dict, int]:
  """
  Get status of http server.
  """
  try:
    if FLAGS.http_port == -1 or address:
      r = requests.get(
        "{}/status".format(FLAGS.http_server_ip_address if not address else address),
        headers = {"Server-Name": (environment.HOSTNAME if not servername else servername)}
      )
    else:
      r = requests.get(
        "http://{}:{}/status".format(FLAGS.http_server_ip_address, FLAGS.http_port),
        headers = {"Server-Name": (environment.HOSTNAME if not servername else servername)}
      )
  except Exception as e:
    l.logger().error("GET status Request at {}:{} has failed.".format(FLAGS.http_server_ip_address, FLAGS.http_port))
    raise e
  return r.json(), r.status_code

def client_get_request(address: str = None, servername: str = None) -> typing.List[typing.Dict]:
  """
  Helper function to perform get request at /read_message of http target host.
  """
  try:
    if FLAGS.http_port == -1 or address:
      r = requests.get(
        "{}/read_message".format(FLAGS.http_server_ip_address if not address else address),
        headers = {"Server-Name": (environment.HOSTNAME if servername is None else servername)}
      )
    else:
      r = requests.get(
        "http://{}:{}/read_message".format(FLAGS.http_server_ip_address, FLAGS.http_port),
        headers = {"Server-Name": (environment.HOSTNAME if servername is None else servername)}
      )
  except Exception as e:
    l.logger().error("GET Request at {}:{} has failed.".format(FLAGS.http_server_ip_address, FLAGS.http_port))
    raise e
  if r.status_code == 200:
    return r.json()
  else:
    l.logger().error("Error code {} in read_message request.".format(r.status_code))
  return None

def client_get_rejects(address: str = None, servername: str = None) -> typing.List[typing.Dict]:
  """
  Helper function to perform get request at /read_rejects of http target host.
  """
  try:
    if FLAGS.http_port == -1 or address:
      r = requests.get(
        "{}/read_rejects".format(FLAGS.http_server_ip_address if not address else address),
        headers = {"Server-Name": (environment.HOSTNAME if servername is None else servername)}
      )
    else:
      r = requests.get(
        "http://{}:{}/read_rejects".format(FLAGS.http_server_ip_address, FLAGS.http_port),
        headers = {"Server-Name": (environment.HOSTNAME if servername is None else servername)}
      )
  except Exception as e:
    l.logger().error("GET Request at {}:{} has failed.".format(FLAGS.http_server_ip_address, FLAGS.http_port))
    raise e
  if r.status_code == 200:
    return r.json()
  else:
    l.logger().error("Error code {} in read_rejects request.".format(r.status_code))
  return None

def client_read_reject_labels(address: str = None, servername: str = None) -> typing.List[typing.Dict]:
  """
  Read the frequency table of labels for rejected benchmarks.
  """
  try:
    if FLAGS.http_port == -1 or address:
      r = requests.get(
        "{}/read_reject_labels".format(FLAGS.http_server_ip_address if not address else address),
        headers = {"Server-Name": (environment.HOSTNAME if servername is None else servername)}
      )
    else:
      r = requests.get(
        "http://{}:{}/read_reject_labels".format(FLAGS.http_server_ip_address, FLAGS.http_port),
        headers = {"Server-Name": (environment.HOSTNAME if servername is None else servername)}
      )
  except Exception as e:
    l.logger().error("GET Request at {}:{} has failed.".format(FLAGS.http_server_ip_address, FLAGS.http_port))
    raise e
  if r.status_code == 200:
    return r.json()
  else:
    l.logger().error("Error code {} in read_reject_labels request.".format(r.status_code))
  return None

def client_get_backlog(address: str = None) -> typing.List[typing.Dict]:
  """
  Read backlog from compute node.
  """
  try:
    if FLAGS.http_port == -1 or address:
      r = requests.get(
        "{}/get_backlog".format(FLAGS.http_server_ip_address if not address else address),
      )
    else:
      r = requests.get(
        "http://{}:{}/get_backlog".format(FLAGS.http_server_ip_address, FLAGS.http_port),
      )
  except Exception as e:
    l.logger().error("GET Request at {}:{} has failed.".format(FLAGS.http_server_ip_address, FLAGS.http_port))
    raise e
  if r.status_code == 200:
    return r.json()
  else:
    l.logger().error("Error code {} in get_backlog request.".format(r.status_code))
  return None

def client_put_request(msg: typing.List[typing.Dict], address: str = None, servername: str = None) -> None:
  """
  Helper function to perform put at /write_message of http target host.
  """
  try:
    if FLAGS.http_port == -1 or address:
      r = requests.put(
        "{}/write_message".format(FLAGS.http_server_ip_address if not address else address),
        data = json.dumps(msg),
        headers = {
          "Content-Type": "application/json",
          "Server-Name": (environment.HOSTNAME if servername is None else servername)
        }
      )
    else:
      r = requests.put(
        "http://{}:{}/write_message".format(FLAGS.http_server_ip_address, FLAGS.http_port),
        data = json.dumps(msg),
        headers = {
          "Content-Type": "application/json",
          "Server-Name": (environment.HOSTNAME if servername is None else servername)
        }
      )
  except Exception as e:
    l.logger().error("PUT Request at {}:{} has failed.".format(FLAGS.http_server_ip_address, FLAGS.http_port))
    raise e
  if r.status_code != 200:
    l.logger().error("Error code {} in write_message request.".format(r.status_code))
  return

def client_read_queue_size(address: str) -> int:
  """
  Read the pending queue size of a compute node.
  """
  try:
    r = requests.get("{}/status".format(address), headers = {"Server-Name": environment.HOSTNAME})
  except Exception as e:
    l.logger().error("GET status Request at {} has failed.".format(address))
    raise e
  return r.json(), r.status_code

########################

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
