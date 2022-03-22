import portpicker
import multiprocessing
import waitress
import subprocess
import json
import flask

from absl import flags

from deeplearning.clgen.util import logging as l

FLAGS = flags.FLAGS

flags.DEFINE_boolean(
  "use_http_server",
  False,
  "Select to use http server in the app. If you set to True, the app will know how to use it with respect to the requested task."
)

flags.DEFINE_integer(
  "http_port",
  None,
  "Define port this current server listens to."
)

app = flask.Flask(__name__)

class FlaskHandler(object):
  def __init__(self):
    self.read_queue  = None
    self.write_queue = None
    self.backlog   = None
    return

  def set_queues(self, read_queue, write_queue):
    self.read_queue  = read_queue
    self.write_queue = write_queue
    self.backlog   = []
    return

handler = FlaskHandler()

@app.route('/write_message', methods=['PUT'])
def write_message():
  """
  Collect a json file with data and send to computation..

  Example command:
    curl -X PUT http://localhost:PORT/write_message \
         --header "Content-Type: application/json" \
         -d @/path/to/json/file.json
  """
  data = flask.request.json
  if not isinstance(data, list):
    return "ERROR: JSON Input has to be a list of dictionaries. One for each entry.\n", 400
  for entry in data:
    handler.read_queue.put(bytes(json.dumps(entry), encoding = "utf-8"))
  return 'OK\n', 200

@app.route('/read_message', methods = ['GET'])
def read_message():
  """
  Publish all the predicted results of the write_queue.
  Before flushing the write_queue, save them into the backlog.

  Example command:
    curl -X GET http://localhost:PORT/read_message
  """
  ret = []
  while not handler.write_queue.empty():
    cur = handler.write_queue.get()
    ret.append(json.loads(cur))
  handler.backlog += ret
  return bytes(json.dumps(ret), encoding="utf-8"), 200

@app.route('/get_backlog', methods = ['GET'])
def get_backlog():
  """
  In case a client side error has occured, proactively I have stored
  the whole backlog in memory. To retrieve it, call this method.

  Example command:
    curl -X GET http://localhost:PORT/get_backlog
  """
  return bytes(json.dumps(handler.backlog), encoding = "utf-8"), 200

@app.route('/status', methods = ['GET'])
def status():
  """
  Read the workload status of the http server.
  """
  status = {
    'read_queue'      : 'EMPTY' if handler.read_queue.empty() else 'NOT_EMPTY',
    'write_queue'     : 'EMPTY' if handler.write_queue.empty() else 'NOT_EMPTY',
    'read_queue_size' : handler.read_queue.qtsize(),
    'write_queue_size': handler.write_queue.qtsize(),
  }

  if status['read_queue'] == 'EMPTY' and status['write_queue'] == 'EMPTY':
    return bytes(json.dumps(status), encoding = 'utf-8'), 203
  elif status['read_queue'] == 'EMPTY' and status['write_queue'] == 'NOT_EMPTY':
    return bytes(json.dumps(status), encoding = 'utf-8'), 202
  elif status['read_queue'] == 'NOT_EMPTY' and status['write_queue'] == 'EMPTY':
    return bytes(json.dumps(status), encoding = 'utf-8'), 201
  elif status['read_queue'] == 'NOT_EMPTY' and status['write_queue'] == 'NOT_EMPTY':
    return bytes(json.dumps(status), encoding = 'utf-8'), 200

@app.route('/', methods = ['GET', 'POST', 'PUT'])
def index():
  """
  In case a client side error has occured, proactively I have stored
  the whole backlog in memory. To retrieve it, call this method.

  Example command:
    curl -X GET http://localhost:PORT/get_backlog
  """
  status = {
    'read_queue'      : 'EMPTY' if handler.read_queue.empty() else 'NOT_EMPTY',
    'write_queue'     : 'EMPTY' if handler.write_queue.empty() else 'NOT_EMPTY',
    'read_queue_size' : handler.read_queue.qtsize(),
    'write_queue_size': handler.write_queue.qtsize(),
  }
  return '\n'.join(["{}: {}".format(k, v) for k, v in status.items()]), 200

def http_serve(read_queue: multiprocessing.Queue, write_queue: multiprocessing.Queue):
  """
  Run http server for read and write workload queues.
  """
  try:
    port = FLAGS.http_port
    if http_port is None:
      port = portpicker.pick_unused_port()
    handler.set_queues(read_queue, write_queue)
    hostname = subprocess.check_output(
      ["hostname", "-i"],
      stderr = subprocess.STDOUT,
    ).decode("utf-8").replace("\n", "").split(' ')
    if len(hostname) == 2:
      ips = "ipv4: {}, ipv6: {}".format(hostname[1], hostname[0])
    else:
      ips = "ipv4: {}".format(hostname[0])
    l.getLogger().warn("Server Public IP: {}".format(ips))
    waitress.serve(app, host = '0.0.0.0', port = port)
  except KeyboardInterrupt:
    return
  except Exception as e:
    raise e
  return

def start_server_process():
  """
  This is an easy wrapper to start server from parent routine.
  Starts a new process or thread and returns all the multiprocessing
  elements needed to control the server.
  """
  rq, wq = multiprocessing.Queue(), multiprocessing.Queue()
  p = multiprocessing.Process(
    target = http_serve,
    kwargs = {
      'read_queue'  : rq,
      'write_queue' : wq,
    }
  )
  p.daemon = True
  p.start()
  return p, rq, wq

def start_thread_process():
  """
  This is an easy wrapper to start server from parent routine.
  Starts a new process or thread and returns all the multiprocessing
  elements needed to control the server.
  """
  rq, wq = multiprocessing.Queue(), multiprocessing.Queue()
  th = threading.Thread(
    target = http_serve,
    kwargs = {
      'read_queue'  : rq,
      'write_queue' : wq,
    },
    daemon = True
  )
  th.start()
  return p, rq, wq
