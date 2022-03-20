import socket
import pickle
import portpicker
import multiprocessing
import time
import waitress
import subprocess
import json
import flask
import waitress

from absl import flags

from deeplearning.clgen.util import logging as l

FLAGS = flags.FLAGS

FLAGS.DEFINE_string(
  "use_server",
  None,
  "Select to use socket server in the app. If you set to True, the app will know how to use it with respect to the requested task. Choices are 'socket' and 'http'."
)

FLAGS.DEFINE_string(
  "target_host",
  None,
  "Define IP Address of target socket server."
)

FLAGS.DEFINE_integer(
  "listen_port",
  None,
  "Define port this current server listens to."
)

FLAGS.DEFINE_integer(
  "send_port",
  None,
  "Define port this current server listens to."
)

MAX_PAYLOAD_SIZE = 65535

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
    'read_queue'      : 'EMPTY' if handler.read_queue.empty() else 'NOT_EMPTY'
    'write_queue'     : 'EMPTY' if handler.write_queue.empty() else 'NOT_EMPTY'
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

def listen_read_queue(read_queue    : multiprocessing.Queue,
                      port          : int,
                      status        : multiprocessing.Value,
                      listen_status : multiprocessing.Value,
                      ) -> None:
  """
  Keep a socket connection open, listen to incoming traffic
  and populate read_queue queue.
  """
  try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Bind to socket.
    s.bind(('0.0.0.0', port))
    # Set listen settings
    s.listen(2**16)
    # Block until connection is established.
    try:
      conn, addr = s.accept()
      while status.value:
        data = conn.recv(MAX_PAYLOAD_SIZE)
        if len(data) > 0:
          read_queue.put(data)
        else:
          break
      conn.close()
    except KeyboardInterrupt:
      try:
        conn.close()
      except Exception:
        pass
      raise KeyboardInterrupt
    except Exception as e:
      try:
        conn.close()
      except Exception:
        pass
      raise e
    s.close()
  except KeyboardInterrupt:
    s.close()
  except Exception as e:
    s.close()
    raise e
  listen_status.value = False
  return

def send_write_queue(write_queue : multiprocessing.Queue,
                     host        : str,
                     port        : int,
                     status      : multiprocessing.Value,
                     send_status : multiprocessing.Value,
                     ) -> None:
  """
  Keep scanning for new unpublished data in write_queue.
  Fetch them and send them over to the out socket connection.
  """
  try:
    # Create a socket connection.
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while status.value:
      try:
        s.connect((host, port))
        break
      except Exception:
        time.sleep(1)

    while status.value:
      cur = write_queue.get()
      try:
        s.send(cur)
      except BrokenPipeError:
        break
    s.close()
  except KeyboardInterrupt:
    s.close()
  except Exception as e:
    s.close()
    raise e
  send_status.value = False
  return

def socket_serve(read_queue    : multiprocessing.Queue,
                 write_queue   : multiprocessing.Queue,
                 status_bit    : multiprocessing.Value,
                 listen_status : multiprocessing.Value,
                 send_status   : multiprocessing.Value,
                 ) -> None:
  """
  A standalone daemon process executes this function and serves.
  It's purpose is to populate input queue and publish out queue.
  """
  target_host = FLAGS.target_host
  listen_port = FLAGS.listen_port
  send_port   = FLAGS.send_port

  if listen_port is None:
    status_bit.value    = False
    listen_status.value = False
    send_status.value   = False
    raise ValueError("You have to define listen_port to use the socket server.")
  if send_port is None:
    status_bit.value    = False
    listen_status.value = False
    send_status.value   = False
    raise ValueError("You have to define send_port to use the socket server.")
  if target_host is None:
    status_bit.value    = False
    listen_status.value = False
    send_status.value   = False
    raise ValueError("You have to define the IP of the target server to use the socket server.")

  try:
    lp = multiprocessing.Process(
      target = listen_read_queue, 
      kwargs = {
        'read_queue'    : read_queue,
        'port'          : listen_port,
        'status'        : status_bit,
        'listen_status' : listen_status,
      }
    )
    sp = multiprocessing.Process(
      target = send_write_queue,  
      kwargs = {
        'write_queue' : write_queue,
        'host'        : target_host,
        'port'        : send_port,
        'status'      : status_bit,
        'send_status' : send_status,
      }
    )
    lp.start()
    sp.start()

    while status_bit.value:
      time.sleep(1)

    lp.join(timeout = 20)
    sp.join(timeout = 20)

    lp.terminate()
    sp.terminate()
  except KeyboardInterrupt:
    status_bit.value = False

    lp.join(timeout = 20)
    sp.join(timeout = 20)

    lp.terminate()
    sp.terminate()
  except Exception as e:
    status_bit.value = False

    lp.join(timeout = 20)
    sp.join(timeout = 20)

    lp.terminate()
    sp.terminate()
    raise e
  return

def http_serve(read_queue: multiprocessing.Queue, write_queue: multiprocessing.Queue):
  """
  Run http server for read and write workload queues.
  """
  try:
    port = FLAGS.listen_port
    if listen_port is None:
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

  if FLAGS.use_server == 'socket':
    sb, rb, wb = multiprocessing.Value('i', True), multiprocessing.Value('i', True), multiprocessing.Value('i', True)
    p = multiprocessing.Process(
      target = socket_serve,
      kwargs = {
        'read_queue'    : rq,
        'write_queue'   : wq,
        'status_bit'    : sb,
        'listen_status' : rb,
        'send_status'   : wb,
      }
    )
    # p.daemon = True
    p.start()
    return p, sb, (rq, rb), (wq, wb)
  elif FLAGS.use_server == 'http':
    p = multiprocessing.Process(
      target = http_serve,
      kwargs = {
        'read_queue'  : rq,
        'write_queue' : wq,
      }
    )
    p.daemon = True
    p.start()
    return p, None, (rq, None), (wq, None)
  else:
    raise ValueError(FLAGS.use_server)

def start_thread_process():
  """
  This is an easy wrapper to start server from parent routine.
  Starts a new process or thread and returns all the multiprocessing
  elements needed to control the server.
  """
  rq, wq = multiprocessing.Queue(), multiprocessing.Queue()

  if FLAGS.use_server == 'socket':
    sb, rb, wb = multiprocessing.Value('i', True), multiprocessing.Value('i', True), multiprocessing.Value('i', True)
    th = threading.Thread(
      target = socket_serve,
      kwargs = {
        'read_queue'    : rq,
        'write_queue'   : wq,
        'status_bit'    : sb,
        'listen_status' : rb,
        'send_status'   : wb,
      },
    )
    th.start()
    return None, sb, (rq, rb), (wq, wb)
  elif FLAGS.use_server == 'http':
    th = threading.Thread(
      target = http_serve,
      kwargs = {
        'read_queue'  : rq,
        'write_queue' : wq,
      },
      daemon = True
    )
    th.start()
    return None, None, (rq, None), (wq, None)
  else:
    raise ValueError(FLAGS.use_server)
