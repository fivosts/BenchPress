import portpicker
import json
import typing
import multiprocessing
import flask

# from deeplearning.clgen.util import logging as l

app = flask.Flask(__name__)

class FlaskHandler(object):
  def __init__(self):
    self.in_queue  = None
    self.out_queue = None
    self.backlog   = None
    return

  def set_queues(self, in_queue, out_queue):
    self.in_queue  = in_queue
    self.out_queue = out_queue
    self.backlog   = []
    return

handler = FlaskHandler()

@app.route('/send_message', methods=['PUT'])
def send_message() -> None:
  """
  Collect a serialized bytes message and place to input queue.

  Example command:
    curl -X PUT http://localhost:PORT/send_message \
         --data "some_serialized string object"
         # Maybe also check --data_binary
  """
  msg = flask.request.data
  handler.in_queue.put(msg)
  return 'OK\n', 200

@app.route('/receive_message', methods = ['GET'])
def receive_message() -> bytes:
  """
  Send one element from the output queue.

  Example command:
    curl -X GET http://localhost:PORT/receive_message
  """
  if not handler.out_queue.empty():
    msg = handler.out_queue.get()
    handler.backlog.append(msg)
    return msg, 200
  else:
    return "", 404

@app.route('/receive_all_message', methods = ['GET'])
def receive_all_message() -> typing.List[bytes]:
  """
  Publish all results from the output queue.

  Example command:
    curl -X GET http://localhost:PORT/receive_message
  """
  msg = []
  while not handler.out_queue.empty():
    cur = handler.out_queue.get()
    handler.backlog.append(cur)
    msg.append(cur)
  return bytes(json.dumps(msg), encoding = "utf-8"), 200

@app.route('/get_backlog', methods = ['GET'])
def get_backlog():
  """
  In case a client side error has occured, proactively I have stored
  the whole backlog in memory. To retrieve it, call this method.

  Example command:
    curl -X GET http://localhost:PORT/get_backlog
  """
  return bytes(json.dumps(handler.backlog), encoding = "utf-8"), 200

def serve(in_queue: multiprocessing.Queue, out_queue: multiprocessing.Queue, port: int = None):
  """
  A standalone daemon process executes this function and serves.
  It's purpose is to populate input queue and publish out queue.
  """
  if port is None:
    port = portpicker.pick_unused_port()
  handler.set_queues(in_queue, out_queue)
  kwargs = {
    "port"  : port,
    "host"  : "localhost",
    "debug" : False,
  }
  app.run(**kwargs)
  return
