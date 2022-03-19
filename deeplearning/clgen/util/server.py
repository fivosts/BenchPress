import socket
import pickle
import portpicker

from deeplearning.clgen.util import logging as l

MAX_PAYLOAD_SIZE = 65535

def listen_in_queue() -> None:
  """
  Keep a socket connection open, listen to incoming traffic
  and populate in_queue queue.
  """
  try:
    HOST = 'localhost'
    PORT = 8080
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen(2**32)

    while True:
      conn, addr = s.accept()
      data = conn.recv(MAX_PAYLOAD_SIZE)
      in_queue.put(data)
      conn.close()
  except Exception as e:
    if conn:
      conn.close()
    s.close()
    raise e
  s.close()
  return

def send_out_queue() -> None:
  """
  Keep scanning for new unpublished data in out_queue.
  Fetch them and send them over to the out socket connection.
  """
  try:
    HOST = 'localhost'
    PORT = 8085
    # Create a socket connection.
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while True:
      try:
        s.connect((HOST, PORT))
        break
      except Exception:
        pass

    while True:
      cur = out_queue.get()
      s.send(cur)
  except Exception e:
    s.close()
    raise e
  s.close()
  return

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
