import socket
import pickle
import portpicker
import multiprocessing
import time

# from deeplearning.clgen.util import logging as l

MAX_PAYLOAD_SIZE = 65535

def listen_in_queue(in_queue: multiprocessing.Queue,
                    host    : str,
                    port    : int
                    ) -> None:
  """
  Keep a socket connection open, listen to incoming traffic
  and populate in_queue queue.
  """
  try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("0.0.0.0", port))
    s.listen(2**16)
    print("Connected for listen!")
    while True:
      try:
        conn, addr = s.accept()
        data = conn.recv(MAX_PAYLOAD_SIZE)
        in_queue.put(data)
        conn.close()
      except Exception as e:
        conn.close()
  except Exception as e:
    s.close()
    raise e
  s.close()
  return

def send_out_queue(out_queue: multiprocessing.Queue,
                   host    : str,
                   port    : int
                   ) -> None:
  """
  Keep scanning for new unpublished data in out_queue.
  Fetch them and send them over to the out socket connection.
  """
  try:
    # Create a socket connection.
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while True:
      try:
        s.connect((host, port))
        break
      except Exception as e:
        time.sleep(3)
        print(e)

    print("Connected to send!")
    while True:
      print("IDLE to collect")
      cur = out_queue.get()
      print("Sending message")
      s.send(cur)
  except Exception as e:
    s.close()
    raise e
  s.close()
  return

def serve(in_queue    : multiprocessing.Queue,
          out_queue   : multiprocessing.Queue,
          target_host : str,
          listen_port : int = None,
          send_port   : int = None,
          ):
  """
  A standalone daemon process executes this function and serves.
  It's purpose is to populate input queue and publish out queue.
  """
  try:
    if listen_port is None:
      listen_port = portpicker.pick_unused_port()
    if send_port is None:
      send_port = portpicker.pick_unused_port()

    lp = multiprocessing.Process(
      target = listen_in_queue, 
      kwargs = {
        'in_queue' : in_queue,
        'host'     : target_host,
        'port'     : listen_port
      }
    )
    sp = multiprocessing.Process(
      target = send_out_queue,  
      kwargs = {
        'out_queue' : out_queue,
        'host'     : target_host,
        'port'      : send_port
      }
    )

    lp.start()
    sp.start()

    while True:
      cur = in_queue.get()
      print(cur)
      out_queue.put(cur)

    lp.join()
    sp.join()
  except KeyboardInterrupt:
    lp.terminate()
    sp.terminate()
  return


class foo():
  def __init__(self, x):
    self.x = x
  def add(self, x):
    return foo(self.x + x)
import pickle

def client():


  a = foo(20)
  ser = pickle.dumps(a)

  iiq, ooq = multiprocessing.Queue(), multiprocessing.Queue()

  p = multiprocessing.Process(
    target = serve,
    kwargs = {
      'in_queue': iiq,
      'out_queue': ooq,
      'target_host': "http://cc3.inf.ed.ac.uk",
      'listen_port': 8085,
      'send_port': 8080,
    }
  )
  p.start()

  ooq.put(ser)
  while True:
    cur = iiq.get()
    obj = pickle.loads(cur)
    print(obj.x)

  return

def server():
  iq, oq = multiprocessing.Queue(), multiprocessing.Queue()
  serve(iq, oq, "http://cc1.inf.ed.ac.uk", 8080, 8085)