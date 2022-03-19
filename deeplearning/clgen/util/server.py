import socket
import pickle
import portpicker
import multiprocessing
import time

from absl, import flags

from deeplearning.clgen.util import logging as l

FLAGS = flags.FLAGS

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

def listen_in_queue(in_queue: multiprocessing.Queue,
                    port    : int,
                    status  : multiprocessing.Value,
                    ) -> None:
  """
  Keep a socket connection open, listen to incoming traffic
  and populate in_queue queue.
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
          in_queue.put(data)
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
  return

def send_out_queue(out_queue: multiprocessing.Queue,
                   host    : str,
                   port    : int,
                   status  : multiprocessing.Value,
                   ) -> None:
  """
  Keep scanning for new unpublished data in out_queue.
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
      cur = out_queue.get()
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
  return

def serve(in_queue   : multiprocessing.Queue,
          out_queue  : multiprocessing.Queue,
          status_bit : multiprocessing.Value,
          ) -> None:
  """
  A standalone daemon process executes this function and serves.
  It's purpose is to populate input queue and publish out queue.
  """
  target_host = FLAGS.target_host
  listen_port = FLAGS.listen_port
  send_port   = FLAGS.send_port

  if listen_port is None:
    raise ValueError("You have to define listen_port to use the socket server.")
  if send_port is None:
    raise ValueError("You have to define send_port to use the socket server.")
  if target_host is None:
    raise ValueError("You have to define the IP of the target server to use the socket server.")

  try:
    lp = multiprocessing.Process(
      target = listen_in_queue, 
      kwargs = {
        'in_queue' : in_queue,
        'port'     : listen_port,
        'status'   : status_bit,
      }
    )
    sp = multiprocessing.Process(
      target = send_out_queue,  
      kwargs = {
        'out_queue' : out_queue,
        'host'      : target_host,
        'port'      : send_port,
        'status'    : status_bit,
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

def client():

  iiq, ooq = multiprocessing.Queue(), multiprocessing.Queue()

  status_bit = multiprocessing.Value('i', True)

  p = multiprocessing.Process(
    target = serve,
    kwargs = {
      'in_queue': iiq,
      'out_queue': ooq,
      'status_bit': status_bit,
      'target_host': "localhost",
      'listen_port': 8088,
      'send_port': 8083,
    }
  )
  p.start()

  while True:
    cur = iiq.get()
    obj = pickle.loads(cur)
    print(obj.x)
    time.sleep(0.1)
    ooq.put(pickle.dumps(obj.add(1)))

  return

def server():
  iiq, ooq = multiprocessing.Queue(), multiprocessing.Queue()

  status_bit = multiprocessing.Value('i', True)

  p = multiprocessing.Process(
    target = serve,
    kwargs = {
      'in_queue': iiq,
      'out_queue': ooq,
      'status_bit': status_bit,
      'target_host': "localhost",
      'listen_port': 8083,
      'send_port': 8088,
    }
  )
  p.start()

  a = foo(20)
  ser = pickle.dumps(a)
  ooq.put(ser)

  counter = 0
  while counter < 100:
    cur = iiq.get()
    obj = pickle.loads(cur)
    print(obj.x)
    time.sleep(0.1)
    ooq.put(pickle.dumps(obj.add(1)))
    counter += 1
  status_bit.value = False