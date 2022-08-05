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
import socket
import multiprocessing
import time

from absl import flags

from deeplearning.benchpress.util import logging as l

FLAGS = flags.FLAGS

flags.DEFINE_boolean(
  "use_socket_server",
  False,
  "Select to use socket server in the app. If you set to True, the app will know how to use it with respect to the requested task."
)

flags.DEFINE_string(
  "target_host",
  None,
  "Define IP Address of target socket server."
)

flags.DEFINE_integer(
  "listen_port",
  None,
  "Define port this current server listens to."
)

flags.DEFINE_integer(
  "send_port",
  None,
  "Define port this current server listens to."
)

MAX_PAYLOAD_SIZE = 65535

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

def start_server_process():
  """
  This is an easy wrapper to start server from parent routine.
  Starts a new process or thread and returns all the multiprocessing
  elements needed to control the server.
  """
  rq, wq = multiprocessing.Queue(), multiprocessing.Queue()
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

def start_thread_process():
  """
  This is an easy wrapper to start server from parent routine.
  Starts a new process or thread and returns all the multiprocessing
  elements needed to control the server.
  """
  rq, wq = multiprocessing.Queue(), multiprocessing.Queue()
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
