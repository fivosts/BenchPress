"""Cluster node handling for Distributed model training and sampling"""
import signal
import threading

from deeplearning.clgen.util import environment
from eupy.native import logger as l

MASTER_PORT = environment.MASTER_PORT
MASTER_ADDR = environment.MASTER_ADDR
LOCAL_RANK  = environment.LOCAL_RANK
WORLD_RANK  = environment.WORLD_RANK
WORLD_SIZE  = environment.WORLD_SIZE

GLOO_SOCKET_IFNAME = environment.GLOO_SOCKET_IFNAME
NCCL_SOCKET_IFNAME = environment.NCCL_SOCKET_IFNAME

EXIT = threading.Event()
EXIT.clear()

def _node_exit_handler(signum, frame) -> None:
  EXIT.set()

signal.signal(signal.SIGINT,  _node_exit_handler)
signal.signal(signal.SIGTERM, _node_exit_handler)
signal.signal(signal.SIGUSR2, _node_exit_handler)