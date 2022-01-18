"""Cluster node handling for Distributed model training and sampling"""
import glob
import os
import time
import pathlib

from deeplearning.clgen.util import environment
from eupy.native import logger as l

MASTER_PORT = environment.MASTER_PORT
MASTER_ADDR = environment.MASTER_ADDR
LOCAL_RANK  = environment.LOCAL_RANK
WORLD_RANK  = environment.WORLD_RANK
WORLD_SIZE  = environment.WORLD_SIZE

PATH = None

def barrier() -> None:
  """
  Node processes are blocked until all nodes have reached this checkpoint.
  !!Warning!!: This function must not be called under a child process or thread.
  """
  if WORLD_SIZE > 1:
    if PATH is None:
      raise FileNotFoundError("Distributed env path has not been set!")
    with open(PATH / "barrier-{}".format(WORLD_RANK), 'w') as outf:
      outf.write("{}\n".format(WORLD_RANK))

    barriers = glob.glob(str(PATH / "barrier-*"))

    while len(barriers) < environment.WORLD_SIZE:
      time.sleep(2)
      barriers = glob.glob(str(PATH / "barrier-*"))

    while len(barriers) > 0:
      barriers = glob.glob(str(PATH / "barrier-*"))
      if environment.WORLD_RANK == 0:
        for b in barriers:
          os.remove(str(b))
      else:
        time.sleep(0.2)
  return

def init(path: pathlib.Path) -> None:
  """
  Initialize parent directory for distrib coordination.
  """
  global PATH
  if isinstance(path, str):
    PATH = pathlib.Path(path).resolve()
  else:
    PATH = path
  return
