"""Cluster node handling for Distributed model training and sampling"""
import glob
import os
import time
import pathlib
import typing
import functools
import progressbar

from deeplearning.clgen.util import environment
from eupy.native import logger as l

MASTER_PORT = environment.MASTER_PORT
MASTER_ADDR = environment.MASTER_ADDR
LOCAL_RANK  = environment.LOCAL_RANK
WORLD_RANK  = environment.WORLD_RANK
WORLD_SIZE  = environment.WORLD_SIZE

PATH = None

def barrier(fn: typing.Callable = None) -> None:
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
      if WORLD_RANK == 0 and fn:
        fn()
      time.sleep(0.5)
      barriers = glob.glob(str(PATH / "barrier-*"))

    with open(PATH / "barrier-escape-{}".format(WORLD_RANK), 'w') as outf:
      outf.write("{}\n".format(WORLD_RANK))

    while len(barriers) > 0:
      barriers = glob.glob(str(PATH / "barrier-*"))
      escapes  = glob.glob(str(PATH / "barrier-escape-*"))
      if environment.WORLD_RANK == 0 and len(escapes) == environment.WORLD_SIZE:
        for be in escapes:
          os.remove(str(be))
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

class ProgressBar(object):
  """
  Creates a distributed progressbar.
  All nodes write their current index to a distinct file.
  Only master node reads the indices and updates the progressbar.
  """
  def __init__(self, max_value: int, offset: int):
    self.max_value = max_value
    self.offset    = offset
    self.path      = PATH
    if self.path is None:
      raise FileNotFoundError("Distributed env path has not been set!")
    if WORLD_RANK == 0:
      self.bar = progressbar.ProgressBar(max_value = max_value)
    return

  def _fetch_indices(self, idx: int) -> int:
    """
    Master node reads current workload indices of all nodes.
    """
    total = idx - self.offset
    for n in range(1, WORLD_SIZE):
      if (self.path / "index-{}".format(n)).exist():
        with open(self.path / "index-{}".format(n), 'r') as inf:
          total += int(inf.read())
    return total

  def _write_index(self, idx: int) -> None:
    """
    Update personal node dictionary with current index.
    """
    with open(self.path / "index-{}".format(WORLD_RANK), 'w') as outf:
      outf.write(idx - self.offset)
    return

  def update(idx: int) -> None:
    if WORLD_RANK == 0:
      total_idx = self._fetch_indices(idx)
      self.bar.update(total_idx)
    else:
      self._write_index(idx)
    return

  def finalize(self, idx: int) -> None:
    """
    Do a final bar update and cleanup progressbar object.
    """
    fn = functools.partial(self.update, idx = idx)
    barrier(fn)
    if WORLD_RANK == 0:
      indices = glob.glob(str(PATH / "index-*"))
      for ip in indices:
        os.remove(str(ip))
    return
