"""Cluster node handling for Distributed model training and sampling"""
import glob
import os
import time
import pathlib
import typing
import functools
import progressbar

from deeplearning.clgen.util import environment
from deeplearning.clgen.util import logging as l

MASTER_PORT = environment.MASTER_PORT
MASTER_ADDR = environment.MASTER_ADDR
LOCAL_RANK  = environment.LOCAL_RANK
WORLD_RANK  = environment.WORLD_RANK
WORLD_SIZE  = environment.WORLD_SIZE

PATH = None

LOCK_TYPES = [
  'barrier-lock-',
  'barrier-escape-',
  'critical-lock-',
  'index-',
  'msg*'
]

def barrier(fn: typing.Callable = None) -> None:
  """
  Node processes are blocked until all nodes have reached this checkpoint.
  !!Warning!!: This function must not be called under a child process or thread.
  """
  if WORLD_SIZE > 1:
    if PATH is None:
      raise FileNotFoundError("Distributed env path has not been set!")
    with open(PATH / "barrier-lock-{}".format(WORLD_RANK), 'w') as outf:
      outf.write("{}\n".format(WORLD_RANK))
      outf.flush()

    barriers = glob.glob(str(PATH / "barrier-lock-*"))

    while len(barriers) < environment.WORLD_SIZE:
      if WORLD_RANK == 0 and fn:
        fn()
      time.sleep(0.5)
      barriers = glob.glob(str(PATH / "barrier-lock-*"))

    with open(PATH / "barrier-escape-{}".format(WORLD_RANK), 'w') as outf:
      outf.write("{}\n".format(WORLD_RANK))
      outf.flush()

    while len(barriers) > 0:
      barriers = glob.glob(str(PATH / "barrier-lock-*"))
      escapes  = glob.glob(str(PATH / "barrier-escape-*"))
      if environment.WORLD_RANK == 0 and len(escapes) == environment.WORLD_SIZE:
        for be in escapes:
          os.remove(str(be))
        for b in barriers:
          os.remove(str(b))
      else:
        time.sleep(0.2)
    time.sleep(0.5)
  return

def lock() -> None:
  """
  Acquire lockfile to proceed to critical section.
  """
  ## Corner-case where no DDP is used.
  if WORLD_SIZE == 1:
    return
  ## Busy waiting to acquire lock.
  while len(glob.glob(str(PATH / "critical-lock-*"))) > 0:
    time.sleep(0.5)

  ## Register lockfile.
  if (PATH / "critical-lock-{}".format(WORLD_RANK)).exists():
    raise ValueError("Node {} lock already exists.".format(WORLD_RANK))
  with open(PATH / "critical-lock-{}".format(WORLD_RANK), 'w') as outf:
    outf.write("{}\n".format(WORLD_RANK))
    outf.flush()

  ## Maybe more than one processes are here already. Prioritize by id.
  ## Unlock and Re-lock if you are not the minimum privileged id.
  locks = glob.glob(str(PATH / "critical-lock-*"))
  if len(locks) > 1:
    min_id = min([int(x.split('critical-lock-')[-1]) for x in locks])
    if WORLD_RANK != min_id:
      unlock()
      lock()
  return

def unlock() -> None:
  """
  Release node lock.
  """
  if WORLD_SIZE == 1:
    return
  if not (PATH / "critical-lock-{}".format(WORLD_RANK)).exists():
    raise FileNotFoundError("Node {} lock missing.".format(WORLD_RANK))
  os.remove(PATH / "critical-lock-{}".format(WORLD_RANK))
  return

def write(msg: str) -> None:
  """
  Node broadcasts a message to all other nodes.
  This function is not process-safe. User must ensure one node calls it
  and all reads have been complete before re-writing.
  """
  for x in range(WORLD_SIZE):
    with open(PATH / "msg-{}".format(x), 'w') as outf:
      outf.write(msg)
      outf.flush()
  msg = read()
  while len(glob.glob(str(PATH / "msg-*"))) > 0:
    time.sleep(0.5)
  return

def read() -> str:
  """
  All nodes read broadcasted message.
  """
  while not (PATH / "msg-{}".format(WORLD_RANK)).exists():
    time.sleep(0.5)
  while True:
    with open(PATH / "msg-{}".format(WORLD_RANK), 'r') as inf:
      msg = inf.read()
    if msg != '':
      break
    time.sleep(0.5)
  os.remove(str(PATH / "msg-{}".format(WORLD_RANK)))
  return msg

def init(path: pathlib.Path) -> None:
  """
  Initialize parent directory for distrib coordination.
  """
  global PATH
  if isinstance(path, str):
    PATH = pathlib.Path(path).resolve()
  else:
    PATH = path
  cleanup()
  return

def cleanup() -> None:
  """
  Cleanup any distributed lock files used.
  """
  for tp in LOCK_TYPES:
    for f in glob.glob(str(PATH / "{}{}".format(tp, WORLD_RANK))):
      os.remove(f)
  barrier()
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
      if (self.path / "index-{}".format(n)).exists():
        with open(self.path / "index-{}".format(n), 'r') as inf:
          total += int(inf.read())
    return total

  def _write_index(self, idx: int) -> None:
    """
    Update personal node dictionary with current index.
    """
    with open(self.path / "index-{}".format(WORLD_RANK), 'w') as outf:
      outf.write(idx - self.offset)
      outf.flush()
    return

  def update(self, idx: int) -> None:
    """
    Master node updates the bar,
    slave nodes update their indices.
    """
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
