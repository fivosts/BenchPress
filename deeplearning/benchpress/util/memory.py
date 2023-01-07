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
"""CPU and GPU memory usage monitor"""
import os
import humanize
import pathlib
import psutil
import threading
import time
import typing

from deeplearning.benchpress.util import gpu
from deeplearning.benchpress.util import monitors

def getRamUsage() -> typing.Dict[str, str]:
  """
  Return memory usage of current PID without its child processes.
  """
  process = psutil.Process(os.getpid())
  return {
    'dss': humanize.naturalsize(process.dss),
    'vms': humanize.naturalsize(process.vms),
    'shared': humanize.naturalsize(process.shared),
    'text': humanize.naturalsize(process.text),
    'lib': humanize.naturalsize(process.lib),
    'data': humanize.naturalsize(process.data),
    'dirty': humanize.naturalsize(process.dirty),
  }

def monRamUsage(path: pathlib.Path) -> None:
  ram_monitor = monitors.HistoryMonitor(
    path, "ram_usage"
  )
  main_process = psutil.Process(os.getpid())
  while True:
    try:
      total_mem = (main_process.memory_info().rss +
                      sum([p.memory_info().rss 
                          for p in main_process.children(recursive = True)]
                      )
                    )
    except psutil._exceptions.NoSuchProcess:
      total_mem = (main_process.memory_info().rss +
                      sum([p.memory_info().rss 
                          for p in main_process.children(recursive = True)]
                      )
                    )
    ram_monitor.register(total_mem / (1024**2)) # MB
    ram_monitor.plot()
    time.sleep(5)
  return

def monGPUsage(path: pathlib.Path) -> None:
  gpu_monitor = monitors.HistoryMonitor(
    path, "gpu_usage"
  )
  main_process = psutil.Process(os.getpid())
  while True:
    process_pids = [main_process.pid] + [p.pid for p in main_process.children(recursive = True)]
    total_mem = gpu.memUsageByPID(process_pids)
    gpu_monitor.register(total_mem) # MB
    gpu_monitor.plot()
    time.sleep(5)
  return

def init_mem_monitors(path: pathlib.Path) -> typing.Tuple[threading.Thread, threading.Thread]:
  cpu_thread = threading.Thread(target = monRamUsage, args = (path,))
  gpu_thread = threading.Thread(target = monGPUsage, args = (path,))

  cpu_thread.setDaemon(True)
  gpu_thread.setDaemon(True)

  cpu_thread.start()
  gpu_thread.start()
  return cpu_thread, gpu_thread
