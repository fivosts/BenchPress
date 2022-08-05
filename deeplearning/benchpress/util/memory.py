"""CPU and GPU memory usage monitor"""
import os
import pathlib
import psutil
import threading
import time
import typing

from deeplearning.benchpress.util import gpu
from deeplearning.benchpress.util import monitors

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
