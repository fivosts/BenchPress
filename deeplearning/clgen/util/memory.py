"""CPU and GPU memory usage monitor"""
import os
import psutil
import threading
import time
import typing

from deeplearning.clgen.util import gpu
from deeplearning.clgen.util import distributions

def monRamUsage() -> None:
  ram_monitor = distributions.TimestampMonitor(
    "/tmp/", "ram_usage"
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

def monGPUsage() -> None:
  gpu_monitor = distributions.TimestampMonitor(
    "/tmp/", "gpu_usage"
  )
  main_process = psutil.Process(os.getpid())
  while True:
    process_pids = [main_process.pid] + [p.pid for p in main_process.children(recursive = True)]
    total_mem = gpu.memUsageByPID(process_pids)
    gpu_monitor.register(total_mem) # MB
    gpu_monitor.plot()
    time.sleep(5)
  return

def init_mem_monitors() -> typing.Tuple[threading.Thread, threading.Thread]:
  cpu_thread = threading.Thread(target = monRamUsage)
  gpu_thread = threading.Thread(target = monGPUsage)

  cpu_thread.setDaemon(True)
  gpu_thread.setDaemon(True)

  cpu_thread.start()
  gpu_thread.start()
  return cpu_thread, gpu_thread
