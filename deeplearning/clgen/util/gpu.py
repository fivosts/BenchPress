"""Helper module for GPU system handling"""
import os
import subprocess

from eupy.native import logger as l

NVIDIA_SMI_GET_GPUS = "nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.total,memory.used,memory.free,driver_version,name,gpu_serial,display_active,display_mode,temperature.gpu --format=csv,noheader,nounits"

def _to_float_or_inf(value: str):
  """Util conversion string to float"""
  try:
    number = float(value)
  except ValueError:
    number = float("nan")
  return number

def getGPUs(smi_output):
  """
  Get all available GPU entries with information.
  """
  gpus = []
  for line in smi_output:
    if line.strip():
      values = line.split(", ")
      gpus.append({
        'id' : values[0],
        'uuid' : values[1],
        'gpu_util'      : _to_float_or_inf(values[2]),
        'mem_total'     : _to_float_or_inf(values[3]),
        'mem_used'      : _to_float_or_inf(values[4]),
        'mem_free'      : _to_float_or_inf(values[5]),
        'driver'        : values[6],
        'gpu_name'      : values[7],
        'serial'        : values[8],
        'display_active': values[9],
        'display_mode'  : values[10],
        'temp_gpu'      : _to_float_or_inf(values[11]),
      })
  return gpus

def getGPUID():
  """
  Get GPU entries and select the one with the most memory available.
  """
  output = subprocess.check_output(NVIDIA_SMI_GET_GPUS.split())
  gpus = getGPUs(output.decode("utf-8").split(os.linesep))
  if len(gpus) > 0:
    selected_gpus = sorted(gpus, key=lambda x: x['mem_used'])
    return selected_gpus
  else:
    return None