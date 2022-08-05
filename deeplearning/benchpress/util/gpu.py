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
"""Helper module for GPU system handling"""
import os
import subprocess
import typing

from deeplearning.benchpress.util import logging as l

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
  try:
    output = subprocess.check_output(NVIDIA_SMI_GET_GPUS.split())
  except FileNotFoundError:
    return None
  gpus = getGPUs(output.decode("utf-8").split(os.linesep))
  if len(gpus) > 0:
    selected_gpus = sorted(gpus, key=lambda x: x['mem_used'])
    return selected_gpus
  else:
    return None

def memUsageByPID(pids: typing.Iterable[int]) -> int:
  """
  Get a python iterable (list, set, dict, tuple) of PIDs.

  Returns the total GPU memory allocation in MB.
  """
  try:
    output = subprocess.check_output("nvidia-smi pmon -c 1 -s m".split())
  except FileNotFoundError:
    return 0
  pid_list = [i.split() for i in output.decode('utf-8').split(os.linesep)[2:]]
  return sum([int(x[3]) for x in pid_list if x and x[1] != '-' and x[3] != '-' and int(x[1]) in pids])
