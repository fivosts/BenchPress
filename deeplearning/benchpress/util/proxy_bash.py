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
"""
A tool that helps running bash commands using BenchPress as proxy.

Especially useful when deploying BenchPress on clusters but still
need to keep an eye on resources (e.g. nvidia-smi) or files.
"""
import subprocess
import threading

from deeplearning.benchpress.util import environment

def listen() -> None:
  """
  Listen for bash commands from standard input
  and execute using a subprocess PIPE.
  """
  while True:
    cmd = input()
    if cmd[:3] == ">> ":
      cmd = cmd[3:]
      try:
        pr = subprocess.Popen(cmd.split(), stdout = subprocess.PIPE, stderr = subprocess.PIPE)
        stdout, stderr = pr.communicate()
        print(stdout.decode('utf-8'))
        if stderr:
          print(stderr.decode('utf-8'))
      except FileNotFoundError:
        print("{}: command not found".format(cmd))
  return

def start() -> None:
  """
  Initialize daemon thread to run your proxy bash commands.
  """
  if environment.WORLD_RANK == 0:
    th = threading.Thread(
      target = listen,
      daemon = True
    )
    th.start()
  return
