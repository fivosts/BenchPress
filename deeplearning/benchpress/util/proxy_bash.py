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
import typing
import subprocess
import threading

from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_boolean(
  "proxy_bash",
  False,
  "Set True to start a proxy bash thread.",
  "Commands are provided from BenchPress's",
  "running terminal and standard's input format",
  "must be: `>> CMD'."
)

def listen() -> None:
  """
  Listen for bash commands from standard input and execute using a subprocess PIPE.
  """
  while True:
    cmd = input()


def start_proxy_bash() -> None:
  """
  Initialize daemon thread to run your proxy bash commands.
  """
  th = threading.Thread(
    target = listen,
    daemon = True
  )
  th.start()
  return