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
import multiprocessing

def isolate(process: callable, **kwargs) -> None:
  """
  Executes a callable in isolated process space by spawning a child process.
  After executing, memory, cpu and gpu resources will be freed.

  Handy in executing TF-graph functions that will not free memory after execution.
  Args:
    process: callable. Function to be executed.
    kwargs: See multiprocessing.Process docs for kwargs.
  """
  pr = multiprocessing.Process(target = process, **kwargs)
  pr.start()
  pr.join()
  return