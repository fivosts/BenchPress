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
Log file evaluation.
"""
from deeplearning.benchpress.experiments import public

@public.evaluator
def LogFile(**kwargs) -> None:
  """
  Write benchmarks  and target stats in log file.
  """
  db_groups     = kwargs.get('db_groups')
  target        = kwargs.get('targets')
  raise NotImplementedError
  return