# coding=utf-8
# Copyright 2023 Foivos Tsimpourlas.
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
Analysis of Human or AI experiment.
"""
import pathlib

from deeplearning.benchpress.experiments import public
from deeplearning.benchpress.experiments.turing import db
from deeplearning.benchpress.util import plotter

@public.evaluator
def HumanLikenessAnalysis(**kwargs) -> None:
  """
  Analyse Human or AI experiment.
  """
  data = kwargs.get("human_likeness_db")
  path = pathlib.Path(data).resolve()
  if not path.exists():
    raise FileNotFoundError(path)
  
  db = db.TuringDB(url = "sqlite:///{}".format(path), must_exist = True)

  """
  Perform a set of analysis.
  """

  return
